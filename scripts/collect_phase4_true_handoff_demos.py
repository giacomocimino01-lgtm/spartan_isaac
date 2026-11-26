"""Collect Phase-4 demonstrations after a real learned Phase-3 handoff.

Each batch is prepared as follows::

    Phase-3 semantic reset (four rings at board centre)
      -> deterministic Phase-3 PPO rollout
      -> the same Cartesian ``park`` primitive used by eval_full_task.py
      -> in-place transition to Phase 4
      -> asynchronous per-arm scripted Phase-4 expert

The resulting state/action pairs cover the arm-timing states that are absent
from direct joint-reset datasets, in particular one arm transporting or
releasing while the other arm is still finishing its source reach/grasp.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T

from app_launcher_utils import pin_process_to_requested_cuda_device
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Collect Phase-4 BC data from real Phase-3 PPO handoffs."
)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=80)
parser.add_argument("--output_path", required=True)
parser.add_argument(
    "--entry_state_bank_output",
    default=None,
    help=(
        "Optional NPZ path for the physical robot joint states observed after "
        "the learned Phase-3 rollout and full-task park transition."
    ),
)
parser.add_argument("--phase3_checkpoint", required=True)
parser.add_argument(
    "--phase3_source",
    choices=("policy", "teacher"),
    default="policy",
    help=(
        "Produce Phase-4 entry states with the learned Phase-3 policy or "
        "with the scripted bimanual teacher used for Phase-3 BC."
    ),
)
parser.add_argument("--phase3_tcc_checkpoint", required=True)
parser.add_argument("--phase4_tcc_checkpoint", required=True)
parser.add_argument(
    "--dataset_root",
    default="/mnt/data/aiprah/data/random_sim_dataset_tcc/train",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_phase3_steps", type=int, default=700)
parser.add_argument("--max_phase4_steps", type=int, default=900)
parser.add_argument("--transition_park_max_steps", type=int, default=150)
parser.add_argument("--transition_settle_steps", type=int, default=3)
parser.add_argument(
    "--action_mask_mode",
    choices=("guided", "goal_agnostic", "minimal"),
    default="minimal",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True
args_cli.device = pin_process_to_requested_cuda_device(
    getattr(args_cli, "device", None)
)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from stable_baselines3 import PPO

from m_dVrk.controllers.ring_sync import sync_attached_and_frozen_rings
from m_dVrk.controllers.spartan_state_machine import SPARTANStateMachine
from m_dVrk.hrl.constants import (
    IDLE_ACTION,
    BOARD_CENTER,
    OUTSIDE_LEFT,
    OUTSIDE_RIGHT,
    PEG_BLUE,
    PEG_GREEN,
    PEG_RED,
    PEG_YELLOW,
    RING_NAMES,
    TARGET_TO_ID,
    VERB_TO_ID,
    target_id_for_phase,
)
from m_dVrk.hrl.expert_scheduling import (
    Phase3BimanualScheduler,
    Phase4BimanualScheduler,
)
from m_dVrk.hrl.masked_policy import masked_policy_custom_objects
from m_dVrk.hrl.tcc import compute_average_goal_embedding, load_tcc_model
from m_dVrk.hrl.wrapper import DVRKVisionHRLWrapper
from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg


def _set_wrapper_phase(
    wrapper: DVRKVisionHRLWrapper,
    phase: str,
    tcc_model: torch.nn.Module,
    goal_embedding: torch.Tensor,
) -> None:
    """Select the phase used by the next full wrapper reset."""
    wrapper.task_phase = phase
    wrapper.tcc_model = tcc_model
    wrapper.goal_embedding = goal_embedding


def _quiescent(sm: SPARTANStateMachine, env_id: int) -> bool:
    return (
        sm.all_idle(env_id)
        and sm.attached_target_l[env_id] is None
        and sm.attached_target_r[env_id] is None
    )


def _park_successful_envs(
    wrapper: DVRKVisionHRLWrapper,
    eligible: np.ndarray,
) -> tuple[int, bool]:
    """Apply the exact full-task park primitive to successful Phase-3 envs."""
    sm = wrapper.sm
    device = wrapper.env.device
    eligible_t = torch.as_tensor(eligible, dtype=torch.bool, device=device)

    for env_id, use_env in enumerate(eligible.tolist()):
        sm.reset_control_state(env_id)
        if use_env:
            sm.set_new_triplet("park", "right_arm", "None", env_id)
            sm.set_new_triplet("park", "left_arm", "None", env_id)

    steps = 0
    parked = np.logical_not(eligible.copy())
    for _ in range(args_cli.transition_park_max_steps):
        _, _, terminated, truncated, _ = wrapper.env.step(sm.get_action())
        steps += 1
        done = terminated.to(torch.bool) | truncated.to(torch.bool)
        sync_attached_and_frozen_rings(wrapper.env, sm, eligible_t & ~done)
        for env_id, use_env in enumerate(eligible.tolist()):
            if use_env and sm.all_idle(env_id):
                parked[env_id] = True
        if bool(np.all(parked)):
            break
    else:
        return steps, False

    for _ in range(args_cli.transition_settle_steps):
        _, _, terminated, truncated, _ = wrapper.env.step(sm.get_action())
        steps += 1
        done = terminated.to(torch.bool) | truncated.to(torch.bool)
        sync_attached_and_frozen_rings(wrapper.env, sm, eligible_t & ~done)
        if bool((done & eligible_t).any().item()):
            return steps, False
    return steps, True


def _phase4_scheduler(sm: SPARTANStateMachine, env_id: int) -> Phase4BimanualScheduler:
    """Build the same arm-relative expert as the native Phase-4 collector."""
    left_rings = [
        sm.get_top_rings_on_peg(env_id, peg, num_rings=1)[0]
        for peg in (PEG_RED, PEG_BLUE)
    ]
    right_rings = [
        sm.get_top_rings_on_peg(env_id, peg, num_rings=1)[0]
        for peg in (PEG_YELLOW, PEG_GREEN)
    ]
    return Phase4BimanualScheduler(
        (
            (left_ring, OUTSIDE_LEFT, right_ring, OUTSIDE_RIGHT)
            for left_ring, right_ring in zip(left_rings, right_rings, strict=True)
        )
    )


def _phase3_scheduler(sm: SPARTANStateMachine, env_id: int) -> Phase3BimanualScheduler:
    """Reproduce the bimanual teacher used by the Phase-3 BC collector."""
    rings = sm.get_top_rings_on_peg(env_id, BOARD_CENTER, num_rings=4)
    if len(rings) != 4:
        raise RuntimeError(
            f"Phase-3 bimanual expert expected 4 centre rings, got {rings}"
        )
    return Phase3BimanualScheduler(
        [
            (rings[0], PEG_RED, rings[1], PEG_GREEN),
            (rings[2], PEG_YELLOW, rings[3], PEG_BLUE),
        ]
    )


def _expert_action(
    sm: SPARTANStateMachine,
    scheduler: Phase3BimanualScheduler | Phase4BimanualScheduler,
    env_id: int,
    phase: str = "phase_4",
) -> tuple[np.ndarray, bool]:
    triplets = scheduler.issue_ready(
        left_idle=sm.arm_idle(env_id, "left_arm"),
        right_idle=sm.arm_idle(env_id, "right_arm"),
    )
    for verb, arm, target in triplets:
        sm.set_new_triplet(verb, arm, target, env_id)

    cmd_l = sm.current_triplet_l[env_id]
    cmd_r = sm.current_triplet_r[env_id]
    action = np.asarray(
        [
            VERB_TO_ID.get(cmd_l["verb"], VERB_TO_ID["idle"])
            if cmd_l
            else VERB_TO_ID["idle"],
            target_id_for_phase(cmd_l["target"], phase)
            if cmd_l
            else TARGET_TO_ID["None"],
            VERB_TO_ID.get(cmd_r["verb"], VERB_TO_ID["idle"])
            if cmd_r
            else VERB_TO_ID["idle"],
            target_id_for_phase(cmd_r["target"], phase)
            if cmd_r
            else TARGET_TO_ID["None"],
        ],
        dtype=np.int64,
    )
    return action, bool(triplets)


def _prepare_phase3_handoff_batch(
    wrapper: DVRKVisionHRLWrapper,
    phase3_policy: PPO,
    phase3_tcc: torch.nn.Module,
    phase3_goal: torch.Tensor,
    phase4_tcc: torch.nn.Module,
    phase4_goal: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Roll Phase 3 to completion, park, and return Phase-4 observations."""
    _set_wrapper_phase(wrapper, "phase_3", phase3_tcc, phase3_goal)
    obs = wrapper.reset()
    active = np.ones(wrapper.num_envs, dtype=bool)
    successful = np.zeros(wrapper.num_envs, dtype=bool)
    completion_steps = np.full(wrapper.num_envs, -1, dtype=np.int64)

    phase3_steps = 0
    while simulation_app.is_running() and active.any():
        actions, _ = phase3_policy.predict(obs, deterministic=True)
        actions = np.asarray(actions, dtype=np.int64)
        actions[~active] = IDLE_ACTION.cpu().numpy()
        obs, _, dones, infos = wrapper.step(actions)
        phase3_steps += 1

        for env_id in np.flatnonzero(active):
            if wrapper.sm.successful(int(env_id), phase="phase_3"):
                successful[env_id] = True
                completion_steps[env_id] = phase3_steps
                active[env_id] = False
            elif bool(dones[env_id]):
                active[env_id] = False

        if phase3_steps >= args_cli.max_phase3_steps:
            active[:] = False

    # Allow final release primitives to become idle before resetting controller
    # state for park, exactly as the quiescence guard does in full-task eval.
    for _ in range(80):
        pending = [
            env_id
            for env_id in np.flatnonzero(successful)
            if not _quiescent(wrapper.sm, int(env_id))
        ]
        if not pending:
            break
        idle = np.repeat(
            IDLE_ACTION.cpu().numpy()[None, :], wrapper.num_envs, axis=0
        )
        obs, _, _, _ = wrapper.step(idle)

    parked_steps, parked = _park_successful_envs(wrapper, successful)
    if not parked:
        successful[:] = False

    phase4_obs = wrapper.transition_to_phase(
        "phase_4", phase4_tcc, phase4_goal
    )
    stats = {
        "phase3_steps": int(phase3_steps),
        "phase3_successes": int(successful.sum()),
        "phase3_completion_steps": completion_steps.tolist(),
        "parking_steps": int(parked_steps),
    }
    return phase4_obs, successful, stats


def _prepare_phase3_teacher_handoff_batch(
    wrapper: DVRKVisionHRLWrapper,
    phase3_tcc: torch.nn.Module,
    phase3_goal: torch.Tensor,
    phase4_tcc: torch.nn.Module,
    phase4_goal: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Execute the original Phase-3 BC teacher, then park for Phase 4."""
    _set_wrapper_phase(wrapper, "phase_3", phase3_tcc, phase3_goal)
    obs = wrapper.reset()
    del obs
    schedulers = [
        _phase3_scheduler(wrapper.sm, env_id)
        for env_id in range(wrapper.num_envs)
    ]
    active = np.ones(wrapper.num_envs, dtype=bool)
    successful = np.zeros(wrapper.num_envs, dtype=bool)
    completion_steps = np.full(wrapper.num_envs, -1, dtype=np.int64)
    decision_counts = np.zeros(wrapper.num_envs, dtype=np.int64)

    phase3_steps = 0
    while simulation_app.is_running() and active.any():
        actions = np.repeat(
            IDLE_ACTION.cpu().numpy()[None, :], wrapper.num_envs, axis=0
        )
        for env_id in np.flatnonzero(active):
            action, issued = _expert_action(
                wrapper.sm,
                schedulers[int(env_id)],
                int(env_id),
                phase="phase_3",
            )
            actions[env_id] = action
            decision_counts[env_id] += int(issued)

        _, _, dones, _ = wrapper.step(actions)
        phase3_steps += 1
        for env_id in np.flatnonzero(active):
            if wrapper.sm.successful(int(env_id), phase="phase_3"):
                successful[env_id] = True
                completion_steps[env_id] = phase3_steps
                active[env_id] = False
            elif bool(dones[env_id]):
                active[env_id] = False

        if phase3_steps >= args_cli.max_phase3_steps:
            active[:] = False

    for _ in range(80):
        pending = [
            env_id
            for env_id in np.flatnonzero(successful)
            if not _quiescent(wrapper.sm, int(env_id))
        ]
        if not pending:
            break
        idle = np.repeat(
            IDLE_ACTION.cpu().numpy()[None, :], wrapper.num_envs, axis=0
        )
        wrapper.step(idle)

    parked_steps, parked = _park_successful_envs(wrapper, successful)
    if not parked:
        successful[:] = False

    phase4_obs = wrapper.transition_to_phase(
        "phase_4", phase4_tcc, phase4_goal
    )
    stats = {
        "phase3_source": "teacher",
        "phase3_steps": int(phase3_steps),
        "phase3_successes": int(successful.sum()),
        "phase3_completion_steps": completion_steps.tolist(),
        "phase3_decisions": decision_counts.tolist(),
        "parking_steps": int(parked_steps),
    }
    return phase4_obs, successful, stats


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.num_episodes <= 0:
        raise ValueError("num_envs and num_episodes must be positive")
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    env_cfg = MDvrkEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = 240.0
    env_cfg.num_rerenders_on_reset = 2
    env_cfg.wait_for_textures = False

    print(f"[handoff-collector] Creating {args_cli.num_envs} envs on {args_cli.device}")
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    sm = SPARTANStateMachine(isaac_env)
    preprocess = T.Compose([T.Resize((112, 112), antialias=True)])

    phase3_tcc = load_tcc_model(args_cli.phase3_tcc_checkpoint, isaac_env.device)
    phase4_tcc = load_tcc_model(args_cli.phase4_tcc_checkpoint, isaac_env.device)
    phase3_goal = compute_average_goal_embedding(
        phase3_tcc,
        preprocess,
        str(Path(args_cli.dataset_root) / "phase_3"),
        isaac_env.device,
    )
    phase4_goal = compute_average_goal_embedding(
        phase4_tcc,
        preprocess,
        str(Path(args_cli.dataset_root) / "phase_4"),
        isaac_env.device,
    )

    wrapper = DVRKVisionHRLWrapper(
        isaac_env,
        sm,
        phase3_tcc,
        task_phase="phase_3",
        action_mask_mode=args_cli.action_mask_mode,
        manipulability_curriculum_scale=1.0,
        auto_reset_on_success=False,
        reward_debug_dump_interval=0,
        success_snapshot_prob=0.0,
        success_video_prob=0.0,
    )
    wrapper.goal_embedding = phase3_goal
    phase3_policy: PPO | None = None
    if args_cli.phase3_source == "policy":
        phase3_policy = PPO.load(
            args_cli.phase3_checkpoint,
            device=args_cli.device,
            custom_objects=masked_policy_custom_objects(
                args_cli.phase3_checkpoint,
                "phase_3",
                args_cli.action_mask_mode,
            ),
        )

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_episode_ids: list[int] = []
    all_decision_indices: list[int] = []
    entry_joint_pos_right: list[np.ndarray] = []
    entry_joint_vel_right: list[np.ndarray] = []
    entry_joint_pos_left: list[np.ndarray] = []
    entry_joint_vel_left: list[np.ndarray] = []
    entry_tip_pos_right: list[np.ndarray] = []
    entry_tip_pos_left: list[np.ndarray] = []
    entry_ring_root_state_local: list[np.ndarray] = []
    entry_ring_frozen_mask: list[np.ndarray] = []
    entry_ring_support_peg: list[np.ndarray] = []
    entry_ring_stack_level: list[np.ndarray] = []
    handoff_stats: list[dict[str, object]] = []
    successful_episodes = 0
    batch_id = 0

    while simulation_app.is_running() and successful_episodes < args_cli.num_episodes:
        if args_cli.phase3_source == "teacher":
            obs, eligible, preparation = _prepare_phase3_teacher_handoff_batch(
                wrapper,
                phase3_tcc,
                phase3_goal,
                phase4_tcc,
                phase4_goal,
            )
        else:
            assert phase3_policy is not None
            obs, eligible, preparation = _prepare_phase3_handoff_batch(
                wrapper,
                phase3_policy,
                phase3_tcc,
                phase3_goal,
                phase4_tcc,
                phase4_goal,
            )
        batch_id += 1
        print(
            f"[handoff-collector] Batch {batch_id}: "
            f"Phase-3 success {int(eligible.sum())}/{args_cli.num_envs}, "
            f"park_steps={preparation['parking_steps']}",
            flush=True,
        )
        if not eligible.any():
            handoff_stats.append({"batch": batch_id, **preparation, "saved": 0})
            continue

        # Snapshot the true physical Phase-4 entry state before the expert
        # moves either arm.  These states are later used as empirical PPO
        # resets, so PPO experiences the same joint configurations as the
        # full-task Phase-3 -> park handoff rather than a synthetic ellipsoid.
        robot_right = isaac_env.scene["robot_right"]
        robot_left = isaac_env.scene["robot_left"]
        ring_root_state_local = torch.stack(
            [
                isaac_env.scene[ring_name].data.root_state_w.detach().clone()
                for ring_name in RING_NAMES
            ],
            dim=1,
        )
        ring_root_state_local[:, :, 0:3] -= isaac_env.scene.env_origins[:, None, :]
        ring_frozen_mask = torch.stack(
            [sm.frozen_rings_mask[ring_name].detach().clone() for ring_name in RING_NAMES],
            dim=1,
        )
        ring_stack_level = torch.stack(
            [sm.ring_stack_level[ring_name].detach().clone() for ring_name in RING_NAMES],
            dim=1,
        )
        ring_support_peg = np.asarray(
            [
                [
                    sm.ring_support_peg[ring_name][env_id] or "None"
                    for ring_name in RING_NAMES
                ]
                for env_id in range(args_cli.num_envs)
            ],
            dtype="<U32",
        )
        batch_entry = {
            "joint_pos_right": robot_right.data.joint_pos.detach().cpu().numpy().copy(),
            "joint_vel_right": robot_right.data.joint_vel.detach().cpu().numpy().copy(),
            "joint_pos_left": robot_left.data.joint_pos.detach().cpu().numpy().copy(),
            "joint_vel_left": robot_left.data.joint_vel.detach().cpu().numpy().copy(),
            "tip_pos_right": (
                robot_right.data.body_pos_w[:, sm.get_tip_idx_r()]
                - isaac_env.scene.env_origins
            ).detach().cpu().numpy().copy(),
            "tip_pos_left": (
                robot_left.data.body_pos_w[:, sm.get_tip_idx_l()]
                - isaac_env.scene.env_origins
            ).detach().cpu().numpy().copy(),
            "ring_root_state_local": ring_root_state_local.cpu().numpy().copy(),
            "ring_frozen_mask": ring_frozen_mask.cpu().numpy().copy(),
            "ring_support_peg": ring_support_peg,
            "ring_stack_level": ring_stack_level.cpu().numpy().copy(),
        }

        schedulers: list[Phase4BimanualScheduler | None] = []
        for env_id in range(args_cli.num_envs):
            schedulers.append(
                _phase4_scheduler(sm, env_id) if eligible[env_id] else None
            )
        active = eligible.copy()
        step_counts = np.zeros(args_cli.num_envs, dtype=np.int64)
        decision_counts = np.zeros(args_cli.num_envs, dtype=np.int64)
        obs_buffers: list[list[np.ndarray]] = [
            [] for _ in range(args_cli.num_envs)
        ]
        action_buffers: list[list[np.ndarray]] = [
            [] for _ in range(args_cli.num_envs)
        ]
        decision_buffers: list[list[int]] = [
            [] for _ in range(args_cli.num_envs)
        ]
        saved_this_batch = 0

        while simulation_app.is_running() and active.any():
            step_actions = np.repeat(
                IDLE_ACTION.cpu().numpy()[None, :], args_cli.num_envs, axis=0
            )
            for env_id in np.flatnonzero(active):
                scheduler = schedulers[int(env_id)]
                assert scheduler is not None
                action, issued = _expert_action(sm, scheduler, int(env_id))
                if issued:
                    decision_counts[env_id] += 1
                step_actions[env_id] = action
                obs_buffers[env_id].append(obs[env_id].copy())
                decision_buffers[env_id].append(
                    int(max(0, decision_counts[env_id] - 1))
                )

            next_obs, _, dones, infos = wrapper.step(step_actions)
            executed = wrapper.last_step_actions.cpu().numpy()
            for env_id in np.flatnonzero(active):
                action_buffers[env_id].append(executed[env_id].copy())
                step_counts[env_id] += 1
                success = bool(infos[env_id].get("phase_success", False)) or (
                    sm.successful(int(env_id), phase="phase_4")
                    and _quiescent(sm, int(env_id))
                )
                timed_out = step_counts[env_id] >= args_cli.max_phase4_steps
                if success:
                    if successful_episodes < args_cli.num_episodes:
                        episode_id = successful_episodes
                        successful_episodes += 1
                        all_obs.extend(obs_buffers[env_id])
                        all_actions.extend(action_buffers[env_id])
                        all_episode_ids.extend(
                            [episode_id] * len(obs_buffers[env_id])
                        )
                        all_decision_indices.extend(decision_buffers[env_id])
                        entry_joint_pos_right.append(
                            batch_entry["joint_pos_right"][env_id].copy()
                        )
                        entry_joint_vel_right.append(
                            batch_entry["joint_vel_right"][env_id].copy()
                        )
                        entry_joint_pos_left.append(
                            batch_entry["joint_pos_left"][env_id].copy()
                        )
                        entry_joint_vel_left.append(
                            batch_entry["joint_vel_left"][env_id].copy()
                        )
                        entry_tip_pos_right.append(
                            batch_entry["tip_pos_right"][env_id].copy()
                        )
                        entry_tip_pos_left.append(
                            batch_entry["tip_pos_left"][env_id].copy()
                        )
                        entry_ring_root_state_local.append(
                            batch_entry["ring_root_state_local"][env_id].copy()
                        )
                        entry_ring_frozen_mask.append(
                            batch_entry["ring_frozen_mask"][env_id].copy()
                        )
                        entry_ring_support_peg.append(
                            batch_entry["ring_support_peg"][env_id].copy()
                        )
                        entry_ring_stack_level.append(
                            batch_entry["ring_stack_level"][env_id].copy()
                        )
                        saved_this_batch += 1
                        print(
                            f"[handoff-collector] Phase-4 SUCCESS env={env_id} "
                            f"steps={step_counts[env_id]} decisions="
                            f"{decision_counts[env_id]} progress="
                            f"{successful_episodes}/{args_cli.num_episodes}",
                            flush=True,
                        )
                    active[env_id] = False
                elif bool(dones[env_id]) or timed_out:
                    print(
                        f"[handoff-collector] Phase-4 discard env={env_id} "
                        f"steps={step_counts[env_id]} done={bool(dones[env_id])}",
                        flush=True,
                    )
                    active[env_id] = False
            obs = next_obs

        handoff_stats.append(
            {"batch": batch_id, **preparation, "saved": saved_this_batch}
        )

    if not all_obs:
        raise RuntimeError("No successful true-handoff demonstrations collected")

    output = Path(args_cli.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    obs_arr = np.asarray(all_obs, dtype=np.float32)
    actions_arr = np.asarray(all_actions, dtype=np.int64)
    episode_ids_arr = np.asarray(all_episode_ids, dtype=np.int64)
    decision_indices_arr = np.asarray(all_decision_indices, dtype=np.int64)
    if not (
        len(obs_arr)
        == len(actions_arr)
        == len(episode_ids_arr)
        == len(decision_indices_arr)
    ):
        raise RuntimeError("Collected dataset arrays are not aligned")

    metadata = {
        "obs_dim": int(obs_arr.shape[1]),
        "emb_dim": int((obs_arr.shape[1] - 62 - 86) // 3),
        "aux_dim": 62,
        "geom_dim": 86,
        "task_phase": "phase_4",
        "action_mask_mode": args_cli.action_mask_mode,
        "manipulability_scale": 1.0,
        "entry_mode": f"{args_cli.phase3_source}_phase3_then_full_task_park",
        "phase3_source": args_cli.phase3_source,
        "phase3_checkpoint": os.path.abspath(args_cli.phase3_checkpoint),
        "phase3_tcc_checkpoint": os.path.abspath(args_cli.phase3_tcc_checkpoint),
        "phase4_tcc_checkpoint": os.path.abspath(args_cli.phase4_tcc_checkpoint),
        "phase4_strategy": "independent_bimanual",
        "record_mode": "all_steps",
        "seed": int(args_cli.seed),
        "successful_episodes": int(successful_episodes),
        "handoff_stats": handoff_stats,
    }
    np.savez(
        output,
        obs=obs_arr,
        actions=actions_arr,
        episode_ids=episode_ids_arr,
        decision_indices=decision_indices_arr,
        metadata=metadata,
    )
    if args_cli.entry_state_bank_output is not None:
        bank_output = Path(args_cli.entry_state_bank_output)
        bank_output.parent.mkdir(parents=True, exist_ok=True)
        bank_metadata = {
            "task_phase": "phase_4",
            "entry_mode": f"{args_cli.phase3_source}_phase3_then_full_task_park",
            "phase3_source": args_cli.phase3_source,
            "seed": int(args_cli.seed),
            "states": int(len(entry_joint_pos_right)),
            "phase3_checkpoint": os.path.abspath(args_cli.phase3_checkpoint),
        }
        np.savez(
            bank_output,
            joint_pos_right=np.asarray(entry_joint_pos_right, dtype=np.float32),
            joint_vel_right=np.asarray(entry_joint_vel_right, dtype=np.float32),
            joint_pos_left=np.asarray(entry_joint_pos_left, dtype=np.float32),
            joint_vel_left=np.asarray(entry_joint_vel_left, dtype=np.float32),
            tip_pos_right=np.asarray(entry_tip_pos_right, dtype=np.float32),
            tip_pos_left=np.asarray(entry_tip_pos_left, dtype=np.float32),
            ring_root_state_local=np.asarray(
                entry_ring_root_state_local, dtype=np.float32
            ),
            ring_frozen_mask=np.asarray(entry_ring_frozen_mask, dtype=np.bool_),
            ring_support_peg=np.asarray(entry_ring_support_peg, dtype="<U32"),
            ring_stack_level=np.asarray(entry_ring_stack_level, dtype=np.int64),
            ring_names=np.asarray(RING_NAMES),
            joint_names_right=np.asarray(robot_right.joint_names),
            joint_names_left=np.asarray(robot_left.joint_names),
            metadata=bank_metadata,
        )
        print(
            f"[handoff-collector] Saved {len(entry_joint_pos_right)} empirical "
            f"Phase-4 entry states to {bank_output}",
            flush=True,
        )
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"[handoff-collector] Saved {successful_episodes} episodes / "
        f"{len(obs_arr)} transitions to {output}",
        flush=True,
    )

    wrapper.close()
    isaac_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
