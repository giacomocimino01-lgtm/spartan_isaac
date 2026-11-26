"""Collect Phase-2 BC demonstrations from genuine Phase-1 terminal states.

The collector deliberately enumerates the Phase-1 layout space instead of
sampling it implicitly:

* all 24 bottom-to-top orders of the four rings on the green peg;
* all 6 valid red/blue destination schedules containing two placements each.

For every case a serial Phase-1 teacher completes the 2-red/2-blue goal, both
arms execute the same park primitive used by ``eval_full_task.py``, and an
asynchronous bimanual Phase-2 teacher moves the two stacks to board centre.
Only successful Phase-2 episodes are written to the dense BC dataset.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T

from app_launcher_utils import pin_process_to_requested_cuda_device
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Collect Phase-2 BC data after exhaustive Phase-1 handoffs."
)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_episodes", type=int, default=160)
parser.add_argument("--output_path", required=True)
parser.add_argument("--phase1_tcc_checkpoint", required=True)
parser.add_argument("--phase2_tcc_checkpoint", required=True)
parser.add_argument(
    "--dataset_root",
    default="/mnt/data/aiprah/data/random_sim_dataset_tcc/train",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_phase1_steps", type=int, default=900)
parser.add_argument("--max_phase2_steps", type=int, default=900)
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

from m_dVrk.controllers.ring_sync import sync_attached_and_frozen_rings
from m_dVrk.controllers.spartan_state_machine import SPARTANStateMachine
from m_dVrk.hrl.constants import (
    BOARD_CENTER,
    IDLE_ACTION,
    PEG_BLUE,
    PEG_GREEN,
    PEG_RED,
    RING_NAMES,
    TARGET_TO_ID,
    VERB_TO_ID,
    target_id_for_phase,
)
from m_dVrk.hrl.expert_scheduling import Phase2BimanualScheduler
from m_dVrk.hrl.tcc import compute_average_goal_embedding, load_tcc_model
from m_dVrk.hrl.wrapper import DVRKVisionHRLWrapper
from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg


Command = tuple[str, str, str]
LayoutCase = tuple[tuple[str, ...], tuple[str, ...]]


def _set_wrapper_phase(
    wrapper: DVRKVisionHRLWrapper,
    phase: str,
    tcc_model: torch.nn.Module,
    goal_embedding: torch.Tensor,
) -> None:
    wrapper.task_phase = phase
    wrapper.tcc_model = tcc_model
    wrapper.goal_embedding = goal_embedding


def _quiescent(sm: SPARTANStateMachine, env_id: int) -> bool:
    return (
        sm.all_idle(env_id)
        and sm.attached_target_l[env_id] is None
        and sm.attached_target_r[env_id] is None
    )


def _encode_current_action(
    sm: SPARTANStateMachine,
    env_id: int,
    phase: str,
) -> np.ndarray:
    cmd_l = sm.current_triplet_l[env_id]
    cmd_r = sm.current_triplet_r[env_id]
    return np.asarray(
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


def _park_successful_envs(
    wrapper: DVRKVisionHRLWrapper,
    eligible: np.ndarray,
) -> tuple[int, bool]:
    """Apply the exact Phase-transition park primitive used at inference."""
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


def _layout_cases(seed: int) -> list[LayoutCase]:
    orders = list(itertools.permutations(RING_NAMES))
    destinations = sorted(
        set(itertools.permutations((PEG_RED, PEG_RED, PEG_BLUE, PEG_BLUE)))
    )
    cases = [(tuple(order), tuple(targets)) for order in orders for targets in destinations]
    random.Random(seed).shuffle(cases)
    return cases


def _phase1_commands(
    sm: SPARTANStateMachine,
    env_id: int,
    destinations: tuple[str, ...],
) -> deque[Command]:
    rings = sm.get_top_rings_on_peg(env_id, PEG_GREEN, num_rings=4)
    if len(rings) != 4:
        raise RuntimeError(
            f"Phase-1 teacher expected four green-stack rings, got {rings}"
        )
    commands: deque[Command] = deque()
    for ring_name, destination in zip(rings, destinations, strict=True):
        # These are demonstrated strategies, not action-mask constraints.
        # Destination-relative arms make the source rollout reliable while
        # still exercising both arms before the common park transition.
        arm = "left_arm" if destination == PEG_RED else "right_arm"
        commands.extend(
            (
                ("reach", arm, ring_name),
                ("grasp", arm, ring_name),
                ("reach", arm, destination),
                ("release", arm, destination),
            )
        )
    return commands


def _phase2_scheduler(
    sm: SPARTANStateMachine,
    env_id: int,
) -> Phase2BimanualScheduler:
    red_rings = sm.get_top_rings_on_peg(env_id, PEG_RED, num_rings=4)
    blue_rings = sm.get_top_rings_on_peg(env_id, PEG_BLUE, num_rings=4)
    if len(red_rings) != 2 or len(blue_rings) != 2:
        raise RuntimeError(
            "Phase-2 teacher requires a 2-red/2-blue terminal layout, got "
            f"red={red_rings}, blue={blue_rings}"
        )
    return Phase2BimanualScheduler(zip(red_rings, blue_rings, strict=True))


def _prepare_phase1_handoff_batch(
    wrapper: DVRKVisionHRLWrapper,
    phase1_tcc: torch.nn.Module,
    phase1_goal: torch.Tensor,
    phase2_tcc: torch.nn.Module,
    phase2_goal: torch.Tensor,
    cases: list[LayoutCase],
    case_cursor: int,
) -> tuple[np.ndarray, np.ndarray, list[LayoutCase], dict[str, object]]:
    """Complete enumerated Phase-1 cases, park, and enter Phase 2 in place."""
    _set_wrapper_phase(wrapper, "phase_1", phase1_tcc, phase1_goal)
    wrapper.reset()

    batch_cases = [
        cases[(case_cursor + env_id) % len(cases)]
        for env_id in range(wrapper.num_envs)
    ]
    for env_id, (ring_order, _) in enumerate(batch_cases):
        wrapper.sm.reset_env(
            env_id,
            phase="phase_1",
            phase1_initial_state="stack_green",
            phase1_ring_order=ring_order,
        )
        wrapper.episode_initial_phase1_layout[env_id] = (
            wrapper.sm.describe_phase1_layout(env_id)
        )

    active_t = torch.ones(
        wrapper.num_envs, dtype=torch.bool, device=wrapper.env.device
    )
    sync_attached_and_frozen_rings(wrapper.env, wrapper.sm, active_t)
    for _ in range(2):
        wrapper.env.step(wrapper.sm.get_action())
        sync_attached_and_frozen_rings(wrapper.env, wrapper.sm, active_t)

    command_queues = [
        _phase1_commands(wrapper.sm, env_id, destinations)
        for env_id, (_, destinations) in enumerate(batch_cases)
    ]
    active = np.ones(wrapper.num_envs, dtype=bool)
    successful = np.zeros(wrapper.num_envs, dtype=bool)
    completion_steps = np.full(wrapper.num_envs, -1, dtype=np.int64)
    decision_counts = np.zeros(wrapper.num_envs, dtype=np.int64)
    terminal_layouts = ["" for _ in range(wrapper.num_envs)]

    phase1_steps = 0
    while simulation_app.is_running() and active.any():
        actions = np.repeat(
            IDLE_ACTION.cpu().numpy()[None, :], wrapper.num_envs, axis=0
        )
        for env_id in np.flatnonzero(active):
            queue = command_queues[int(env_id)]
            if wrapper.sm.all_idle(int(env_id)) and queue:
                verb, arm, target = queue.popleft()
                wrapper.sm.set_new_triplet(verb, arm, target, int(env_id))
                decision_counts[env_id] += 1
            actions[env_id] = _encode_current_action(
                wrapper.sm, int(env_id), "phase_1"
            )

        _, _, dones, _ = wrapper.step(actions)
        phase1_steps += 1
        for env_id in np.flatnonzero(active):
            if wrapper.sm.successful(int(env_id), phase="phase_1"):
                successful[env_id] = True
                completion_steps[env_id] = phase1_steps
                terminal_layouts[env_id] = wrapper.sm.describe_phase1_layout(
                    int(env_id)
                )
                active[env_id] = False
            elif bool(dones[env_id]):
                active[env_id] = False
        if phase1_steps >= args_cli.max_phase1_steps:
            active[:] = False

    # Do not truncate a final release before parking the arms.
    for _ in range(80):
        pending = [
            int(env_id)
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

    phase2_obs = wrapper.transition_to_phase("phase_2", phase2_tcc, phase2_goal)
    stats = {
        "phase1_steps": int(phase1_steps),
        "phase1_successes": int(successful.sum()),
        "phase1_completion_steps": completion_steps.tolist(),
        "phase1_decisions": decision_counts.tolist(),
        "phase1_terminal_layouts": terminal_layouts,
        "parking_steps": int(parked_steps),
    }
    return phase2_obs, successful, batch_cases, stats


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

    print(
        f"[phase12-collector] Creating {args_cli.num_envs} envs on "
        f"{args_cli.device}", flush=True
    )
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    sm = SPARTANStateMachine(isaac_env)
    preprocess = T.Compose([T.Resize((112, 112), antialias=True)])

    phase1_tcc = load_tcc_model(args_cli.phase1_tcc_checkpoint, isaac_env.device)
    phase2_tcc = load_tcc_model(args_cli.phase2_tcc_checkpoint, isaac_env.device)
    phase1_goal = compute_average_goal_embedding(
        phase1_tcc,
        preprocess,
        str(Path(args_cli.dataset_root) / "phase_1"),
        isaac_env.device,
    )
    phase2_goal = compute_average_goal_embedding(
        phase2_tcc,
        preprocess,
        str(Path(args_cli.dataset_root) / "phase_2"),
        isaac_env.device,
    )

    wrapper = DVRKVisionHRLWrapper(
        isaac_env,
        sm,
        phase1_tcc,
        task_phase="phase_1",
        phase1_initial_state="stack_green",
        action_mask_mode=args_cli.action_mask_mode,
        manipulability_curriculum_scale=1.0,
        auto_reset_on_success=False,
        reward_debug_dump_interval=0,
        success_snapshot_prob=0.0,
        success_video_prob=0.0,
    )
    wrapper.goal_embedding = phase1_goal

    cases = _layout_cases(args_cli.seed)
    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_episode_ids: list[int] = []
    all_decision_indices: list[int] = []
    saved_orders: list[tuple[str, ...]] = []
    saved_destinations: list[tuple[str, ...]] = []
    saved_terminal_layouts: list[str] = []
    handoff_stats: list[dict[str, object]] = []
    successful_episodes = 0
    case_cursor = 0
    batch_id = 0

    while simulation_app.is_running() and successful_episodes < args_cli.num_episodes:
        obs, eligible, batch_cases, preparation = _prepare_phase1_handoff_batch(
            wrapper,
            phase1_tcc,
            phase1_goal,
            phase2_tcc,
            phase2_goal,
            cases,
            case_cursor,
        )
        case_cursor += args_cli.num_envs
        batch_id += 1
        print(
            f"[phase12-collector] Batch {batch_id}: Phase-1 success "
            f"{int(eligible.sum())}/{args_cli.num_envs}, "
            f"park_steps={preparation['parking_steps']}",
            flush=True,
        )
        if not eligible.any():
            handoff_stats.append({"batch": batch_id, **preparation, "saved": 0})
            continue

        schedulers: list[Phase2BimanualScheduler | None] = [
            _phase2_scheduler(sm, env_id) if eligible[env_id] else None
            for env_id in range(args_cli.num_envs)
        ]
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
                triplets = scheduler.issue_ready(
                    left_idle=sm.arm_idle(int(env_id), "left_arm"),
                    right_idle=sm.arm_idle(int(env_id), "right_arm"),
                )
                for verb, arm, target in triplets:
                    sm.set_new_triplet(verb, arm, target, int(env_id))
                decision_counts[env_id] += int(bool(triplets))
                step_actions[env_id] = _encode_current_action(
                    sm, int(env_id), "phase_2"
                )
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
                    sm.successful(int(env_id), phase="phase_2")
                    and _quiescent(sm, int(env_id))
                )
                timed_out = step_counts[env_id] >= args_cli.max_phase2_steps
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
                        ring_order, destinations = batch_cases[int(env_id)]
                        saved_orders.append(ring_order)
                        saved_destinations.append(destinations)
                        saved_terminal_layouts.append(
                            str(preparation["phase1_terminal_layouts"][env_id])
                        )
                        saved_this_batch += 1
                        print(
                            f"[phase12-collector] Phase-2 SUCCESS env={env_id} "
                            f"steps={step_counts[env_id]} decisions="
                            f"{decision_counts[env_id]} progress="
                            f"{successful_episodes}/{args_cli.num_episodes}",
                            flush=True,
                        )
                    active[env_id] = False
                elif bool(dones[env_id]) or timed_out:
                    print(
                        f"[phase12-collector] Phase-2 discard env={env_id} "
                        f"steps={step_counts[env_id]} done={bool(dones[env_id])}",
                        flush=True,
                    )
                    active[env_id] = False
            obs = next_obs

        handoff_stats.append(
            {"batch": batch_id, **preparation, "saved": saved_this_batch}
        )

    if not all_obs:
        raise RuntimeError("No successful Phase-1-to-Phase-2 demonstrations collected")

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

    unique_terminal_layouts = sorted(set(saved_terminal_layouts))
    metadata = {
        "obs_dim": int(obs_arr.shape[1]),
        "emb_dim": int((obs_arr.shape[1] - 62 - 86) // 3),
        "aux_dim": 62,
        "geom_dim": 86,
        "task_phase": "phase_2",
        "action_mask_mode": args_cli.action_mask_mode,
        "manipulability_scale": 1.0,
        "entry_mode": "enumerated_phase1_teacher_then_full_task_park",
        "phase1_initial_state": "stack_green",
        "layout_space_size": len(cases),
        "unique_terminal_layouts": len(unique_terminal_layouts),
        "phase2_strategy": "asynchronous_bimanual",
        "record_mode": "all_steps",
        "seed": int(args_cli.seed),
        "successful_episodes": int(successful_episodes),
        "phase1_tcc_checkpoint": str(Path(args_cli.phase1_tcc_checkpoint).resolve()),
        "phase2_tcc_checkpoint": str(Path(args_cli.phase2_tcc_checkpoint).resolve()),
        "handoff_stats": handoff_stats,
    }
    np.savez(
        output,
        obs=obs_arr,
        actions=actions_arr,
        episode_ids=episode_ids_arr,
        decision_indices=decision_indices_arr,
        phase1_ring_orders=np.asarray(saved_orders, dtype="<U32"),
        phase1_destination_patterns=np.asarray(saved_destinations, dtype="<U32"),
        phase1_terminal_layouts=np.asarray(saved_terminal_layouts, dtype="<U256"),
        metadata=metadata,
    )
    summary_path = output.with_suffix(".summary.json")
    summary = {
        **metadata,
        "transitions": int(len(obs_arr)),
        "terminal_layouts": unique_terminal_layouts,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[phase12-collector] Saved {successful_episodes} episodes / "
        f"{len(obs_arr)} transitions / {len(unique_terminal_layouts)} unique "
        f"terminal layouts to {output}",
        flush=True,
    )

    wrapper.close()
    isaac_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
