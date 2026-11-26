"""Collect genuine Phase-2 reach-timeout states for an RL recovery curriculum.

The deterministic policy is run from the normal parked Phase-2 distribution.
Whenever the state machine reports a reach timeout, snapshots are taken along
the ensuing physical recovery/correction trajectory.  No expert action is
stored: this bank is only an initial-state distribution for subsequent PPO.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T

from app_launcher_utils import pin_process_to_requested_cuda_device
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--tcc_checkpoint", required=True)
parser.add_argument("--dataset_path", required=True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--target_states", type=int, default=64)
parser.add_argument("--max_episodes", type=int, default=500)
parser.add_argument(
    "--max_collection_steps",
    type=int,
    default=0,
    help=(
        "Global vector steps before giving up. Zero derives a conservative "
        "limit from max_episodes and num_envs."
    ),
)
parser.add_argument("--snapshot_stride", type=int, default=5)
parser.add_argument("--max_states_per_failure", type=int, default=24)
parser.add_argument(
    "--late_state_start",
    type=int,
    default=350,
    help=(
        "Also capture an attached-ring state after this many episode steps. "
        "Nominal successes finish earlier; this catches policy stalls that do "
        "not raise a low-level reach-timeout flag."
    ),
)
parser.add_argument("--seed", type=int, default=20261020)
parser.add_argument(
    "--stochastic_actions",
    action="store_true",
    help=(
        "Sample the BC distribution to expose nearby physical failure states. "
        "The bank still stores no action labels."
    ),
)
parser.add_argument(
    "--action_mask_mode",
    choices=("guided", "goal_agnostic", "minimal"),
    default="minimal",
)
parser.add_argument(
    "--phase2_ring_order",
    nargs=4,
    default=None,
    metavar=("RED_BOTTOM", "RED_TOP", "BLUE_BOTTOM", "BLUE_TOP"),
    help="Optional fixed bottom-to-top order; default shuffles every reset.",
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

from m_dVrk.controllers.spartan_state_machine import SPARTANStateMachine
from m_dVrk.hrl.constants import RING_NAMES, apply_phase_entry_park_joint_pose
from m_dVrk.hrl.masked_policy import masked_policy_custom_objects
from m_dVrk.hrl.tcc import compute_average_goal_embedding, load_tcc_model
from m_dVrk.hrl.wrapper import DVRKVisionHRLWrapper
from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg


def _attachment_name(value: str | None) -> str:
    return "None" if value is None else str(value)


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.target_states <= 0:
        raise ValueError("num_envs and target_states must be positive")
    if args_cli.max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if (
        args_cli.snapshot_stride <= 0
        or args_cli.max_states_per_failure <= 0
        or args_cli.late_state_start <= 0
    ):
        raise ValueError("snapshot controls must be positive")
    if args_cli.phase2_ring_order is not None:
        if set(args_cli.phase2_ring_order) != set(RING_NAMES):
            raise ValueError("--phase2_ring_order must contain every ring once")

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
    apply_phase_entry_park_joint_pose(env_cfg)

    print(
        f"[recovery-bank] Creating {args_cli.num_envs} Phase-2 envs on "
        f"{args_cli.device}",
        flush=True,
    )
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    isaac_env.phase_entry_pose_scale = 1.0
    sm = SPARTANStateMachine(isaac_env)
    tcc = load_tcc_model(args_cli.tcc_checkpoint, isaac_env.device)
    preprocess = T.Compose([T.Resize((112, 112), antialias=True)])
    goal = compute_average_goal_embedding(
        tcc,
        preprocess,
        args_cli.dataset_path,
        isaac_env.device,
    )
    wrapper = DVRKVisionHRLWrapper(
        isaac_env,
        sm,
        tcc,
        task_phase="phase_2",
        action_mask_mode=args_cli.action_mask_mode,
        phase2_initial_state="red2_blue2",
        phase2_ring_order=args_cli.phase2_ring_order,
        reward_debug_dump_interval=0,
        success_snapshot_prob=0.0,
        success_video_prob=0.0,
    )
    wrapper.goal_embedding = goal
    policy = PPO.load(
        args_cli.checkpoint,
        env=wrapper,
        device=args_cli.device,
        custom_objects=masked_policy_custom_objects(
            args_cli.checkpoint,
            "phase_2",
            args_cli.action_mask_mode,
        ),
    )

    robot_right = isaac_env.scene["robot_right"]
    robot_left = isaac_env.scene["robot_left"]
    saved: dict[str, list] = {
        "joint_pos_right": [],
        "joint_vel_right": [],
        "joint_pos_left": [],
        "joint_vel_left": [],
        "tip_pos_right": [],
        "tip_pos_left": [],
        "ring_root_state_local": [],
        "ring_frozen_mask": [],
        "ring_support_peg": [],
        "ring_stack_level": [],
        "attached_target_right": [],
        "attached_target_left": [],
        "failed_arm": [],
        "episode_index": [],
        "episode_step": [],
    }
    failure_capture_count = np.zeros(args_cli.num_envs, dtype=np.int64)
    steps_since_capture = np.zeros(args_cli.num_envs, dtype=np.int64)
    episode_steps = np.zeros(args_cli.num_envs, dtype=np.int64)
    completed_episodes = 0
    observed_failures = 0
    collection_steps = 0
    max_collection_steps = args_cli.max_collection_steps or (
        math.ceil(args_cli.max_episodes / args_cli.num_envs) * 900
    )

    def snapshot(env_id: int, failed_arm: str) -> None:
        ring_state = torch.stack(
            [
                isaac_env.scene[name].data.root_state_w[env_id].detach().clone()
                for name in RING_NAMES
            ],
            dim=0,
        )
        ring_state[:, 0:3] -= isaac_env.scene.env_origins[env_id]
        saved["joint_pos_right"].append(
            robot_right.data.joint_pos[env_id].detach().cpu().numpy().copy()
        )
        saved["joint_vel_right"].append(
            robot_right.data.joint_vel[env_id].detach().cpu().numpy().copy()
        )
        saved["joint_pos_left"].append(
            robot_left.data.joint_pos[env_id].detach().cpu().numpy().copy()
        )
        saved["joint_vel_left"].append(
            robot_left.data.joint_vel[env_id].detach().cpu().numpy().copy()
        )
        saved["tip_pos_right"].append(
            (
                robot_right.data.body_pos_w[env_id, sm.get_tip_idx_r()]
                - isaac_env.scene.env_origins[env_id]
            ).detach().cpu().numpy().copy()
        )
        saved["tip_pos_left"].append(
            (
                robot_left.data.body_pos_w[env_id, sm.get_tip_idx_l()]
                - isaac_env.scene.env_origins[env_id]
            ).detach().cpu().numpy().copy()
        )
        saved["ring_root_state_local"].append(ring_state.cpu().numpy().copy())
        saved["ring_frozen_mask"].append(
            np.asarray(
                [bool(sm.frozen_rings_mask[name][env_id].item()) for name in RING_NAMES],
                dtype=np.bool_,
            )
        )
        saved["ring_support_peg"].append(
            np.asarray(
                [sm.ring_support_peg[name][env_id] or "None" for name in RING_NAMES],
                dtype="<U32",
            )
        )
        saved["ring_stack_level"].append(
            np.asarray(
                [int(sm.ring_stack_level[name][env_id].item()) for name in RING_NAMES],
                dtype=np.int64,
            )
        )
        saved["attached_target_right"].append(
            _attachment_name(sm.attached_target_r[env_id])
        )
        saved["attached_target_left"].append(
            _attachment_name(sm.attached_target_l[env_id])
        )
        saved["failed_arm"].append(failed_arm)
        saved["episode_index"].append(completed_episodes + env_id)
        saved["episode_step"].append(int(episode_steps[env_id]))

    obs = wrapper.reset()
    while (
        simulation_app.is_running()
        and len(saved["joint_pos_right"]) < args_cli.target_states
        and collection_steps < max_collection_steps
    ):
        collection_steps += 1
        actions, _ = policy.predict(
            obs,
            deterministic=not args_cli.stochastic_actions,
        )
        obs, _, dones, infos = wrapper.step(actions)
        episode_steps += 1
        for env_id in range(args_cli.num_envs):
            failed_r = bool(sm.last_reach_failed_r[env_id])
            failed_l = bool(sm.last_reach_failed_l[env_id])
            has_attachment = (
                sm.attached_target_r[env_id] is not None
                or sm.attached_target_l[env_id] is not None
            )
            late_stall = episode_steps[env_id] >= args_cli.late_state_start
            failed_now = has_attachment and (failed_r or failed_l or late_stall)
            if failed_now:
                if failure_capture_count[env_id] == 0:
                    observed_failures += 1
                    print(
                        f"[recovery-bank] failure #{observed_failures} env={env_id} "
                        f"episode_step={episode_steps[env_id]} "
                        f"attached=({_attachment_name(sm.attached_target_l[env_id])},"
                        f"{_attachment_name(sm.attached_target_r[env_id])})",
                        flush=True,
                    )
                steps_since_capture[env_id] += 1
                if (
                    failure_capture_count[env_id] < args_cli.max_states_per_failure
                    and (
                        failure_capture_count[env_id] == 0
                        or steps_since_capture[env_id] >= args_cli.snapshot_stride
                    )
                    and len(saved["joint_pos_right"]) < args_cli.target_states
                ):
                    if failed_r or failed_l:
                        failed_arm = "both" if failed_r and failed_l else (
                            "right" if failed_r else "left"
                        )
                    else:
                        holds_r = sm.attached_target_r[env_id] is not None
                        holds_l = sm.attached_target_l[env_id] is not None
                        failed_arm = "both" if holds_r and holds_l else (
                            "right" if holds_r else "left"
                        )
                    snapshot(env_id, failed_arm)
                    failure_capture_count[env_id] += 1
                    steps_since_capture[env_id] = 0
            else:
                failure_capture_count[env_id] = 0
                steps_since_capture[env_id] = 0

            if bool(dones[env_id]):
                completed_episodes += 1
                episode_steps[env_id] = 0
                failure_capture_count[env_id] = 0
                steps_since_capture[env_id] = 0

        if completed_episodes and completed_episodes % 50 == 0:
            print(
                f"[recovery-bank] episodes={completed_episodes} "
                f"failures={observed_failures} states={len(saved['joint_pos_right'])}/"
                f"{args_cli.target_states}",
                flush=True,
            )

    if not saved["joint_pos_right"]:
        raise RuntimeError(
            "No attached-ring reach failures were observed; increase --max_episodes"
        )

    output = Path(args_cli.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task_phase": "phase_2",
        "source": (
            "stochastic_policy_recovery_trajectory"
            if args_cli.stochastic_actions
            else "deterministic_policy_recovery_trajectory"
        ),
        "checkpoint": os.path.abspath(args_cli.checkpoint),
        "seed": int(args_cli.seed),
        "states": len(saved["joint_pos_right"]),
        "observed_failures": int(observed_failures),
        "completed_episodes": int(completed_episodes),
        "collection_steps": int(collection_steps),
        "snapshot_stride": int(args_cli.snapshot_stride),
        "max_states_per_failure": int(args_cli.max_states_per_failure),
        "late_state_start": int(args_cli.late_state_start),
        "phase2_ring_order": args_cli.phase2_ring_order,
        "contains_expert_actions": False,
        "stochastic_actions": bool(args_cli.stochastic_actions),
    }
    np.savez(
        output,
        task_phase=np.asarray("phase_2"),
        joint_pos_right=np.asarray(saved["joint_pos_right"], dtype=np.float32),
        joint_vel_right=np.asarray(saved["joint_vel_right"], dtype=np.float32),
        joint_pos_left=np.asarray(saved["joint_pos_left"], dtype=np.float32),
        joint_vel_left=np.asarray(saved["joint_vel_left"], dtype=np.float32),
        tip_pos_right=np.asarray(saved["tip_pos_right"], dtype=np.float32),
        tip_pos_left=np.asarray(saved["tip_pos_left"], dtype=np.float32),
        ring_root_state_local=np.asarray(
            saved["ring_root_state_local"], dtype=np.float32
        ),
        ring_frozen_mask=np.asarray(saved["ring_frozen_mask"], dtype=np.bool_),
        ring_support_peg=np.asarray(saved["ring_support_peg"], dtype="<U32"),
        ring_stack_level=np.asarray(saved["ring_stack_level"], dtype=np.int64),
        attached_target_right=np.asarray(
            saved["attached_target_right"], dtype="<U32"
        ),
        attached_target_left=np.asarray(
            saved["attached_target_left"], dtype="<U32"
        ),
        failed_arm=np.asarray(saved["failed_arm"], dtype="<U8"),
        episode_index=np.asarray(saved["episode_index"], dtype=np.int64),
        episode_step=np.asarray(saved["episode_step"], dtype=np.int64),
        ring_names=np.asarray(RING_NAMES),
        joint_names_right=np.asarray(robot_right.joint_names),
        joint_names_left=np.asarray(robot_left.joint_names),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    output.with_suffix(".summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[recovery-bank] Saved {len(saved['joint_pos_right'])} states from "
        f"{observed_failures} failures to {output}",
        flush=True,
    )
    wrapper.close()
    isaac_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
