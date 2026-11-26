"""Validate that a Phase-4 empirical reset reproduces one paired NPZ row."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from app_launcher_utils import pin_process_to_requested_cuda_device
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--bank", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.device = pin_process_to_requested_cuda_device(
    getattr(args_cli, "device", None)
)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv

from m_dVrk.controllers.spartan_state_machine import SPARTANStateMachine
from m_dVrk.hrl.constants import RING_NAMES, apply_phase_entry_empirical_bank
from m_dVrk.hrl.wrapper import DVRKVisionHRLWrapper
from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg


def main() -> None:
    bank_path = os.path.abspath(args_cli.bank)
    cfg = MDvrkEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.seed = args_cli.seed
    cfg.num_rerenders_on_reset = 2
    cfg.wait_for_textures = False
    apply_phase_entry_empirical_bank(
        cfg,
        bank_path=bank_path,
        empirical_probability=1.0,
        use_recorded_velocity=True,
    )

    env = ManagerBasedRLEnv(cfg=cfg)
    sm = SPARTANStateMachine(env)
    wrapper = DVRKVisionHRLWrapper(
        env,
        sm,
        None,
        task_phase="phase_4",
        action_mask_mode="minimal",
        success_snapshot_prob=0.0,
        success_video_prob=0.0,
    )
    wrapper.reset()

    # Refresh IsaacLab buffers after the direct state writes performed by reset.
    env.sim.forward()
    env.scene.update(dt=0.0)

    with np.load(bank_path, allow_pickle=False) as bank:
        rows = env._phase4_empirical_entry_indices.detach().cpu().numpy()
        enabled = env._phase4_empirical_entry_enabled.detach().cpu().numpy()
        if not bool(enabled.all()):
            raise RuntimeError("Expected every validation environment to be empirical")

        max_joint_pos_error = 0.0
        max_joint_vel_error = 0.0
        for side in ("right", "left"):
            robot = env.scene[f"robot_{side}"]
            name_to_col = {name: idx for idx, name in enumerate(robot.joint_names)}
            cols = [name_to_col[str(name)] for name in bank[f"joint_names_{side}"]]
            actual_pos = robot.data.joint_pos[:, cols].detach().cpu().numpy()
            actual_vel = robot.data.joint_vel[:, cols].detach().cpu().numpy()
            expected_pos = bank[f"joint_pos_{side}"][rows]
            expected_vel = bank[f"joint_vel_{side}"][rows]
            max_joint_pos_error = max(
                max_joint_pos_error,
                float(np.max(np.abs(actual_pos - expected_pos))),
            )
            max_joint_vel_error = max(
                max_joint_vel_error,
                float(np.max(np.abs(actual_vel - expected_vel))),
            )

        bank_ring_names = [str(name) for name in bank["ring_names"]]
        ring_col = {name: idx for idx, name in enumerate(bank_ring_names)}
        max_ring_state_error = 0.0
        support_mismatches = 0
        stack_level_mismatches = 0
        for ring_name in RING_NAMES:
            col = ring_col[ring_name]
            actual = env.scene[ring_name].data.root_state_w.detach().cpu().numpy().copy()
            actual[:, 0:3] -= env.scene.env_origins.detach().cpu().numpy()
            expected = bank["ring_root_state_local"][rows, col]
            max_ring_state_error = max(
                max_ring_state_error,
                float(np.max(np.abs(actual - expected))),
            )
            for env_id, row in enumerate(rows.tolist()):
                expected_support = str(bank["ring_support_peg"][row, col])
                actual_support = sm.ring_support_peg[ring_name][env_id] or "None"
                support_mismatches += int(actual_support != expected_support)
                expected_level = int(bank["ring_stack_level"][row, col])
                actual_level = int(sm.ring_stack_level[ring_name][env_id].item())
                stack_level_mismatches += int(actual_level != expected_level)

    result = {
        "rows": rows.tolist(),
        "max_joint_pos_error": max_joint_pos_error,
        "max_joint_vel_error": max_joint_vel_error,
        "max_ring_state_error": max_ring_state_error,
        "support_mismatches": support_mismatches,
        "stack_level_mismatches": stack_level_mismatches,
    }
    print(f"[full-state-validation] {result}", flush=True)
    if (
        max_joint_pos_error > 1e-5
        or max_joint_vel_error > 1e-5
        or max_ring_state_error > 1e-5
        or support_mismatches
        or stack_level_mismatches
    ):
        raise RuntimeError(f"Empirical full-state reset mismatch: {result}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
