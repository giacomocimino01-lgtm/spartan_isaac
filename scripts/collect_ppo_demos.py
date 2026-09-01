"""Collect expert demonstrations (observations and actions) for PPO Behavior Cloning.

This script runs the SPARTANStateMachine to solve the Peg-and-Ring task, wraps the environment
in DVRKVisionHRLWrapper (the same wrapper used in parallel_run.py), and records successful
(observation, action) pairs to a .npz file.

Example usage:
    PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 nohup ${IsaacLab_PATH}/isaaclab.sh -p scripts/collect_ppo_demos.py --num_envs 8 --num_episodes 50 --headless --save_videos > outlog.log 2>&1 &

"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import torchvision.io as io
torch.cuda.empty_cache()

from app_launcher_utils import pin_process_to_requested_cuda_device, resolve_tcc_checkpoint
from isaaclab.app import AppLauncher
from m_dVrk.hrl.constants import PEG_GREEN

# 1. PARSE ARGUMENTS
parser = argparse.ArgumentParser(description="Collect expert demonstrations for PPO Behavior Cloning.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel Isaac environments.")
parser.add_argument("--num_episodes", type=int, default=25, help="Number of successful episodes to collect.")
parser.add_argument(
    "--output_path",
    type=str,
    default="/home/aiprah/Documents/m_dVrk/dataset_supervisionato_randomized.npz",
    help="Output path for the collected dataset .npz file.",
)
parser.add_argument(
    "--save_videos",
    action="store_true",
    help="Save video frames of successful episodes as PNGs in a subfolder.",
)
parser.add_argument(
    "--randomize_rings",
    action="store_true",
    help="Use the environment random ring reset instead of the fixed deterministic layout.",
)
parser.add_argument(
    "--ring_order",
    choices=("random", "fixed"),
    default="random",
    help="Ring order inside each scripted video.",
)
parser.add_argument(
    "--arm_mode",
    choices=("right", "left", "random", "alternate"),
    default="random",
    help="Arm selection for each ring.",
)
parser.add_argument("--target_peg", type=str, default=PEG_GREEN, help="Peg where rings are placed.")
parser.add_argument(
    "--num_rings",
    type=int,
    default=4,
    choices=(1, 2, 3, 4),
    help="Number of rings to place in each episode.",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--max_steps_per_episode", type=int, default=6500, help="Discard episode after this many steps.")
parser.add_argument(
    "--task_phase",
    type=str,
    choices=["phase_0", "phase_1"],
    default="phase_0",
    help="Phase for the environment task.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Load project-level paths from configs/defaults.yaml (optional)
def _load_project_paths():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = os.path.join(repo_root, "configs", "defaults.yaml")
    try:
        import yaml

        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("paths", {})
    except Exception:
        pass
    return {}

_paths = _load_project_paths()
ARTIFACTS_DIR = _paths.get("artifacts_dir", "artifacts/")

# If the user left the legacy default output_path, prefer artifacts location
_legacy_default_output = "/home/aiprah/Documents/m_dVrk/dataset_supervisionato_randomized.npz"
if getattr(args_cli, "output_path", None) == _legacy_default_output:
    args_cli.output_path = os.path.join(ARTIFACTS_DIR, "datasets", "bc", os.path.basename(_legacy_default_output))

# The camera sensor requires Isaac rendering to be enabled.
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True

requested_device = getattr(args_cli, "device", None)
args_cli.device = pin_process_to_requested_cuda_device(requested_device)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg

# Single source of truth — import from m_dVrk package
from m_dVrk.hrl.constants import (
    VERB_MAP, TARGET_MAP, VERB_TO_ID, TARGET_TO_ID,
    RING_TARGETS, PEG_TARGETS, RING_NAMES, PEG_NAMES, IDLE_ACTION,
)
from m_dVrk.hrl.tcc import XIRLResnet18, load_tcc_model
from m_dVrk.hrl.wrapper import DVRKVisionHRLWrapper
from m_dVrk.controllers.spartan_state_machine import SPARTANStateMachine
from m_dVrk.controllers.ring_sync import get_scene_entity_positions_w, sync_attached_and_frozen_rings


# ─── Helper classes & functions ───────────────────────────────────────────────

class EpisodeSlot:
    """Per-environment state used by the collection loop."""
    def __init__(self):
        self.obs_buffer: list = []
        self.actions_buffer: list = []
        self.rgb_buffer: list = []
        self.step_idx: int = 0
        self.done: bool = False
        # High-level command queue: list of (verb, arm, target) triples
        self.commands: list = []
        self.command_idx: int = 0


def _build_commands(ring_order: str, arm_mode: str, target_peg: str, num_rings: int) -> list:
    """Build the ordered list of (verb, arm, target) triples for one episode."""
    import random as _rnd
    rings = list(RING_NAMES)[:num_rings]
    if ring_order == "random":
        _rnd.shuffle(rings)
    commands = []
    for i, ring in enumerate(rings):
        if arm_mode == "right":
            arm = "right_arm"
        elif arm_mode == "left":
            arm = "left_arm"
        elif arm_mode == "alternate":
            arm = "right_arm" if i % 2 == 0 else "left_arm"
        else:
            arm = _rnd.choice(["right_arm", "left_arm"])
        commands += [
            ("reach",   arm, ring),
            ("grasp",   arm, ring),
            ("reach",   arm, target_peg),
            ("release", arm, target_peg),
        ]
    return commands


def start_episode(isaac_env, sm, slots, env_id: int):
    """Reset an environment slot and arm the command queue for a new episode."""
    slot = slots[env_id]
    slot.obs_buffer = []
    slot.actions_buffer = []
    slot.rgb_buffer = []
    slot.step_idx = 0
    slot.done = False
    slot.commands = _build_commands(
        args_cli.ring_order,
        args_cli.arm_mode,
        args_cli.target_peg,
        args_cli.num_rings,
    )
    slot.command_idx = 0
    sm.reset_env(env_id, phase=args_cli.task_phase)


_IDLE_VERB_ID   = VERB_TO_ID["idle"]
_NONE_TARGET_ID = TARGET_TO_ID["None"]


def get_triplet_action_for_env(sm, slot: EpisodeSlot, env_id: int):
    """Return (verb_l, tgt_l, verb_r, tgt_r) for this env's current command."""
    # Issue the next command when the state machine is free
    if sm.all_idle(env_id) and slot.command_idx < len(slot.commands):
        verb_str, arm_str, tgt_str = slot.commands[slot.command_idx]
        slot.command_idx += 1
        sm.set_new_triplet(verb_str, arm_str, tgt_str, env_id)

    cmd_l = sm.current_triplet_l[env_id]
    cmd_r = sm.current_triplet_r[env_id]

    verb_l = VERB_TO_ID.get(cmd_l["verb"], _IDLE_VERB_ID)   if cmd_l else _IDLE_VERB_ID
    tgt_l  = TARGET_TO_ID.get(cmd_l["target"], _NONE_TARGET_ID) if cmd_l else _NONE_TARGET_ID
    verb_r = VERB_TO_ID.get(cmd_r["verb"], _IDLE_VERB_ID)   if cmd_r else _IDLE_VERB_ID
    tgt_r  = TARGET_TO_ID.get(cmd_r["target"], _NONE_TARGET_ID) if cmd_r else _NONE_TARGET_ID

    return verb_l, tgt_l, verb_r, tgt_r


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():

    if args_cli.task_phase == "phase_1":
        args_cli.num_rings = 4
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    env_cfg = MDvrkEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.num_rerenders_on_reset = 2
    env_cfg.events.reset_rings.params["randomize"] = bool(args_cli.randomize_rings)

    print(f"[collector] Starting environment on device {env_cfg.sim.device}...")
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    sm = SPARTANStateMachine(isaac_env)

    # Load TCC Model weights to ensure valid observation embedding stacks
    tcc = XIRLResnet18(embedding_size=32).to(isaac_env.device)
    try:
        experiment_dir = f"/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_{args_cli.task_phase.replace('_', '')}_tcc"
        tcc_ckpt_path = resolve_tcc_checkpoint(experiment_dir)
        print(f"[collector] Loading XIRL TCC weights from: {tcc_ckpt_path}")
        ckpt = torch.load(
            tcc_ckpt_path,
            map_location=isaac_env.device,
        )
        sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
        clean_sd = {} 
        for k, v in sd.items():
            if k.startswith("module."): k = k[len("module."):]
            if k.startswith("model."): k = k[len("model."):]
            clean_sd[k] = v
        tcc.load_state_dict(clean_sd, strict=True)
        print("[collector] XIRL TCC weights loaded successfully.")
    except Exception as e:
        print(f"[collector] WARNING: Failed to load XIRL weights: {e}. Observation embeddings might be noisy.")

    tcc.eval()

    rl_env = DVRKVisionHRLWrapper(isaac_env, sm, tcc, task_phase=args_cli.task_phase)

    # Compute goal embedding from the dataset so the reward function is not None
    import torchvision.transforms as T
    goal_dataset_path = f"/mnt/data/aiprah/data/random_sim_dataset_tcc/train/{args_cli.task_phase}/"
    proc = T.Compose([T.Resize((112, 112), antialias=True)])
    try:
        rl_env.compute_and_set_goal_embedding(tcc, proc, goal_dataset_path, raise_on_error=True)
        print(f"[collector] Goal embedding set from {goal_dataset_path}")
    except Exception as e:
        print(f"[collector] WARNING: Could not set goal embedding from {goal_dataset_path}: {e}")
        print("[collector] Using zero goal embedding as fallback.")
        import torch as _torch
        rl_env.goal_embedding = _torch.zeros(rl_env.emb_dim, device=isaac_env.device)


    slots = [EpisodeSlot() for _ in range(args_cli.num_envs)]
    
    successful_episodes_count = 0
    all_observations = []
    all_actions = []

    # Reset environment to get initial observations
    obs = rl_env.reset()

    # Start initial episodes
    for env_id in range(args_cli.num_envs):
        start_episode(isaac_env, sm, slots, env_id)

    print(f"[collector] Starting collection of {args_cli.num_episodes} successful episodes...")

    while simulation_app.is_running() and successful_episodes_count < args_cli.num_episodes:
        # Determine high-level commands to step the environment
        step_actions = np.zeros((args_cli.num_envs, 4), dtype=np.int64)
        for env_id in range(args_cli.num_envs):
            slot = slots[env_id]
            if not slot.done:
                verb_l, tgt_l, verb_r, tgt_r = get_triplet_action_for_env(sm, slot, env_id)
                step_actions[env_id] = [verb_l, tgt_l, verb_r, tgt_r]

        # Save current observations and actions to buffer BEFORE stepping the environment.
        # This aligns the state observed at time t with the action executed at time t.
        # Note: the wrapper will modify continuous commands inside step_wait, but it keeps track
        # of the active high-level discrete actions in `prev_actions` which is what we need to record.
        for env_id in range(args_cli.num_envs):
            slot = slots[env_id]
            if not slot.done:
                slot.obs_buffer.append(obs[env_id].copy())
                if args_cli.save_videos:
                    rgb_frame = rl_env.env.scene.sensors["camera"].data.output["rgb"][env_id, :, :, :3].cpu()
                    slot.rgb_buffer.append(rgb_frame)

        # Step the environment
        next_obs, rewards, dones, infos = rl_env.step(step_actions)

        # Retrieve the actual actions that were executed by the wrapper (handling arm busy overrides)
        executed_actions = rl_env.prev_actions.cpu().numpy()

        for env_id in range(args_cli.num_envs):
            slot = slots[env_id]
            if not slot.done:
                slot.actions_buffer.append(executed_actions[env_id].copy())
                slot.step_idx += 1

                # Check if episode ended
                if dones[env_id]:
                    is_success = infos[env_id].get("is_success", False)
                    # Verify task logical completion
                    sm_success = sm.successful(env_id, phase=args_cli.task_phase)
                    
                    if is_success or sm_success:
                        successful_episodes_count += 1
                        all_observations.extend(slot.obs_buffer)
                        all_actions.extend(slot.actions_buffer)
                        print(
                            f"[collector] Episode SUCCESS! Saved {len(slot.obs_buffer)} steps. "
                            f"Progress: {successful_episodes_count}/{args_cli.num_episodes}"
                        )
                        if args_cli.save_videos:

                            video_dir = Path(ARTIFACTS_DIR) / "video_raccolti" / f"episodio_{successful_episodes_count:04d}"
                            video_dir.mkdir(parents=True, exist_ok=True)
                            for frame_idx, img in enumerate(slot.rgb_buffer):
                                if img.dtype != torch.uint8:
                                    img = (img.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
                                img = img.permute(2, 0, 1).contiguous()
                                io.write_png(img, str(video_dir / f"{frame_idx:06d}.png"))
                            print(f"[collector] Video frames saved to {video_dir}")
                    else:
                        print(f"[collector] Episode FAILED (steps: {slot.step_idx}). Discarding buffer.")

                    # Start a new episode in this environment
                    if successful_episodes_count < args_cli.num_episodes:
                        start_episode(isaac_env, sm, slots, env_id)
                    else:
                        slot.done = True
                
                # Check for max step timeout
                elif slot.step_idx >= args_cli.max_steps_per_episode:
                    print(f"[collector] Episode TIMEOUT in env {env_id}. Discarding buffer.")
                    start_episode(isaac_env, sm, slots, env_id)

        obs = next_obs

    # Save collected dataset to disk
    if len(all_observations) > 0:
        os.makedirs(os.path.dirname(args_cli.output_path), exist_ok=True)
        obs_arr = np.array(all_observations, dtype=np.float32)
        acts_arr = np.array(all_actions, dtype=np.int64)

        # Metadata to help reproducibility and downstream checks
        metadata = {
            "obs_dim": int(obs_arr.shape[1]) if obs_arr.ndim > 1 else int(obs_arr.size),
            "emb_dim": int((obs_arr.shape[1] - 62 - 86) // 3) if obs_arr.ndim > 1 else None,
            "aux_dim": 62,
            "geom_dim": 86,
            "task_phase": args_cli.task_phase,
            "num_rings": int(args_cli.num_rings),
            "seed": int(args_cli.seed),
        }

        # Git commit if available
        try:
            import subprocess

            git_rev = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
            )
            metadata["git_commit"] = git_rev
        except Exception:
            metadata["git_commit"] = None

        # Timestamp
        try:
            from datetime import datetime

            metadata["timestamp"] = datetime.utcnow().isoformat() + "Z"
        except Exception:
            metadata["timestamp"] = None

        # Basic validation against our minimal metadata schema
        def _validate_metadata(m: dict) -> bool:
            required = ["obs_dim", "aux_dim", "geom_dim", "emb_dim", "task_phase", "timestamp"]
            for k in required:
                if k not in m:
                    print(f"[collector] WARNING: metadata missing required key: {k}")
                    return False
            if not isinstance(m["obs_dim"], int) or m["obs_dim"] <= 0:
                print("[collector] WARNING: metadata.obs_dim invalid")
                return False
            return True

        if not _validate_metadata(metadata):
            print("[collector] WARNING: metadata failed basic validation. Saving dataset anyway.")

        np.savez(
            args_cli.output_path,
            obs=obs_arr,
            actions=acts_arr,
            metadata=metadata,
        )
        print(f"[collector] Saved dataset successfully to {args_cli.output_path}")
        print(f"[collector] Total transitions collected: {len(all_observations)}")
    else:
        print("[collector] ERROR: No transitions were collected.")

    rl_env.close()
    isaac_env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
