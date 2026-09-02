import argparse
import csv
import threading
import os
import glob
import random
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import torchvision.io as io
import cv2

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

# 1. INITIAL SETUP OF ISAAC SIM
from isaaclab.app import AppLauncher
from app_launcher_utils import resolve_tcc_checkpoint

parser = argparse.ArgumentParser(description="Inference evaluation of RL agent.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (default 1 for evaluation).")
parser.add_argument("--checkpoint", type=str, default="/home/aiprah/Documents/m_dVrk/modelli_salvati_sim/dvrk_ppo_best_terminal_distance.zip", help="Path to the .zip model file to load.")
parser.add_argument("--tcc_checkpoint", type=str, default="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase0_tcc/checkpoints/4001.ckpt", help="Path to the TCC representation weights.")
parser.add_argument("--dataset_path", type=str, default="/mnt/data/aiprah/data/sim_dataset_xirl_extra/train/phase_0/", help="Path to Xirl demonstrations dataset for computing mean goal embedding.")
parser.add_argument(
    "--task_phase",
    type=str,
    default="phase_0",
    choices=["phase_0", "phase_1"],
    help="Goal phase to evaluate. phase_1 starts stacked on green and succeeds with 2 rings on red and 2 on blue.",
)
parser.add_argument(
    "--phase1_initial_state",
    type=str,
    default="stack_green",
    choices=["stack_green", "random_workspace"],
    help="Initial state used only for phase_1 eval. random_workspace is an OOD stress test with loose rings.",
)
parser.add_argument(
    "--randomize_rings",
    action="store_true",
    help="Randomize ring reset positions. By default, rings reset to a fixed deterministic layout.",
)
parser.add_argument(
    "--stochastic_actions",
    action="store_true",
    help="Sample actions from the policy instead of using deterministic argmax actions.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for Python, NumPy, Torch and IsaacLab eval resets.",
)
parser.add_argument(
    "--episodes",
    type=int,
    default=0,
    help="Stop after this many episodes. Zero keeps the interactive infinite loop.",
)
parser.add_argument(
    "--metrics_output",
    type=str,
    default=None,
    help="Optional per-episode CSV output path for deterministic checkpoint ranking.",
)
parser.add_argument(
    "--disable_video",
    action="store_true",
    help="Do not accumulate or encode evaluation videos.",
)
parser.add_argument(
    "--quiet_steps",
    action="store_true",
    help="Suppress the per-step arm command output.",
)
parser.add_argument(
    "--action_masking",
    action="store_true",
    help="Apply state-aware joint masking while retaining the four-component MultiDiscrete action.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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

# Smart checkpoint resolution: if given path is missing, search for the filename in artifacts/repo
if not os.path.exists(args_cli.checkpoint):
    target_name = os.path.basename(args_cli.checkpoint)
    found_path = None
    for root_dir in [ARTIFACTS_DIR, ".", ".."]:
        if os.path.exists(root_dir):
            for r, _, files in os.walk(root_dir):
                if target_name in files:
                    found_path = os.path.abspath(os.path.join(r, target_name))
                    break
            if found_path:
                break
    if found_path:
        print(f"[INFO] Resolved checkpoint path to: {found_path}")
        args_cli.checkpoint = found_path

# The camera sensor requires Isaac rendering to be enabled.
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils
from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg

# Single source of truth — import from m_dVrk package
from m_dVrk.hrl.constants import (
    VERB_MAP, TARGET_MAP, VERB_TO_ID, TARGET_TO_ID,
    RING_TARGETS, PEG_TARGETS, RING_NAMES, PEG_NAMES, IDLE_ACTION,
    INVALID_COMMAND_PENALTY,
    REWARD_DEBUG_DUMP_INTERVAL, REWARD_DEBUG_DUMP_ENV_IDS,
    REWARD_DEBUG_DUMP_DIR, REWARD_DEBUG_DUMP_MAX_PER_ENV,
)
from m_dVrk.hrl.tcc import XIRLResnet18, load_tcc_model
from m_dVrk.hrl.wrapper import DVRKVisionHRLWrapper
from m_dVrk.hrl.masked_policy import masked_policy_custom_objects
from m_dVrk.hrl.checkpoints import vecnormalize_candidates
from m_dVrk.controllers.spartan_state_machine import SPARTANStateMachine
from m_dVrk.controllers.ring_sync import get_scene_entity_positions_w, sync_attached_and_frozen_rings

PHYSICS_STEPS_PER_RL_STEP = 400

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main_eval():
    if args_cli.episodes < 0:
        raise ValueError("--episodes must be non-negative")
    if args_cli.episodes and args_cli.num_envs != 1:
        raise ValueError("Bounded evaluation currently requires --num_envs 1 for unambiguous episode metrics")
    env_cfg = MDvrkEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.num_rerenders_on_reset = 2
    env_cfg.wait_for_textures = False
    set_seed(args_cli.seed)
    env_cfg.seed = args_cli.seed
    env_cfg.events.reset_rings.params["randomize"] = bool(args_cli.randomize_rings)
    print(
        "[INFO] Ring reset mode: "
        f"{'randomized' if args_cli.randomize_rings else 'fixed deterministic'}"
    )
    print(f"[INFO] Task phase: {args_cli.task_phase} | phase1_initial_state: {args_cli.phase1_initial_state}")
    print(f"[INFO] Simulation device: {env_cfg.sim.device}")
    
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    sm = SPARTANStateMachine(isaac_env)
    tcc = XIRLResnet18(32).to(isaac_env.device)
    
    try:
        resolved_tcc_ckpt = resolve_tcc_checkpoint(args_cli.tcc_checkpoint)
        print(f"[INFO] Loading TCC weights from: {resolved_tcc_ckpt}")
        ckpt = torch.load(resolved_tcc_ckpt, map_location=isaac_env.device)
        if "model" in ckpt:
            sd = ckpt["model"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt

        clean_sd = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("model."):
                k = k[len("model."):]
            clean_sd[k] = v

        load_result = tcc.load_state_dict(clean_sd, strict=True)
        print("[INFO] Weights loaded!")
    except Exception as e: 
        print(f"[ERR] Weights: {e}")

    tcc.eval()
    rl_env = DVRKVisionHRLWrapper(
        isaac_env,
        sm,
        tcc,
        task_phase=args_cli.task_phase,
    )
    
    # Compute goal embedding on the raw unwrapped environment first
    proc = T.Compose([T.Resize((112, 112), antialias=True)])
    try:
        rl_env.compute_and_set_goal_embedding(tcc, proc, args_cli.dataset_path, raise_on_error=False)
        print("[INFO] Goal ready (computed as dataset mean)!")
    except Exception:
        print("[ERR] Error computing mean goal from dataset; using zero goal embedding.")

    # Try to find the matching VecNormalize file and wrap the environment
    checkpoint_path = args_cli.checkpoint
    base_no_ext, _ = os.path.splitext(checkpoint_path)
    
    vec_normalize_path = None
    possible_paths = [str(path) for path in vecnormalize_candidates(checkpoint_path)]
    
    # also check artifacts locations
    possible_paths.extend([
        os.path.join(ARTIFACTS_DIR, "logs", "sb3", "vecnormalize.pkl"),
        os.path.join(
            ARTIFACTS_DIR,
            "checkpoints",
            "ppo",
            os.path.basename(base_no_ext) + "_vecnormalize.pkl",
        ),
    ])
    
    for path in possible_paths:
        if os.path.exists(path):
            vec_normalize_path = path
            break
            
    if vec_normalize_path is not None:
        print(f"[INFO] Loading VecNormalize statistics from: {vec_normalize_path}")
        rl_env = VecNormalize.load(vec_normalize_path, rl_env)
        rl_env.training = False
        rl_env.norm_reward = False
    else:
        print("[WARNING] No VecNormalize statistics found! Observations will not be normalized.")

    if args_cli.action_masking and isinstance(rl_env, VecNormalize) and rl_env.norm_obs:
        raise ValueError(
            "--action_masking requires raw observations, but this checkpoint's VecNormalize has norm_obs=True"
        )

    print(f"[INFO] Loading model from: {args_cli.checkpoint}")
    custom_objects = None
    if args_cli.action_masking:
        custom_objects = masked_policy_custom_objects(args_cli.checkpoint, args_cli.task_phase)
    model = PPO.load(
        args_cli.checkpoint,
        env=rl_env,
        device="cuda",
        custom_objects=custom_objects,
    )
    camera_sensor = rl_env.unwrapped.env.scene["camera"]
    video_frames = []
    print("[INFO] Starting evaluation...")
    obs = rl_env.reset()
    
    episode_count = 0
    steps = 0
    episode_return = 0.0
    invalid_commands = 0
    bimanual_steps = 0
    metric_rows = []
    while simulation_app.is_running():
        # The RL agent computes actions deterministically
        action, _states = model.predict(obs, deterministic=not args_cli.stochastic_actions)
        # Executes the step in the environment
        obs, rewards, dones, infos = rl_env.step(action)
        episode_return += float(rewards[0])
        invalid_commands += int(infos[0].get("invalid_command_count", 0))
        
        # Get active triplets for env 0
        cmd_l = rl_env.sm.current_triplet_l[0]
        cmd_r = rl_env.sm.current_triplet_r[0]
        str_l = f"{cmd_l['verb']} -> {cmd_l['target']}" if cmd_l else "idle -> None"
        str_r = f"{cmd_r['verb']} -> {cmd_r['target']}" if cmd_r else "idle -> None"
        if cmd_l is not None and cmd_r is not None:
            bimanual_steps += 1
        if not args_cli.quiet_steps:
            print(f"[Step {steps:03d}] Left Arm: {str_l:<20} | Right Arm: {str_r:<20}")

        # Retrieve virtual camera frame (H, W, 4) or (H, W, 3)
        if not args_cli.disable_video:
            rgb_data = camera_sensor.data.output["rgb"][0]
            frame_np = rgb_data.cpu().numpy()[:, :, :3].copy()
            overlay = frame_np.copy()
            cv2.rectangle(overlay, (0, 0), (frame_np.shape[1], 75), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame_np, 0.6, 0, frame_np)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame_np, f"LEFT ARM:  {str_l.upper()}", (15, 30), font, 0.6, (50, 255, 50), 2, cv2.LINE_AA)
            cv2.putText(frame_np, f"RIGHT ARM: {str_r.upper()}", (15, 55), font, 0.6, (50, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_np, f"STEP: {steps:03d}", (frame_np.shape[1] - 150, 30), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_np, f"REWARD: {rewards[0]:.4f}", (frame_np.shape[1] - 200, 55), font, 0.6, (100, 230, 255), 2, cv2.LINE_AA)
            video_frames.append(torch.from_numpy(frame_np))
        
        # Save video when episode terminates (at reset)
        if dones[0]:
            info0 = infos[0]
            print(
                f"[DONE] success={info0.get('is_success', False)} "
                f"rings_complete={info0.get('rings_complete', False)} "
                f"red={info0.get('red_peg_ring_count', 0)} "
                f"blue={info0.get('blue_peg_ring_count', 0)} "
                f"green={info0.get('green_peg_ring_count', 0)} "
                f"distance={info0.get('terminal_distance', float('nan')):.4f}"
            )

            metric_rows.append(
                {
                    "checkpoint": os.path.abspath(args_cli.checkpoint),
                    "seed": args_cli.seed,
                    "episode": episode_count,
                    "deterministic": not args_cli.stochastic_actions,
                    "action_masking": args_cli.action_masking,
                    "success": bool(info0.get("is_success", False)),
                    "episode_steps": steps + 1,
                    "episode_return": episode_return,
                    "terminal_distance": float(info0.get("terminal_distance", float("nan"))),
                    "green_peg_ring_count": int(info0.get("green_peg_ring_count", 0)),
                    "red_peg_ring_count": int(info0.get("red_peg_ring_count", 0)),
                    "blue_peg_ring_count": int(info0.get("blue_peg_ring_count", 0)),
                    "invalid_commands": invalid_commands,
                    "bimanual_steps": bimanual_steps,
                }
            )
            
            # Save video
            if video_frames:
                phase_num = args_cli.task_phase.replace("phase_", "")
                success_str = "success" if info0.get('is_success', False) else "fail"
                video_name = f"episode_{episode_count:03d}_{success_str}_phase_{phase_num}.mp4"
                video_tensor = torch.stack(video_frames)  # (T, H, W, C)
                io.write_video(video_name, video_tensor, fps=30)
                print(f"\033[92m✓ Video saved: {video_name} ({len(video_frames)} frames, {steps} steps)\033[0m")
                video_frames = []
            
            episode_count += 1
            steps = 0
            episode_return = 0.0
            invalid_commands = 0
            bimanual_steps = 0
            if args_cli.episodes and episode_count >= args_cli.episodes:
                break
        else:
            steps += 1

    if args_cli.metrics_output is not None:
        output_dir = os.path.dirname(os.path.abspath(args_cli.metrics_output))
        os.makedirs(output_dir, exist_ok=True)
        with open(args_cli.metrics_output, "w", newline="", encoding="utf-8") as stream:
            if metric_rows:
                writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
                writer.writeheader()
                writer.writerows(metric_rows)
        print(f"[INFO] Evaluation metrics saved: {args_cli.metrics_output}")

if __name__ == "__main__":
    main_eval()
    if args_cli.episodes:
        # Isaac/Kit can block indefinitely during shutdown in headless batch
        # evaluation. All result files are closed at this point; an immediate
        # process exit also guarantees that GPU resources are released before
        # the checkpoint ranker launches the next candidate.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    simulation_app.close()
