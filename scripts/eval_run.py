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

from m_dVrk.hrl.constants import (
    PHASE1_DEFAULT_INITIAL_STATE,
    PHASE1_INITIAL_STATES,
    PHASE2_DEFAULT_INITIAL_STATE,
    PHASE2_INITIAL_STATES,
    RING_NAMES,
)

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
    choices=["phase_0", "phase_1", "phase_2", "phase_3", "phase_4"],
    help=(
        "Goal phase. Phase 3 starts at center4 and requires one arbitrary "
        "ring on each coloured peg. Phase 4 moves two rings to each outside side."
    ),
)
parser.add_argument(
    "--phase2_initial_state",
    type=str,
    default=PHASE2_DEFAULT_INITIAL_STATE,
    choices=PHASE2_INITIAL_STATES,
    help="Initial state used only for Phase-2 evaluation.",
)
parser.add_argument(
    "--phase1_initial_state",
    type=str,
    default=PHASE1_DEFAULT_INITIAL_STATE,
    choices=PHASE1_INITIAL_STATES,
    help="Initial state used only for Phase-1 evaluation.",
)
parser.add_argument(
    "--phase4_ring_order",
    nargs=4,
    choices=RING_NAMES,
    default=None,
    metavar=("PEG_RED", "PEG_YELLOW", "PEG_GREEN", "PEG_BLUE"),
    help=(
        "Optional fixed Phase-4 layout: four ring names assigned respectively "
        "to the red, yellow, green and blue pegs. By default the order is shuffled."
    ),
)
parser.add_argument(
    "--randomize_rings",
    action="store_true",
    help="Randomize ring reset positions. By default, rings reset to a fixed deterministic layout.",
)
parser.add_argument(
    "--ring_randomization_scale",
    type=float,
    default=1.0,
    help=(
        "Fraction of the full ring-reset range used with --randomize_rings "
        "(0=fixed anchors, 1=full configured range)."
    ),
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
parser.add_argument(
    "--action_mask_mode",
    choices=("guided", "goal_agnostic", "minimal"),
    default="guided",
    help="Select the historical goal-aware mask or the minimal physical mask.",
)
parser.add_argument(
    "--park_before_evaluation",
    action="store_true",
    help=(
        "Diagnostic mode: move both arms to the same neutral park pose used "
        "between phases by eval_full_task.py before the first policy action."
    ),
)
parser.add_argument(
    "--phase_entry_park_reset",
    action="store_true",
    help="Reset the robots directly into the full-task parked joint pose.",
)
parser.add_argument(
    "--phase_entry_park_neighborhood_probability",
    type=float,
    default=0.0,
    help="Probability of a safe perturbation around parked on each reset.",
)
parser.add_argument(
    "--phase_entry_park_neighborhood_scale",
    type=float,
    default=1.0,
    help="Scale in [0,1] of the parked-pose perturbation ellipsoid.",
)
parser.add_argument(
    "--phase_entry_empirical_bank",
    type=str,
    default=None,
    help="Reset Phase 2 or 4 from a paired empirical joint/ring-state NPZ bank.",
)
parser.add_argument(
    "--phase_entry_empirical_probability",
    type=float,
    default=1.0,
    help="Probability of sampling the empirical bank instead of parked fallback.",
)
parser.add_argument(
    "--phase_entry_empirical_sample_mode",
    choices=("random", "sequential"),
    default="random",
    help="Use random bank rows or cycle through every row in order.",
)
parser.add_argument(
    "--phase_entry_empirical_velocity",
    action="store_true",
    help="Restore recorded robot joint velocities from the empirical bank.",
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
    apply_phase_entry_empirical_bank,
    apply_phase_entry_park_joint_pose,
)
from m_dVrk.hrl.tcc import XIRLResnet18, load_tcc_model
from m_dVrk.hrl.wrapper import DVRKVisionHRLWrapper
from m_dVrk.hrl.masked_policy import masked_policy_custom_objects
from m_dVrk.hrl.checkpoints import vecnormalize_candidates
from m_dVrk.controllers.spartan_state_machine import SPARTANStateMachine
from m_dVrk.controllers.ring_sync import get_scene_entity_positions_w, sync_attached_and_frozen_rings

PHYSICS_STEPS_PER_RL_STEP = 400


def park_arms_before_evaluation(wrapper, max_steps: int = 150, settle_steps: int = 3):
    """Reproduce the physical arm state at a full-task phase boundary."""
    sm = wrapper.sm
    sm.reset_control_state(0)
    sm.set_new_triplet("park", "right_arm", "None", 0)
    sm.set_new_triplet("park", "left_arm", "None", 0)
    active_mask = torch.ones(
        wrapper.num_envs, dtype=torch.bool, device=wrapper.env.device
    )

    for step in range(1, max_steps + 1):
        _, _, terminated, truncated, _ = wrapper.env.step(sm.get_action())
        done = terminated.to(dtype=torch.bool) | truncated.to(dtype=torch.bool)
        sync_attached_and_frozen_rings(wrapper.env, sm, active_mask & ~done)
        if bool(done[0].item()):
            raise RuntimeError("Environment terminated while parking arms")
        if sm.all_idle(0):
            break
    else:
        raise RuntimeError("Arms did not reach the park pose")

    for _ in range(settle_steps):
        _, _, terminated, truncated, _ = wrapper.env.step(sm.get_action())
        done = terminated.to(dtype=torch.bool) | truncated.to(dtype=torch.bool)
        sync_attached_and_frozen_rings(wrapper.env, sm, active_mask & ~done)
        if bool(done[0].item()):
            raise RuntimeError("Environment terminated while settling parked arms")

    print(f"[INFO] Parked arms in {step + settle_steps} low-level steps")
    for robot_name in ("robot_right", "robot_left"):
        robot = wrapper.env.scene[robot_name]
        joint_values = robot.data.joint_pos[0].detach().cpu().tolist()
        print(
            f"[INFO] {robot_name} parked joints: "
            + ", ".join(
                f"{name}={value:.9f}"
                for name, value in zip(robot.joint_names, joint_values, strict=True)
            )
        )
    return wrapper.transition_to_phase(
        wrapper.task_phase,
        wrapper.tcc_model,
        wrapper.goal_embedding,
    )

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
    if not 0.0 <= args_cli.ring_randomization_scale <= 1.0:
        raise ValueError("--ring_randomization_scale must be in [0, 1]")
    if not 0.0 <= args_cli.phase_entry_park_neighborhood_probability <= 1.0:
        raise ValueError("park-neighborhood probability must be in [0, 1]")
    if not 0.0 <= args_cli.phase_entry_park_neighborhood_scale <= 1.0:
        raise ValueError("park-neighborhood scale must be in [0, 1]")
    if not 0.0 <= args_cli.phase_entry_empirical_probability <= 1.0:
        raise ValueError("empirical-bank probability must be in [0, 1]")
    if (
        args_cli.phase_entry_park_neighborhood_probability > 0.0
        and not args_cli.phase_entry_park_reset
    ):
        raise ValueError("park-neighborhood sampling requires --phase_entry_park_reset")
    if args_cli.phase_entry_empirical_bank and args_cli.phase_entry_park_reset:
        raise ValueError(
            "empirical-bank reset and parked-pose reset are mutually exclusive"
        )
    if args_cli.phase_entry_empirical_bank and args_cli.task_phase not in {
        "phase_2",
        "phase_4",
    }:
        raise ValueError("empirical-bank reset is supported for Phase 2 and Phase 4")
    if args_cli.episodes and args_cli.num_envs != 1:
        raise ValueError("Bounded evaluation currently requires --num_envs 1 for unambiguous episode metrics")
    if args_cli.phase4_ring_order is not None and set(args_cli.phase4_ring_order) != set(RING_NAMES):
        raise ValueError("--phase4_ring_order must contain every ring exactly once")
    env_cfg = MDvrkEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.num_rerenders_on_reset = 2
    env_cfg.wait_for_textures = False
    set_seed(args_cli.seed)
    env_cfg.seed = args_cli.seed
    if args_cli.phase_entry_park_reset:
        apply_phase_entry_park_joint_pose(
            env_cfg,
            neighborhood_probability=(
                args_cli.phase_entry_park_neighborhood_probability
            ),
            neighborhood_scale=args_cli.phase_entry_park_neighborhood_scale,
        )
        print(
            "[INFO] Arm reset mode: full-task parked phase-entry pose "
            f"(neighborhood_probability="
            f"{args_cli.phase_entry_park_neighborhood_probability:.2f}, "
            f"neighborhood_scale={args_cli.phase_entry_park_neighborhood_scale:.2f})"
        )
    elif args_cli.phase_entry_empirical_bank:
        apply_phase_entry_empirical_bank(
            env_cfg,
            bank_path=os.path.abspath(args_cli.phase_entry_empirical_bank),
            empirical_probability=args_cli.phase_entry_empirical_probability,
            empirical_sample_mode=args_cli.phase_entry_empirical_sample_mode,
            use_recorded_velocity=args_cli.phase_entry_empirical_velocity,
            task_phase=args_cli.task_phase,
        )
        print(
            "[INFO] Arm/ring reset mode: empirical full-state bank "
            f"(path={os.path.abspath(args_cli.phase_entry_empirical_bank)}, "
            f"probability={args_cli.phase_entry_empirical_probability:.2f}, "
            f"sample_mode={args_cli.phase_entry_empirical_sample_mode}, "
            f"recorded_velocity={args_cli.phase_entry_empirical_velocity})"
        )
    env_cfg.events.reset_rings.params["randomize"] = bool(args_cli.randomize_rings)
    env_cfg.events.reset_rings.params["randomization_scale"] = float(
        args_cli.ring_randomization_scale
    )
    print(
        "[INFO] Ring reset mode: "
        + (
            f"randomized (scale={args_cli.ring_randomization_scale:.4f})"
            if args_cli.randomize_rings
            else "fixed deterministic"
        )
    )
    print(
        f"[INFO] Task phase: {args_cli.task_phase} | "
        f"phase1_initial_state: {args_cli.phase1_initial_state} | "
        f"phase2_initial_state: {args_cli.phase2_initial_state} | "
        f"phase4_ring_order: {args_cli.phase4_ring_order or 'shuffled'}"
    )
    print(f"[INFO] Simulation device: {env_cfg.sim.device}")
    
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    if args_cli.phase_entry_park_reset:
        isaac_env.phase_entry_pose_scale = 1.0
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
    task_wrapper = DVRKVisionHRLWrapper(
        isaac_env,
        sm,
        tcc,
        task_phase=args_cli.task_phase,
        action_mask_mode=args_cli.action_mask_mode,
        phase1_initial_state=args_cli.phase1_initial_state,
        phase2_initial_state=args_cli.phase2_initial_state,
        phase4_ring_order=args_cli.phase4_ring_order,
    )
    rl_env = task_wrapper
    
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
        custom_objects = masked_policy_custom_objects(
            args_cli.checkpoint,
            args_cli.task_phase,
            args_cli.action_mask_mode,
        )
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
    if args_cli.park_before_evaluation:
        obs = park_arms_before_evaluation(task_wrapper)
        if isinstance(rl_env, VecNormalize):
            obs = rl_env.normalize_obs(obs)
    
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
                f"yellow={info0.get('yellow_peg_ring_count', 0)} "
                f"blue={info0.get('blue_peg_ring_count', 0)} "
                f"green={info0.get('green_peg_ring_count', 0)} "
                f"center={info0.get('board_center_ring_count', 0)} "
                f"distance={info0.get('terminal_distance', float('nan')):.4f}"
            )

            metric_rows.append(
                {
                    "checkpoint": os.path.abspath(args_cli.checkpoint),
                    "seed": args_cli.seed,
                    "episode": episode_count,
                    "deterministic": not args_cli.stochastic_actions,
                    "action_masking": args_cli.action_masking,
                    "action_mask_mode": args_cli.action_mask_mode,
                    "park_before_evaluation": args_cli.park_before_evaluation,
                    "phase1_initial_state": info0.get(
                        "phase1_initial_state", args_cli.phase1_initial_state
                    ),
                    "phase1_resolved_initial_state": info0.get(
                        "phase1_resolved_initial_state", ""
                    ),
                    "phase1_initial_layout": info0.get(
                        "phase1_initial_layout", ""
                    ),
                    "phase2_initial_state": info0.get(
                        "phase2_initial_state", args_cli.phase2_initial_state
                    ),
                    "phase2_resolved_initial_state": info0.get(
                        "phase2_resolved_initial_state", ""
                    ),
                    "phase3_initial_state": info0.get(
                        "phase3_initial_state", "center4"
                    ),
                    "phase3_resolved_initial_state": info0.get(
                        "phase3_resolved_initial_state", ""
                    ),
                    "phase4_initial_state": info0.get(
                        "phase4_initial_state", "one_per_coloured_peg"
                    ),
                    "phase4_resolved_initial_state": info0.get(
                        "phase4_resolved_initial_state", ""
                    ),
                    "phase4_ring_order": ",".join(args_cli.phase4_ring_order or ()),
                    "phase4_empirical_entry_index": info0.get(
                        "phase4_empirical_entry_index", ""
                    ),
                    "phase4_empirical_entry_enabled": info0.get(
                        "phase4_empirical_entry_enabled", ""
                    ),
                    "success": bool(info0.get("is_success", False)),
                    "episode_steps": steps + 1,
                    "episode_return": episode_return,
                    "terminal_distance": float(info0.get("terminal_distance", float("nan"))),
                    "green_peg_ring_count": int(info0.get("green_peg_ring_count", 0)),
                    "red_peg_ring_count": int(info0.get("red_peg_ring_count", 0)),
                    "yellow_peg_ring_count": int(info0.get("yellow_peg_ring_count", 0)),
                    "blue_peg_ring_count": int(info0.get("blue_peg_ring_count", 0)),
                    "board_center_ring_count": int(info0.get("board_center_ring_count", 0)),
                    "outside_right_ring_count": int(
                        info0.get("outside_right_ring_count", 0)
                    ),
                    "outside_left_ring_count": int(
                        info0.get("outside_left_ring_count", 0)
                    ),
                    "invalid_commands": invalid_commands,
                    "bimanual_steps": bimanual_steps,
                    "left_active_steps": int(
                        info0.get("episode_left_active_steps", 0)
                    ),
                    "right_active_steps": int(
                        info0.get("episode_right_active_steps", 0)
                    ),
                    "joint_action_changes": int(
                        info0.get("episode_joint_action_changes", 0)
                    ),
                    "free_arm_step_slots": int(
                        info0.get("episode_free_arm_step_slots", 0)
                    ),
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
            if args_cli.park_before_evaluation:
                obs = park_arms_before_evaluation(task_wrapper)
                if isinstance(rl_env, VecNormalize):
                    obs = rl_env.normalize_obs(obs)
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
