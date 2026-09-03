"""Script VETTORIZZATO dVRK: RL Multi-Ambiente + XIRL Batched + SPARTAN Hive-Mind.

This file is the PPO training entrypoint. All shared logic (constants,
observations, wrapper, callbacks, TCC model) lives in:
  source/m_dVrk/m_dVrk/hrl/
  source/m_dVrk/m_dVrk/controllers/
"""
import argparse
import os
import sys
import math
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as T
torch.cuda.empty_cache()

from app_launcher_utils import pin_process_to_requested_cuda_device, resolve_tcc_checkpoint
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure

# 1. SETUP INIZIALE DI ISAAC SIM
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Macchina a stati per il Peg and Ring con RL.")
parser.add_argument("--num_envs", type=int, default=64, help="Numero di ambienti da spawnare in parallelo.")
parser.add_argument(
    "--randomize_rings",
    action="store_true",
    help="Randomize ring reset positions. By default, rings reset to a fixed deterministic layout.",
)
parser.add_argument(
    "--ring_curriculum",
    action="store_true",
    help=(
        "Progressively expand ring reset positions from the fixed layout to the "
        "full randomized workspace. Mutually exclusive with --randomize_rings."
    ),
)
parser.add_argument(
    "--ring_curriculum_start_fraction",
    type=float,
    default=0.10,
    help="Training fraction kept at the fixed layout before ring randomization begins.",
)
parser.add_argument(
    "--ring_curriculum_full_fraction",
    type=float,
    default=0.70,
    help="Training fraction at which the full ring randomization area is reached.",
)
parser.add_argument(
    "--pretrained_checkpoint",
    type=str,
    default=None,
    help="Path to a zip file of a pre-trained PPO policy (Behavior Cloning).",
)
parser.add_argument(
    "--freeze_policy_timesteps",
    type=int,
    default=1500000,
    help="Number of timesteps to freeze the policy (actor) weights at the beginning of training (to train the critic first).",
)
parser.add_argument(
    "--task_phase",
    type=str,
    default="phase_0",
    choices=["phase_0", "phase_1"],
    help="Goal phase of the task. 'phase_0' places 4 rings on the green peg, 'phase_1' places 2 on red and 2 on blue (starting stacked on green).",
)
parser.add_argument(
    "--goal_dataset_root",
    type=str,
    default="/mnt/data/aiprah/data/sim_dataset_xirl_extra",
    help="Root directory of the dataset used to compute the goal embedding.",
)
parser.add_argument(
    "--disable_obs_normalization",
    action="store_true",
    help="Disable VecNormalize observation normalization. Useful for raw-observation BC checkpoint checks.",
)
parser.add_argument(
    "--total_timesteps",
    type=int,
    default=5_000_000,
    help="Total PPO training timesteps. Use a small value for ablations.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=3e-4,
    help="PPO learning rate. Lower values are useful when fine-tuning a BC-pretrained actor.",
)
parser.add_argument(
    "--lr_schedule",
    choices=["constant", "low_high_cosine"],
    default="low_high_cosine",
    help="Learning-rate schedule. low_high_cosine uses a low critic LR during actor freeze, then rises and decays.",
)
parser.add_argument(
    "--freeze_learning_rate",
    type=float,
    default=None,
    help="LR during actor freeze. Defaults to one third of --learning_rate.",
)
parser.add_argument(
    "--final_learning_rate",
    type=float,
    default=None,
    help="Final LR after cosine decay. Defaults to one tenth of --learning_rate.",
)
parser.add_argument(
    "--lr_ramp_timesteps",
    type=int,
    default=250_000,
    help="Linear LR ramp duration after actor unfreezing.",
)
parser.add_argument(
    "--lr_decay_start_fraction",
    type=float,
    default=0.70,
    help="Fraction of total training at which cosine LR decay starts.",
)
parser.add_argument(
    "--ent_coef",
    type=float,
    default=0.015,
    help="PPO entropy coefficient. Lower values reduce exploration pressure after BC pretraining.",
)
parser.add_argument(
    "--final_ent_coef",
    type=float,
    default=0.0,
    help="Final entropy coefficient after cosine annealing.",
)
parser.add_argument(
    "--ent_decay_start_fraction",
    type=float,
    default=0.70,
    help="Fraction of total training at which entropy annealing starts.",
)
parser.add_argument(
    "--action_masking",
    action="store_true",
    help="Enable state-aware joint masking while keeping MultiDiscrete([4,10,4,10]). Requires raw observations.",
)
parser.add_argument(
    "--log_suffix",
    type=str,
    default="",
    help="Optional suffix appended to the PPO log directory and checkpoint names.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
requested_device = getattr(args_cli, "device", None)
args_cli.device = pin_process_to_requested_cuda_device(requested_device)
if requested_device != args_cli.device and requested_device is not None and "cuda" in requested_device:
    print(
        f"[INFO] Remapped device {requested_device} -> {args_cli.device} "
        f"with CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

# The camera sensor requires Isaac rendering to be enabled.
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg

# --- Import from the m_dVrk package (single source of truth) ---
from m_dVrk.hrl.constants import (
    VERB_MAP, TARGET_MAP, VERB_TO_ID, TARGET_TO_ID,
    RING_TARGETS, PEG_TARGETS, RING_NAMES, PEG_NAMES,
    IDLE_ACTION,
    INVALID_COMMAND_PENALTY,
    EPISODE_CSV_FILENAME,
    BEST_TERMINAL_DISTANCE_WINDOW_EPISODES,
    BEST_TERMINAL_DISTANCE_PREFIX,
    SUCCESS_SNAPSHOT_PROB, SUCCESS_VIDEO_PROB,
    SUCCESS_DUMP_DIR, SUCCESS_VIDEO_DIR, SUCCESS_VIDEO_FRAME_SKIP,
    REWARD_DEBUG_DUMP_INTERVAL, REWARD_DEBUG_DUMP_ENV_IDS,
    REWARD_DEBUG_DUMP_DIR, REWARD_DEBUG_DUMP_MAX_PER_ENV,
    PPO_TARGET_TRANSITIONS_PER_UPDATE, PPO_PREFERRED_BATCH_SIZE, PPO_N_EPOCHS,
)
from m_dVrk.hrl.tcc import XIRLResnet18, load_tcc_model
from m_dVrk.hrl.wrapper import DVRKVisionHRLWrapper
from m_dVrk.hrl.callbacks import (
    EpisodeCsvLoggerCallback,
    BestTerminalDistanceCheckpointCallback,
    FreezePolicyCallback,
    EntropyCoefficientScheduleCallback,
    RingRandomizationCurriculumCallback,
)
from m_dVrk.hrl.masked_policy import (
    MaskedMultiDiscreteActorCriticPolicy,
    masked_policy_custom_objects,
)
from m_dVrk.hrl.schedules import CosineDecaySchedule, LowHighCosineSchedule
from m_dVrk.hrl.curriculum import RingRandomizationCurriculum
from m_dVrk.controllers.spartan_state_machine import SPARTANStateMachine

# (All constants are now imported from m_dVrk.hrl.constants above)

# ==========================================
# XIRLResnet18, DVRKVisionHRLWrapper, and callbacks are imported
# from m_dVrk.hrl.tcc / m_dVrk.hrl.wrapper / m_dVrk.hrl.callbacks
# ==========================================

# NOTE: XIRLResnet18, DVRKVisionHRLWrapper, EpisodeCsvLoggerCallback,
# BestTerminalDistanceCheckpointCallback, FreezePolicyCallback are all
# imported from the m_dVrk package above — do NOT redefine them here.


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


set_seed(42)

# Load optional project paths from configs
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


def derive_ppo_n_steps(num_envs: int, target_transitions_per_update: int) -> int:
    safe_num_envs = max(1, num_envs)
    safe_target = max(1, target_transitions_per_update)
    return max(1, math.ceil(safe_target / safe_num_envs))


def derive_ppo_batch_size(effective_batch_size: int, preferred_batch_size: int) -> int:
    safe_effective_batch_size = max(1, int(effective_batch_size))
    start = min(max(1, int(preferred_batch_size)), safe_effective_batch_size)
    for batch_size in range(start, 0, -1):
        if safe_effective_batch_size % batch_size == 0:
            return batch_size
    return 1





# ==========================================
# TRAINING MAIN
# ==========================================
def main_train():
    if not 0.0 <= args_cli.lr_decay_start_fraction <= 1.0:
        raise ValueError("--lr_decay_start_fraction must be in [0, 1]")
    if not 0.0 <= args_cli.ent_decay_start_fraction <= 1.0:
        raise ValueError("--ent_decay_start_fraction must be in [0, 1]")
    if args_cli.ring_curriculum and args_cli.randomize_rings:
        raise ValueError("--ring_curriculum and --randomize_rings are mutually exclusive")
    if args_cli.ring_curriculum and not (
        0.0
        <= args_cli.ring_curriculum_start_fraction
        < args_cli.ring_curriculum_full_fraction
        <= 1.0
    ):
        raise ValueError(
            "ring curriculum fractions must satisfy "
            "0 <= start < full <= 1"
        )
    if args_cli.action_masking and not args_cli.disable_obs_normalization:
        raise ValueError(
            "--action_masking currently requires --disable_obs_normalization because "
            "the mask reads the categorical state fields from the raw 244-D observation."
        )
    ring_curriculum_schedule = None
    if args_cli.ring_curriculum:
        ring_curriculum_schedule = RingRandomizationCurriculum(
            total_timesteps=args_cli.total_timesteps,
            start_timesteps=int(
                args_cli.total_timesteps * args_cli.ring_curriculum_start_fraction
            ),
            full_timesteps=int(
                args_cli.total_timesteps * args_cli.ring_curriculum_full_fraction
            ),
        )

    env_cfg = MDvrkEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.num_rerenders_on_reset = 2
    env_cfg.wait_for_textures = False
    env_cfg.seed = 42
    env_cfg.events.reset_rings.params["randomize"] = bool(
        args_cli.randomize_rings or args_cli.ring_curriculum
    )
    initial_ring_randomization_scale = (
        ring_curriculum_schedule.value_at(0)
        if ring_curriculum_schedule is not None
        else float(args_cli.randomize_rings)
    )
    env_cfg.events.reset_rings.params["randomization_scale"] = (
        initial_ring_randomization_scale
    )
    if ring_curriculum_schedule is not None:
        print(
            "[INFO] Ring reset mode: curriculum "
            f"(fixed through {ring_curriculum_schedule.start_timesteps}, "
            f"full randomization at {ring_curriculum_schedule.full_timesteps})"
        )
    else:
        print(
            "[INFO] Ring reset mode: "
            f"{'randomized' if args_cli.randomize_rings else 'fixed deterministic'}"
        )
    print(f"[INFO] Simulation device: {env_cfg.sim.device}")
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    isaac_env.ring_randomization_scale = initial_ring_randomization_scale

    # DEBUG: inspect scene entities
    print("\n[DEBUG] Scene entity types:")
    for name in list(RING_NAMES) + list(PEG_NAMES):
        entity = isaac_env.scene[name]
        print(
            f"{name:12s} | "
            f"type={type(entity)} | "
            f"has_data={hasattr(entity, 'data')} | "
            f"has_root_pos_w={hasattr(entity, 'data') and hasattr(entity.data, 'root_pos_w')} | "
            f"has_get_world_poses={hasattr(entity, 'get_world_poses')}"
        )

    print("[DEBUG] env_origins:")
    print(isaac_env.scene.env_origins[:min(5, isaac_env.num_envs)])
    print()


    sm = SPARTANStateMachine(isaac_env)
    tcc = XIRLResnet18(embedding_size=32).to(isaac_env.device)

    try:
        experiment_dir = f"/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_{args_cli.task_phase.replace('_', '')}_tcc"
        tcc_ckpt_path = resolve_tcc_checkpoint(experiment_dir)
        print(f"[INFO] Loading XIRL weights from: {tcc_ckpt_path}")
        ckpt = torch.load(
            tcc_ckpt_path,
            map_location=isaac_env.device,
        )

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

        print("[INFO] XIRL weights loaded.")
        print("[XIRL] missing keys:", load_result.missing_keys)
        print("[XIRL] unexpected keys:", load_result.unexpected_keys)

    except Exception as e:
        print(f"[ERR] XIRL weights: {e}")
        raise RuntimeError("XIRL checkpoint failed to load. Aborting training.") from e

    tcc.eval()

    rl_env = DVRKVisionHRLWrapper(isaac_env, sm, tcc, task_phase=args_cli.task_phase)
    
    dataset_path = f"{args_cli.goal_dataset_root}/train/{args_cli.task_phase}/"
    print(f"[INFO] Computing goal embedding from dataset path: {dataset_path}")
    proc = T.Compose([T.Resize((112, 112), antialias=True)])
    rl_env.compute_and_set_goal_embedding(tcc, proc, dataset_path, raise_on_error=True)

    rl_env = VecMonitor(rl_env)
    norm_obs = not bool(args_cli.disable_obs_normalization)
    if args_cli.pretrained_checkpoint is not None and norm_obs:
        print(
            "[PPO] Warning: VecNormalize(norm_obs=True) is active with a BC checkpoint. "
            "pretrain_ppo.py trains on raw observations; use --disable_obs_normalization "
            "for the raw-BC freeze ablation.",
            flush=True,
        )
    print(f"[PPO] VecNormalize: norm_obs={norm_obs}, norm_reward=True", flush=True)
    rl_env = VecNormalize(
        rl_env,
        norm_obs=norm_obs,
        norm_reward=True,
        clip_obs=np.inf,
        clip_reward=np.inf,
        training=True,
    )

    log_suffix = args_cli.log_suffix.strip()
    log_suffix = f"_{log_suffix}" if log_suffix else ""
    tmp_path = os.path.join(ARTIFACTS_DIR, "logs", f"random_sb3_log_sim_{args_cli.task_phase}{log_suffix}")
    episode_csv_path = os.path.join(tmp_path, EPISODE_CSV_FILENAME)
    vec_normalize_path = os.path.join(tmp_path, "vecnormalize.pkl")
    new_logger = configure(tmp_path, ["stdout","csv", "tensorboard"])
    checkpoint_callback = CheckpointCallback(
        6000,
        os.path.join(ARTIFACTS_DIR, "checkpoints", "ppo"),
        f"dvrk_ppo_{args_cli.task_phase}{log_suffix}",
        save_vecnormalize=True,
    )
    
    n_steps = derive_ppo_n_steps(args_cli.num_envs, PPO_TARGET_TRANSITIONS_PER_UPDATE)
    effective_batch_size = n_steps * args_cli.num_envs
    batch_size = derive_ppo_batch_size(effective_batch_size, PPO_PREFERRED_BATCH_SIZE)
    freeze_timesteps = (
        args_cli.freeze_policy_timesteps
        if args_cli.pretrained_checkpoint is not None
        else 0
    )
    print(
        f"[PPO] num_envs={args_cli.num_envs} | n_steps={n_steps} | "
        f"transitions_per_update={effective_batch_size} | batch_size={batch_size} | "
        f"n_epochs={PPO_N_EPOCHS}"
    , flush=True)
    freeze_learning_rate = (
        args_cli.freeze_learning_rate
        if args_cli.freeze_learning_rate is not None
        else args_cli.learning_rate / 3.0
    )
    final_learning_rate = (
        args_cli.final_learning_rate
        if args_cli.final_learning_rate is not None
        else args_cli.learning_rate / 10.0
    )
    lr_decay_start = int(args_cli.total_timesteps * args_cli.lr_decay_start_fraction)
    if args_cli.lr_schedule == "low_high_cosine":
        learning_rate = LowHighCosineSchedule(
            total_timesteps=args_cli.total_timesteps,
            freeze_timesteps=freeze_timesteps,
            ramp_timesteps=args_cli.lr_ramp_timesteps,
            decay_start_timesteps=lr_decay_start,
            low_value=freeze_learning_rate,
            high_value=args_cli.learning_rate,
            final_value=final_learning_rate,
        )
        print(
            "[PPO] LR schedule low-high-cosine: "
            f"low={freeze_learning_rate:g} peak={args_cli.learning_rate:g} "
            f"final={final_learning_rate:g} freeze_end={freeze_timesteps} "
            f"ramp={args_cli.lr_ramp_timesteps} decay_start={lr_decay_start}",
            flush=True,
        )
    else:
        learning_rate = args_cli.learning_rate
        print(f"[PPO] Constant learning_rate={args_cli.learning_rate:g}", flush=True)

    entropy_schedule = CosineDecaySchedule(
        total_timesteps=args_cli.total_timesteps,
        decay_start_timesteps=int(args_cli.total_timesteps * args_cli.ent_decay_start_fraction),
        initial_value=args_cli.ent_coef,
        final_value=args_cli.final_ent_coef,
    )
    print(
        f"[PPO] Entropy schedule: initial={args_cli.ent_coef:g} "
        f"final={args_cli.final_ent_coef:g} "
        f"decay_start={entropy_schedule.decay_start_timesteps}",
        flush=True,
    )
    print(f"[PPO] Action masking: {args_cli.action_masking}", flush=True)
    
    episode_csv_callback = EpisodeCsvLoggerCallback(episode_csv_path)
    best_terminal_distance_callback = BestTerminalDistanceCheckpointCallback(
        save_dir=os.path.join(ARTIFACTS_DIR, "checkpoints", "ppo"),
        file_prefix=f"{BEST_TERMINAL_DISTANCE_PREFIX}_{args_cli.task_phase}{log_suffix}",
        window_size=BEST_TERMINAL_DISTANCE_WINDOW_EPISODES,
        verbose=1,
    )
    callbacks_list = [
        checkpoint_callback,
        episode_csv_callback,
        best_terminal_distance_callback,
        EntropyCoefficientScheduleCallback(entropy_schedule, verbose=1),
    ]
    if ring_curriculum_schedule is not None:
        callbacks_list.append(
            RingRandomizationCurriculumCallback(
                raw_env=isaac_env,
                schedule=ring_curriculum_schedule,
                verbose=1,
            )
        )

    if args_cli.freeze_policy_timesteps > 0 and args_cli.pretrained_checkpoint is None:
        print(
            f"[PPO] Warning: Policy freezing requested ({args_cli.freeze_policy_timesteps} timesteps), "
            "but no pre-trained checkpoint was provided. Disabling freeze.",
            flush=True,
        )
    if freeze_timesteps > 0:
        print(
            f"[PPO] Pre-trained checkpoint loaded. Policy will be frozen for the first "
            f"{freeze_timesteps} timesteps.",
            flush=True,
        )
        callbacks_list.append(FreezePolicyCallback(freeze_timesteps=freeze_timesteps))

    callback = CallbackList(callbacks_list)

    policy_kwargs = dict(
        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64],
        ),
        activation_fn=nn.ReLU,
    )
    if args_cli.action_masking:
        policy_kwargs["mask_task_phase"] = args_cli.task_phase

    if args_cli.pretrained_checkpoint is not None:
        print(f"[PPO] Loading pre-trained policy from {args_cli.pretrained_checkpoint}...", flush=True)
        custom_objects = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": PPO_N_EPOCHS,
            "ent_coef": args_cli.ent_coef,
        }
        if args_cli.action_masking:
            custom_objects.update(
                masked_policy_custom_objects(args_cli.pretrained_checkpoint, args_cli.task_phase)
            )
        model = PPO.load(
            args_cli.pretrained_checkpoint,
            env=rl_env,
            custom_objects=custom_objects,
            device="cuda",
            tensorboard_log=os.path.join(ARTIFACTS_DIR, "logs", "tensorboard"),
        )
    else:
        print("[PPO] Initializing model with random weights...", flush=True)
        model = PPO(
            MaskedMultiDiscreteActorCriticPolicy if args_cli.action_masking else "MlpPolicy",
            rl_env,
            verbose=1,
            device="cuda",
            seed=42,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=PPO_N_EPOCHS,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=args_cli.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            stats_window_size=1,
            tensorboard_log=os.path.join(ARTIFACTS_DIR, "logs", "tensorboard"),
            policy_kwargs=policy_kwargs,
        )

    model.set_logger(new_logger)
    
    # === DEBUG: Print policy structure ===
    print("\n[DEBUG] === PPO Policy Parameter Structure ===", flush=True)
    total_params = 0
    for name, param in model.policy.named_parameters():
        total_params += param.numel()
        print(f"  {name:60s} | shape: {param.shape} | numel: {param.numel():10d}", flush=True)
    print(f"[DEBUG] Total parameters in policy: {total_params}", flush=True)
    print()
    
    print("[PPO] Model initialized. Starting learn()...", flush=True)
    model.learn(total_timesteps=args_cli.total_timesteps, log_interval=1, callback=callback)
    print("[PPO] learn() completed. Saving final model...", flush=True)
    checkpoint_dir = os.path.join(ARTIFACTS_DIR, "checkpoints", "ppo")
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_stem = (
        f"dvrk_ppo_finale_{args_cli.task_phase}{log_suffix}"
        if log_suffix
        else "dvrk_ppo_finale"
    )
    final_model_path = os.path.join(checkpoint_dir, final_stem)
    final_vecnormalize_path = os.path.join(
        checkpoint_dir, f"{final_stem}_vecnormalize.pkl"
    )
    model.save(final_model_path)
    rl_env.save(final_vecnormalize_path)
    print(f"[PPO] Final model saved: {final_model_path}.zip", flush=True)
    print(
        f"[PPO] Final VecNormalize saved: {final_vecnormalize_path}",
        flush=True,
    )

    rl_env.save(vec_normalize_path)

if __name__ == "__main__":
    main_train()
    if args_cli.headless:
        # All checkpoints and normalization statistics have been flushed.
        # Kit shutdown can otherwise hang indefinitely in unattended jobs.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    simulation_app.close()
