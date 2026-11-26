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
from m_dVrk.hrl.constants import (
    PHASE2_DEFAULT_INITIAL_STATE,
    PHASE2_INITIAL_STATES,
)

parser = argparse.ArgumentParser(description="Macchina a stati per il Peg and Ring con RL.")
parser.add_argument("--num_envs", type=int, default=64, help="Numero di ambienti da spawnare in parallelo.")
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for Python, NumPy, Torch, IsaacLab and PPO.",
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
        "Workspace randomization scale used with --randomize_rings: "
        "0 is the fixed layout and 1 is the full range."
    ),
)
parser.add_argument(
    "--phase_entry_park_reset",
    action="store_true",
    help=(
        "Start every episode from the joint configuration produced by the "
        "full-task inter-phase park handoff. Intended for interface fine-tuning."
    ),
)
parser.add_argument(
    "--phase_entry_park_neighborhood_probability",
    type=float,
    default=0.0,
    help="Probability of a safe joint perturbation around parked per episode.",
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
    help=(
        "NPZ bank of paired right/left joint states recorded after real "
        "Phase-3 -> park handoffs. Intended for robust Phase-4 PPO."
    ),
)
parser.add_argument(
    "--phase_entry_empirical_probability",
    type=float,
    default=0.8,
    help="Probability of sampling a real handoff state; remaining resets use exact parked.",
)
parser.add_argument(
    "--phase_entry_empirical_velocity",
    action="store_true",
    help="Restore recorded handoff joint velocities instead of resetting velocity to zero.",
)
parser.add_argument(
    "--adaptive_phase_entry_pose_curriculum",
    action="store_true",
    help=(
        "Success-gated Phase-4 curriculum from canonical arm resets to the "
        "full-task parked pose, while preserving the canonical joint observation frame."
    ),
)
parser.add_argument(
    "--adaptive_phase_entry_pose_start_scale",
    type=float,
    default=0.0,
    help="Initial maximum canonical-to-parked interpolation scale.",
)
parser.add_argument(
    "--adaptive_phase_entry_pose_scale_step",
    type=float,
    default=0.25,
    help="Pose-scale increment after each successful curriculum stage.",
)
parser.add_argument(
    "--adaptive_phase_entry_pose_success_threshold",
    type=float,
    default=0.85,
    help="Rolling success rate required to advance pose difficulty.",
)
parser.add_argument(
    "--adaptive_phase_entry_pose_window_episodes",
    type=int,
    default=128,
    help="Completed same-stage episodes required before advancing pose difficulty.",
)
parser.add_argument(
    "--adaptive_phase_entry_pose_frontier_width",
    type=float,
    default=None,
    help=(
        "Optional width of the interpolation band sampled immediately below "
        "the current pose scale. By default resets sample cumulatively from zero."
    ),
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
    "--adaptive_ring_curriculum",
    action="store_true",
    help=(
        "Advance ring randomization only after a rolling task-success target is met. "
        "Mutually exclusive with static and time-based ring randomization."
    ),
)
parser.add_argument(
    "--adaptive_ring_start_scale",
    type=float,
    default=0.75,
    help="Initial workspace randomization scale for adaptive fine-tuning.",
)
parser.add_argument(
    "--adaptive_ring_scale_step",
    type=float,
    default=0.05,
    help="Randomization-scale increment after each successful adaptive stage.",
)
parser.add_argument(
    "--adaptive_ring_success_threshold",
    type=float,
    default=0.85,
    help="Rolling success rate required to advance adaptive difficulty.",
)
parser.add_argument(
    "--adaptive_ring_window_episodes",
    type=int,
    default=512,
    help="Number of same-scale completed episodes in the adaptive success window.",
)
parser.add_argument(
    "--pretrained_checkpoint",
    type=str,
    default=None,
    help="Path to a zip file of a pre-trained PPO policy (Behavior Cloning).",
)
parser.add_argument(
    "--pretrained_vecnormalize",
    type=str,
    default=None,
    help=(
        "Optional VecNormalize state paired with --pretrained_checkpoint. "
        "Use this when resuming a PPO checkpoint so the critic keeps the "
        "same observation/reward normalization statistics."
    ),
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
    choices=["phase_0", "phase_1", "phase_2", "phase_3", "phase_4"],
    help=(
        "Goal phase. Phase 3 starts with all rings at board centre and "
        "places exactly one arbitrary ring on each coloured peg. Phase 4 "
        "moves two rings into each outside side region."
    ),
)
parser.add_argument(
    "--phase1_initial_state",
    choices=(
        "stack_green",
        "red3_blue1",
        "red1_blue3",
        "red2_blue1_green1",
        "red1_blue2_green1",
        "recovery_mixed",
        "mixed_curriculum",
    ),
    default="stack_green",
    help="Phase-1 reset state used for fixed-start training or recovery probes.",
)
parser.add_argument(
    "--phase2_initial_state",
    choices=PHASE2_INITIAL_STATES,
    default=PHASE2_DEFAULT_INITIAL_STATE,
    help="Phase-2 full-task or recovery reset state used during PPO rollouts.",
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
    "--gamma",
    type=float,
    default=0.99,
    help="PPO reward discount factor.",
)
parser.add_argument(
    "--gae_lambda",
    type=float,
    default=0.95,
    help="PPO generalized-advantage-estimation lambda.",
)
parser.add_argument(
    "--kl_anchor_checkpoint",
    type=str,
    default=None,
    help=(
        "Optional frozen BC checkpoint used as a forward-KL policy anchor. "
        "Requires --kl_anchor_coef > 0 and state-aware action masking."
    ),
)
parser.add_argument(
    "--kl_anchor_coef",
    type=float,
    default=0.0,
    help="Coefficient multiplying KL(BC || current policy) in the PPO loss.",
)
parser.add_argument(
    "--bc_replay_dataset",
    type=str,
    default=None,
    help=(
        "Optional raw .npz demonstration dataset replayed during PPO updates. "
        "Requires --bc_replay_coef > 0, action masking and raw observations."
    ),
)
parser.add_argument(
    "--bc_replay_coef",
    type=float,
    default=0.0,
    help="Coefficient multiplying expert joint-action NLL in the PPO loss.",
)
parser.add_argument(
    "--bc_replay_batch_size",
    type=int,
    default=256,
    help="Expert transitions sampled for each PPO minibatch update.",
)
parser.add_argument(
    "--bc_replay_decision_fraction",
    type=float,
    default=0.0,
    help=(
        "Fraction of every BC replay batch reserved for rows where the joint "
        "expert action changes. Zero preserves uniform replay."
    ),
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
    "--action_mask_mode",
    choices=("guided", "goal_agnostic", "minimal"),
    default="guided",
    help=(
        "Mask profile: 'guided' preserves the historical goal-aware mask; "
        "'goal_agnostic' removes goal targets but keeps manipulation grammar; "
        "'minimal' retains only controller/physical constraints."
    ),
)
parser.add_argument(
    "--soft_mask_initial_penalty",
    type=float,
    default=0.0,
    help=(
        "Initial non-negative logit penalty for actions admitted by the minimal "
        "mask but excluded by the guided mask. Zero disables the soft curriculum."
    ),
)
parser.add_argument(
    "--soft_mask_final_penalty",
    type=float,
    default=0.0,
    help="Final soft-mask logit penalty after annealing (normally zero).",
)
parser.add_argument(
    "--soft_mask_anneal_start_timesteps",
    type=int,
    default=0,
    help="Timestep at which cosine annealing of the soft-mask penalty begins.",
)
parser.add_argument(
    "--soft_mask_anneal_end_timesteps",
    type=int,
    default=None,
    help="Timestep at which the soft-mask penalty reaches its final value.",
)
parser.add_argument(
    "--manipulability_curriculum",
    action="store_true",
    help=(
        "For a minimal mask, ramp the observed manipulability of completed "
        "top rings from the historical guided value (0) to the physical value (1)."
    ),
)
parser.add_argument(
    "--manipulability_ramp_start_timesteps",
    type=int,
    default=250_000,
    help="Timestep at which the manipulability observation starts increasing.",
)
parser.add_argument(
    "--manipulability_ramp_end_timesteps",
    type=int,
    default=1_250_000,
    help="Timestep at which the manipulability observation reaches one.",
)
parser.add_argument(
    "--adaptive_minimal_mask_curriculum",
    action="store_true",
    help=(
        "Advance manipulability scale and then remove the soft-mask penalty "
        "only after sustained successful episode windows."
    ),
)
parser.add_argument("--adaptive_mask_start_scale", type=float, default=0.55)
parser.add_argument("--adaptive_mask_scale_step", type=float, default=0.05)
parser.add_argument("--adaptive_mask_penalty_step", type=float, default=1.0)
parser.add_argument("--adaptive_mask_success_threshold", type=float, default=0.85)
parser.add_argument("--adaptive_mask_window_episodes", type=int, default=256)
parser.add_argument("--adaptive_mask_cooldown_timesteps", type=int, default=32_768)
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
    apply_phase_entry_empirical_bank,
    apply_phase_entry_park_joint_pose,
    configure_phase_entry_pose_reset,
)
from m_dVrk.hrl.tcc import XIRLResnet18, load_tcc_model
from m_dVrk.hrl.wrapper import DVRKVisionHRLWrapper
from m_dVrk.hrl.callbacks import (
    AdaptiveMinimalMaskCurriculumCallback,
    AdaptivePhaseEntryPoseCurriculumCallback,
    AdaptiveRingRandomizationCurriculumCallback,
    EpisodeCsvLoggerCallback,
    BestTerminalDistanceCheckpointCallback,
    FreezePolicyCallback,
    EntropyCoefficientScheduleCallback,
    ManipulabilityObservationCurriculumCallback,
    SoftMaskPenaltyScheduleCallback,
    RingRandomizationCurriculumCallback,
)
from m_dVrk.hrl.masked_policy import (
    MaskedMultiDiscreteActorCriticPolicy,
    masked_policy_custom_objects,
)
from m_dVrk.hrl.anchored_ppo import BCRegularizedPPO
from m_dVrk.hrl.schedules import CosineDecaySchedule, LowHighCosineSchedule
from m_dVrk.hrl.curriculum import (
    AdaptiveMinimalMaskCurriculum,
    AdaptivePhaseEntryPoseCurriculum,
    AdaptiveRingRandomizationCurriculum,
    RingRandomizationCurriculum,
)
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


set_seed(args_cli.seed)

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
    if not 0.0 < args_cli.gamma <= 1.0:
        raise ValueError("--gamma must be in (0, 1]")
    if not 0.0 < args_cli.gae_lambda <= 1.0:
        raise ValueError("--gae_lambda must be in (0, 1]")
    if args_cli.kl_anchor_coef < 0:
        raise ValueError("--kl_anchor_coef must be non-negative")
    if (args_cli.kl_anchor_checkpoint is None) != (args_cli.kl_anchor_coef == 0):
        raise ValueError(
            "--kl_anchor_checkpoint and a positive --kl_anchor_coef must be provided together"
        )
    if args_cli.kl_anchor_checkpoint is not None and not args_cli.action_masking:
        raise ValueError("KL anchoring currently requires --action_masking")
    if args_cli.bc_replay_coef < 0:
        raise ValueError("--bc_replay_coef must be non-negative")
    if (args_cli.bc_replay_dataset is None) != (args_cli.bc_replay_coef == 0):
        raise ValueError(
            "--bc_replay_dataset and a positive --bc_replay_coef must be provided together"
        )
    if args_cli.bc_replay_batch_size <= 0:
        raise ValueError("--bc_replay_batch_size must be positive")
    if not 0.0 <= args_cli.bc_replay_decision_fraction <= 1.0:
        raise ValueError("--bc_replay_decision_fraction must be in [0, 1]")
    if args_cli.bc_replay_dataset is None and args_cli.bc_replay_decision_fraction > 0:
        raise ValueError(
            "--bc_replay_decision_fraction requires --bc_replay_dataset"
        )
    if args_cli.bc_replay_dataset is not None and not args_cli.action_masking:
        raise ValueError("BC replay currently requires --action_masking")
    if args_cli.bc_replay_dataset is not None and not args_cli.disable_obs_normalization:
        raise ValueError(
            "BC replay requires --disable_obs_normalization because the dataset is raw"
        )
    if not 0.0 <= args_cli.ring_randomization_scale <= 1.0:
        raise ValueError("--ring_randomization_scale must be in [0, 1]")
    if not 0.0 <= args_cli.lr_decay_start_fraction <= 1.0:
        raise ValueError("--lr_decay_start_fraction must be in [0, 1]")
    if not 0.0 <= args_cli.ent_decay_start_fraction <= 1.0:
        raise ValueError("--ent_decay_start_fraction must be in [0, 1]")
    if min(args_cli.soft_mask_initial_penalty, args_cli.soft_mask_final_penalty) < 0:
        raise ValueError("soft-mask penalties must be non-negative")
    if args_cli.soft_mask_final_penalty > args_cli.soft_mask_initial_penalty:
        raise ValueError("--soft_mask_final_penalty cannot exceed the initial penalty")
    soft_mask_end_timesteps = (
        args_cli.soft_mask_anneal_end_timesteps
        if args_cli.soft_mask_anneal_end_timesteps is not None
        else args_cli.total_timesteps
    )
    if not (
        0
        <= args_cli.soft_mask_anneal_start_timesteps
        < soft_mask_end_timesteps
        <= args_cli.total_timesteps
    ):
        raise ValueError(
            "soft-mask schedule must satisfy 0 <= start < end <= total_timesteps"
        )
    if args_cli.soft_mask_initial_penalty > 0 and not (
        args_cli.action_masking and args_cli.action_mask_mode == "minimal"
    ):
        raise ValueError(
            "a positive soft-mask penalty requires --action_masking "
            "--action_mask_mode minimal"
        )
    if args_cli.manipulability_curriculum and not (
        args_cli.action_masking and args_cli.action_mask_mode == "minimal"
    ):
        raise ValueError(
            "--manipulability_curriculum requires --action_masking "
            "--action_mask_mode minimal"
        )
    if args_cli.adaptive_minimal_mask_curriculum and not (
        args_cli.action_masking
        and args_cli.action_mask_mode == "minimal"
        and args_cli.soft_mask_initial_penalty > 0
    ):
        raise ValueError(
            "--adaptive_minimal_mask_curriculum requires minimal action masking "
            "and a positive --soft_mask_initial_penalty"
        )
    if args_cli.adaptive_minimal_mask_curriculum and args_cli.manipulability_curriculum:
        raise ValueError(
            "adaptive and timestep-based manipulability curricula are mutually exclusive"
        )
    if args_cli.phase_entry_park_reset and args_cli.adaptive_phase_entry_pose_curriculum:
        raise ValueError(
            "--phase_entry_park_reset and --adaptive_phase_entry_pose_curriculum "
            "are mutually exclusive"
        )
    phase_entry_modes = sum(
        bool(value)
        for value in (
            args_cli.phase_entry_park_reset,
            args_cli.adaptive_phase_entry_pose_curriculum,
            args_cli.phase_entry_empirical_bank is not None,
        )
    )
    if phase_entry_modes > 1:
        raise ValueError(
            "park, adaptive pose, and empirical-bank entry modes are mutually exclusive"
        )
    if args_cli.phase_entry_empirical_bank is not None:
        if args_cli.task_phase not in {"phase_2", "phase_4"}:
            raise ValueError(
                "--phase_entry_empirical_bank is supported for phase_2 and phase_4"
            )
        if not os.path.isfile(args_cli.phase_entry_empirical_bank):
            raise ValueError(
                f"Empirical phase-entry bank not found: {args_cli.phase_entry_empirical_bank}"
            )
    if not 0.0 <= args_cli.phase_entry_empirical_probability <= 1.0:
        raise ValueError("--phase_entry_empirical_probability must be in [0, 1]")
    if not 0.0 <= args_cli.phase_entry_park_neighborhood_probability <= 1.0:
        raise ValueError("park-neighborhood probability must be in [0, 1]")
    if not 0.0 <= args_cli.phase_entry_park_neighborhood_scale <= 1.0:
        raise ValueError("park-neighborhood scale must be in [0, 1]")
    if (
        args_cli.phase_entry_park_neighborhood_probability > 0.0
        and not args_cli.phase_entry_park_reset
    ):
        raise ValueError("park-neighborhood sampling requires --phase_entry_park_reset")
    if args_cli.adaptive_phase_entry_pose_curriculum and args_cli.task_phase != "phase_4":
        raise ValueError(
            "--adaptive_phase_entry_pose_curriculum is currently defined only for phase_4"
        )
    if args_cli.adaptive_phase_entry_pose_frontier_width is not None:
        if not args_cli.adaptive_phase_entry_pose_curriculum:
            raise ValueError(
                "--adaptive_phase_entry_pose_frontier_width requires the pose curriculum"
            )
        if not 0.0 < args_cli.adaptive_phase_entry_pose_frontier_width <= 1.0:
            raise ValueError(
                "--adaptive_phase_entry_pose_frontier_width must be in (0, 1]"
            )
    if args_cli.manipulability_curriculum and not (
        0
        <= args_cli.manipulability_ramp_start_timesteps
        < args_cli.manipulability_ramp_end_timesteps
        <= args_cli.total_timesteps
    ):
        raise ValueError(
            "manipulability curriculum must satisfy "
            "0 <= start < end <= total_timesteps"
        )
    ring_reset_modes = sum(
        bool(value)
        for value in (
            args_cli.randomize_rings,
            args_cli.ring_curriculum,
            args_cli.adaptive_ring_curriculum,
        )
    )
    if ring_reset_modes > 1:
        raise ValueError(
            "--randomize_rings, --ring_curriculum, and "
            "--adaptive_ring_curriculum are mutually exclusive"
        )
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
    adaptive_ring_curriculum = None
    manipulability_curriculum_schedule = None
    adaptive_minimal_mask_curriculum = None
    adaptive_phase_entry_pose_curriculum = None
    if args_cli.adaptive_phase_entry_pose_curriculum:
        adaptive_phase_entry_pose_curriculum = AdaptivePhaseEntryPoseCurriculum(
            start_scale=args_cli.adaptive_phase_entry_pose_start_scale,
            scale_step=args_cli.adaptive_phase_entry_pose_scale_step,
            success_threshold=args_cli.adaptive_phase_entry_pose_success_threshold,
            window_episodes=args_cli.adaptive_phase_entry_pose_window_episodes,
        )
    if args_cli.adaptive_minimal_mask_curriculum:
        adaptive_minimal_mask_curriculum = AdaptiveMinimalMaskCurriculum(
            start_scale=args_cli.adaptive_mask_start_scale,
            scale_step=args_cli.adaptive_mask_scale_step,
            initial_penalty=args_cli.soft_mask_initial_penalty,
            penalty_step=args_cli.adaptive_mask_penalty_step,
            success_threshold=args_cli.adaptive_mask_success_threshold,
            window_episodes=args_cli.adaptive_mask_window_episodes,
            cooldown_timesteps=args_cli.adaptive_mask_cooldown_timesteps,
        )
    if args_cli.manipulability_curriculum:
        manipulability_curriculum_schedule = RingRandomizationCurriculum(
            total_timesteps=args_cli.total_timesteps,
            start_timesteps=args_cli.manipulability_ramp_start_timesteps,
            full_timesteps=args_cli.manipulability_ramp_end_timesteps,
        )
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
    elif args_cli.adaptive_ring_curriculum:
        adaptive_ring_curriculum = AdaptiveRingRandomizationCurriculum(
            start_scale=args_cli.adaptive_ring_start_scale,
            scale_step=args_cli.adaptive_ring_scale_step,
            success_threshold=args_cli.adaptive_ring_success_threshold,
            window_episodes=args_cli.adaptive_ring_window_episodes,
        )

    env_cfg = MDvrkEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.num_rerenders_on_reset = 2
    env_cfg.wait_for_textures = False
    env_cfg.seed = args_cli.seed
    if args_cli.phase_entry_empirical_bank is not None:
        apply_phase_entry_empirical_bank(
            env_cfg,
            bank_path=os.path.abspath(args_cli.phase_entry_empirical_bank),
            empirical_probability=args_cli.phase_entry_empirical_probability,
            use_recorded_velocity=args_cli.phase_entry_empirical_velocity,
            task_phase=args_cli.task_phase,
        )
        print(
            "[INFO] Arm/ring reset mode: empirical full-state bank "
            f"(path={args_cli.phase_entry_empirical_bank}, "
            f"probability={args_cli.phase_entry_empirical_probability:.2f}, "
            f"recorded_velocity={args_cli.phase_entry_empirical_velocity})"
        )
    elif adaptive_phase_entry_pose_curriculum is not None:
        configure_phase_entry_pose_reset(
            env_cfg,
            scale=adaptive_phase_entry_pose_curriculum.current_scale,
            sample_uniform_scale=True,
            sample_frontier_width=args_cli.adaptive_phase_entry_pose_frontier_width,
        )
        print(
            "[INFO] Arm reset mode: adaptive canonical-to-parked pose curriculum "
            f"(start={adaptive_phase_entry_pose_curriculum.current_scale:.3f}, "
            f"step={adaptive_phase_entry_pose_curriculum.scale_step:.3f}, "
            f"threshold={adaptive_phase_entry_pose_curriculum.success_threshold:.2%}, "
            f"window={adaptive_phase_entry_pose_curriculum.window_episodes}, "
            f"frontier_width={args_cli.adaptive_phase_entry_pose_frontier_width})"
        )
    elif args_cli.phase_entry_park_reset:
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
    env_cfg.events.reset_rings.params["randomize"] = bool(
        args_cli.randomize_rings
        or args_cli.ring_curriculum
        or args_cli.adaptive_ring_curriculum
    )
    initial_ring_randomization_scale = (
        ring_curriculum_schedule.value_at(0)
        if ring_curriculum_schedule is not None
        else (
            adaptive_ring_curriculum.current_scale
            if adaptive_ring_curriculum is not None
            else (
                args_cli.ring_randomization_scale
                if args_cli.randomize_rings
                else 0.0
            )
        )
    )
    env_cfg.events.reset_rings.params["randomization_scale"] = (
        initial_ring_randomization_scale
    )
    if adaptive_ring_curriculum is not None:
        print(
            "[INFO] Ring reset mode: adaptive curriculum "
            f"(start={adaptive_ring_curriculum.current_scale:.4f}, "
            f"step={adaptive_ring_curriculum.scale_step:.4f}, "
            f"threshold={adaptive_ring_curriculum.success_threshold:.2%}, "
            f"window={adaptive_ring_curriculum.window_episodes})"
        )
    elif ring_curriculum_schedule is not None:
        print(
            "[INFO] Ring reset mode: curriculum "
            f"(fixed through {ring_curriculum_schedule.start_timesteps}, "
            f"full randomization at {ring_curriculum_schedule.full_timesteps})"
        )
    else:
        reset_description = (
            f"randomized (scale={initial_ring_randomization_scale:.4f})"
            if args_cli.randomize_rings
            else "fixed deterministic"
        )
        print(f"[INFO] Ring reset mode: {reset_description}")
    print(f"[INFO] Simulation device: {env_cfg.sim.device}")
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    if args_cli.phase_entry_empirical_bank is not None:
        isaac_env.phase_entry_pose_scale = 1.0
    elif adaptive_phase_entry_pose_curriculum is not None:
        isaac_env.phase_entry_pose_scale = (
            adaptive_phase_entry_pose_curriculum.current_scale
        )
        isaac_env.phase_entry_pose_frontier_width = (
            args_cli.adaptive_phase_entry_pose_frontier_width
        )
    elif args_cli.phase_entry_park_reset:
        isaac_env.phase_entry_pose_scale = 1.0
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

    hrl_env = DVRKVisionHRLWrapper(
        isaac_env,
        sm,
        tcc,
        task_phase=args_cli.task_phase,
        discount_gamma=args_cli.gamma,
        action_mask_mode=args_cli.action_mask_mode,
        phase1_initial_state=args_cli.phase1_initial_state,
        phase2_initial_state=args_cli.phase2_initial_state,
        manipulability_curriculum_scale=(
            adaptive_minimal_mask_curriculum.current_scale
            if adaptive_minimal_mask_curriculum is not None
            else (
                manipulability_curriculum_schedule.value_at(0)
                if manipulability_curriculum_schedule is not None
                else 1.0
            )
        ),
    )
    rl_env = hrl_env
    
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
    if args_cli.pretrained_vecnormalize is not None:
        if args_cli.pretrained_checkpoint is None:
            raise ValueError(
                "--pretrained_vecnormalize requires --pretrained_checkpoint"
            )
        print(
            f"[PPO] Loading VecNormalize state from: "
            f"{args_cli.pretrained_vecnormalize}",
            flush=True,
        )
        rl_env = VecNormalize.load(args_cli.pretrained_vecnormalize, rl_env)
        rl_env.training = True
        rl_env.norm_reward = True
        print(
            f"[PPO] VecNormalize restored: norm_obs={rl_env.norm_obs}, "
            f"norm_reward={rl_env.norm_reward}",
            flush=True,
        )
    else:
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
        f"n_epochs={PPO_N_EPOCHS} | gamma={args_cli.gamma:g} | "
        f"gae_lambda={args_cli.gae_lambda:g}"
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
    print(
        f"[PPO] Action masking: {args_cli.action_masking} "
        f"(mode={args_cli.action_mask_mode})",
        flush=True,
    )
    if args_cli.kl_anchor_checkpoint is not None:
        print(
            "[PPO] Frozen BC KL anchor: "
            f"checkpoint={args_cli.kl_anchor_checkpoint} "
            f"coefficient={args_cli.kl_anchor_coef:g} direction=KL(BC||PPO)",
            flush=True,
        )
    if args_cli.bc_replay_dataset is not None:
        print(
            "[PPO] Supervised BC replay: "
            f"dataset={args_cli.bc_replay_dataset} "
            f"coefficient={args_cli.bc_replay_coef:g} "
            f"batch_size={args_cli.bc_replay_batch_size}",
            flush=True,
        )
    soft_mask_schedule = None
    if (
        args_cli.soft_mask_initial_penalty > 0
        and adaptive_minimal_mask_curriculum is None
    ):
        soft_mask_schedule = CosineDecaySchedule(
            total_timesteps=soft_mask_end_timesteps,
            decay_start_timesteps=args_cli.soft_mask_anneal_start_timesteps,
            initial_value=args_cli.soft_mask_initial_penalty,
            final_value=args_cli.soft_mask_final_penalty,
        )
        print(
            "[PPO] Soft-mask schedule: "
            f"initial={args_cli.soft_mask_initial_penalty:g} "
            f"final={args_cli.soft_mask_final_penalty:g} "
            f"start={args_cli.soft_mask_anneal_start_timesteps} "
            f"end={soft_mask_end_timesteps}",
            flush=True,
        )
    
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
    if soft_mask_schedule is not None:
        callbacks_list.append(
            SoftMaskPenaltyScheduleCallback(soft_mask_schedule, verbose=1)
        )
    if manipulability_curriculum_schedule is not None:
        callbacks_list.append(
            ManipulabilityObservationCurriculumCallback(
                hrl_env=hrl_env,
                schedule=manipulability_curriculum_schedule,
                verbose=1,
            )
        )
    if adaptive_minimal_mask_curriculum is not None:
        callbacks_list.append(
            AdaptiveMinimalMaskCurriculumCallback(
                hrl_env=hrl_env,
                curriculum=adaptive_minimal_mask_curriculum,
                min_advance_timesteps=freeze_timesteps,
                verbose=1,
            )
        )
    if ring_curriculum_schedule is not None:
        callbacks_list.append(
            RingRandomizationCurriculumCallback(
                raw_env=isaac_env,
                schedule=ring_curriculum_schedule,
                verbose=1,
            )
        )
    if adaptive_ring_curriculum is not None:
        callbacks_list.append(
            AdaptiveRingRandomizationCurriculumCallback(
                raw_env=isaac_env,
                curriculum=adaptive_ring_curriculum,
                min_advance_timesteps=freeze_timesteps,
                verbose=1,
            )
        )
    if adaptive_phase_entry_pose_curriculum is not None:
        callbacks_list.append(
            AdaptivePhaseEntryPoseCurriculumCallback(
                raw_env=isaac_env,
                curriculum=adaptive_phase_entry_pose_curriculum,
                min_advance_timesteps=freeze_timesteps,
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
        policy_kwargs["action_mask_mode"] = args_cli.action_mask_mode
        policy_kwargs["soft_mask_penalty"] = args_cli.soft_mask_initial_penalty

    use_bc_regularization = (
        args_cli.kl_anchor_checkpoint is not None
        or args_cli.bc_replay_dataset is not None
    )
    algorithm_class = BCRegularizedPPO if use_bc_regularization else PPO
    if args_cli.pretrained_checkpoint is not None:
        print(f"[PPO] Loading pre-trained policy from {args_cli.pretrained_checkpoint}...", flush=True)
        custom_objects = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": PPO_N_EPOCHS,
            "ent_coef": args_cli.ent_coef,
            "gamma": args_cli.gamma,
            "gae_lambda": args_cli.gae_lambda,
        }
        if args_cli.action_masking:
            custom_objects.update(
                masked_policy_custom_objects(
                    args_cli.pretrained_checkpoint,
                    args_cli.task_phase,
                    args_cli.action_mask_mode,
                    args_cli.soft_mask_initial_penalty,
                )
            )
        model = algorithm_class.load(
            args_cli.pretrained_checkpoint,
            env=rl_env,
            custom_objects=custom_objects,
            device="cuda",
            tensorboard_log=os.path.join(ARTIFACTS_DIR, "logs", "tensorboard"),
        )
    else:
        print("[PPO] Initializing model with random weights...", flush=True)
        model = algorithm_class(
            MaskedMultiDiscreteActorCriticPolicy if args_cli.action_masking else "MlpPolicy",
            rl_env,
            verbose=1,
            device="cuda",
            seed=args_cli.seed,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=PPO_N_EPOCHS,
            gamma=args_cli.gamma,
            gae_lambda=args_cli.gae_lambda,
            ent_coef=args_cli.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            stats_window_size=1,
            tensorboard_log=os.path.join(ARTIFACTS_DIR, "logs", "tensorboard"),
            policy_kwargs=policy_kwargs,
        )

    # PPO.load restores the seed stored in the checkpoint. Reset it explicitly
    # so independent fine-tuning replicates really use the requested seed.
    model.set_random_seed(args_cli.seed)

    if args_cli.kl_anchor_checkpoint is not None:
        anchor_custom_objects = masked_policy_custom_objects(
            args_cli.kl_anchor_checkpoint,
            args_cli.task_phase,
            args_cli.action_mask_mode,
            args_cli.soft_mask_initial_penalty,
        )
        reference_model = PPO.load(
            args_cli.kl_anchor_checkpoint,
            custom_objects=anchor_custom_objects,
            device="cuda",
        )
        model.set_kl_anchor(reference_model.policy, args_cli.kl_anchor_coef)
        del reference_model
        print(
            "[PPO] Frozen BC reference loaded; KL anchor is active.",
            flush=True,
        )

    if args_cli.bc_replay_dataset is not None:
        replay_data = np.load(args_cli.bc_replay_dataset, allow_pickle=True)
        if not {"obs", "actions"}.issubset(replay_data.files):
            raise ValueError("BC replay dataset must contain obs and actions arrays")
        replay_metadata = (
            replay_data["metadata"].item()
            if "metadata" in replay_data.files
            else {}
        )
        if replay_metadata.get("task_phase") not in {None, args_cli.task_phase}:
            raise ValueError(
                "BC replay dataset task phase does not match --task_phase"
            )
        if replay_metadata.get("action_mask_mode") not in {
            None,
            args_cli.action_mask_mode,
        }:
            raise ValueError(
                "BC replay dataset action-mask mode does not match training"
            )
        model.set_bc_replay(
            replay_data["obs"],
            replay_data["actions"],
            coefficient=args_cli.bc_replay_coef,
            batch_size=args_cli.bc_replay_batch_size,
            decision_fraction=args_cli.bc_replay_decision_fraction,
        )
        decision_count = len(model.bc_replay_decision_indices)
        decision_rate = decision_count / len(replay_data["actions"])
        print(
            "[PPO] BC replay loaded: "
            f"transitions={len(replay_data['obs'])} ",
            f"observation_shape={replay_data['obs'].shape[1:]} ",
            f"action_shape={replay_data['actions'].shape[1:]} ",
            f"decision_rows={decision_count} ({decision_rate:.2%}) ",
            f"sampled_decision_fraction={args_cli.bc_replay_decision_fraction:.2%}",
            flush=True,
        )
        replay_data.close()

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
