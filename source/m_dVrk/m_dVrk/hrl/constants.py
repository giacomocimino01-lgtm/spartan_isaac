"""Single source of truth for action/entity names and shared constants.

Every script that sends HRL commands (training, collection, evaluation,
BC pretraining, offline diagnostics) must import from here instead of
redefining these mappings locally.
"""

from __future__ import annotations

import os

import torch

# ---------------------------------------------------------------------------
# Action maps
# ---------------------------------------------------------------------------

VERB_MAP: dict[int, str] = {
    0: "reach",
    1: "grasp",
    2: "release",
    3: "idle",
}

TARGET_MAP: dict[int, str] = {
    0: "ring_red",
    1: "ring_yellow",
    2: "ring_green",
    3: "ring_blue",
    4: "peg_red",
    5: "peg_yellow",
    6: "peg_green",
    7: "peg_blue",
    8: "peg_gray",
    9: "None",
}

VERB_TO_ID: dict[str, int] = {v: k for k, v in VERB_MAP.items()}
TARGET_TO_ID: dict[str, int] = {v: k for k, v in TARGET_MAP.items()}

# Phase 2 reuses the otherwise unavailable ``peg_gray`` action slot as a
# semantic target at the geometric centre of the board.  This deliberately
# preserves the established MultiDiscrete([4, 10, 4, 10]) action contract and
# therefore keeps existing observation/checkpoint dimensions unchanged.
BOARD_CENTER = "board_center"
BOARD_CENTER_TARGET_ID = TARGET_TO_ID["peg_gray"]

# Phase 4 keeps the historical 10-way target head so every Phase 0--3
# checkpoint remains loadable.  The two otherwise unused gray-peg tokens are
# represented by slots 8 and 9: slot 8 is the existing ``peg_gray`` target;
# slot 9 is the phase-local ``peg_gray1`` alias (historically the don't-care
# ``None`` target used by idle). The verb disambiguates ``idle,9`` from the
# non-idle Phase-4 destination, so no output or observation dimension changes.
OUTSIDE_RIGHT = "outside_right"
OUTSIDE_LEFT = "outside_left"
OUTSIDE_RIGHT_TARGET_ID = TARGET_TO_ID["peg_gray"]
PEG_GRAY1_TARGET_ID = TARGET_TO_ID["None"]
OUTSIDE_LEFT_TARGET_ID = PEG_GRAY1_TARGET_ID
OUTSIDE_TARGETS: tuple[str, str] = (OUTSIDE_RIGHT, OUTSIDE_LEFT)


def target_name_for_phase(
    target_id: int,
    task_phase: str,
    verb: str | None = None,
) -> str:
    """Decode an action target, applying phase-specific semantic aliases."""
    if task_phase == "phase_2" and int(target_id) == BOARD_CENTER_TARGET_ID:
        return BOARD_CENTER
    if task_phase == "phase_4":
        if int(target_id) == OUTSIDE_RIGHT_TARGET_ID:
            return OUTSIDE_RIGHT
        if int(target_id) == OUTSIDE_LEFT_TARGET_ID and verb != "idle":
            return OUTSIDE_LEFT
    return TARGET_MAP[int(target_id)]


def target_id_for_phase(target_name: str, task_phase: str) -> int:
    """Encode a semantic target without changing the target-head dimension."""
    if task_phase == "phase_2" and target_name == BOARD_CENTER:
        return BOARD_CENTER_TARGET_ID
    if task_phase == "phase_4":
        if target_name == OUTSIDE_RIGHT:
            return OUTSIDE_RIGHT_TARGET_ID
        if target_name == OUTSIDE_LEFT:
            return OUTSIDE_LEFT_TARGET_ID
    return TARGET_TO_ID[target_name]

RING_TARGETS: set[str] = {t for t in TARGET_MAP.values() if t.startswith("ring_")}
PEG_TARGETS: set[str] = {t for t in TARGET_MAP.values() if t.startswith("peg_")}

# ---------------------------------------------------------------------------
# Entity name lists (ordered — index position matters for observation builders)
# ---------------------------------------------------------------------------

RING_NAMES: list[str] = ["ring_red", "ring_yellow", "ring_green", "ring_blue"]
PEG_NAMES: list[str] = ["peg_red", "peg_yellow", "peg_green", "peg_blue"]

# Used by the state machine (includes extra pegs that are not HRL targets)
ALL_PEG_NAMES: list[str] = ["peg_red", "peg_yellow", "peg_green", "peg_gray", "peg_gray1", "peg_blue"]

# Named peg constants for readability
PEG_RED = "peg_red"
PEG_YELLOW = "peg_yellow"
PEG_GREEN = "peg_green"
PEG_BLUE = "peg_blue"
PEG_GRAY = "peg_gray"

# Phase-1 reset states.  These are physical start-state curricula, not action
# constraints: the policy still has to infer every manipulation command.
PHASE1_INITIAL_STATES: tuple[str, ...] = (
    "stack_green",
    "red3_blue1",
    "red1_blue3",
    "red2_blue1_green1",
    "red1_blue2_green1",
    "recovery_mixed",
    "mixed_curriculum",
)
PHASE1_DEFAULT_INITIAL_STATE = "stack_green"

# Phase-2 reset states used by the targeted recovery curriculum.  ``red2_blue2``
# is the canonical full-task start.  The recovery states put two or three
# already-completed rings at board centre while preserving the same task,
# observation and action representation.
PHASE2_INITIAL_STATES: tuple[str, ...] = (
    "red2_blue2",
    "center2",
    "center3",
    "recovery_mixed",
)
PHASE2_DEFAULT_INITIAL_STATE = "red2_blue2"

# Phase-3 starts from the terminal state of Phase 2: all four rings are
# stacked at the board centre.  The goal is permutation-invariant occupancy of
# the four coloured pegs (exactly one arbitrary ring per peg).
PHASE3_INITIAL_STATES: tuple[str, ...] = ("center4",)
PHASE3_DEFAULT_INITIAL_STATE = "center4"

# Phase 4 follows Phase 3: one arbitrary ring starts on each coloured peg.
# Success requires two rings in each outside release region.
PHASE4_INITIAL_STATES: tuple[str, ...] = ("one_per_coloured_peg",)
PHASE4_DEFAULT_INITIAL_STATE = "one_per_coloured_peg"

# Local robot-root geometry shared by masking and scripted expert generation.
ARM_ROOT_XY: dict[str, tuple[float, float]] = {
    "right_arm": (0.25, -0.15),
    "left_arm": (0.50, -0.15),
}
MAX_RING_PLANAR_REACH: float = 0.22
BOARD_CENTER_LOCAL_X: float = 0.375

# Virtual Phase-4 targets on the table, outside the circular board.  The names
# are arm-relative: the right region is closest to ``right_arm`` and the left
# region to ``left_arm``.  Two slots per region avoid overlapping rings.
OUTSIDE_REGION_LOCAL_XY: dict[str, tuple[float, float]] = {
    OUTSIDE_RIGHT: (0.285, -0.075),
    OUTSIDE_LEFT: (0.465, -0.075),
}
OUTSIDE_SLOT_LATERAL_OFFSETS: tuple[float, float] = (-0.014, 0.014)

# Joint-space reset state measured after the deterministic ``park`` handoff
# used by eval_full_task.py.  Phase-4 interface training can opt into this
# reset distribution without advancing every parallel environment through
# hidden low-level simulation steps.
PHASE_ENTRY_PARK_JOINT_POS: dict[str, dict[str, float]] = {
    "robot_right": {
        "psm_yaw_joint": 1.511144876,
        "psm_pitch_end_joint": -0.497384220,
        "psm_main_insertion_joint": 0.163815215,
        "psm_tool_roll_joint": 0.000033679,
        "psm_tool_pitch_joint": 0.497358710,
        "psm_tool_yaw_joint": -0.523590000,
        "psm_tool_gripper1_joint": -0.000001000,
        "psm_tool_gripper2_joint": 0.523590000,
    },
    "robot_left": {
        "psm_yaw_joint": -1.567942262,
        "psm_pitch_end_joint": -0.772992969,
        "psm_main_insertion_joint": 0.203288972,
        "psm_tool_roll_joint": -0.000097386,
        "psm_tool_pitch_joint": 0.772946119,
        "psm_tool_yaw_joint": 0.523590000,
        "psm_tool_gripper1_joint": -0.000000160,
        "psm_tool_gripper2_joint": 0.523590000,
    },
}

# Per-joint semi-axes of the small ellipsoid used to make the Phase-4 park
# interface robust.  Angular joints are in radians and insertion is in metres.
# Gripper joints are intentionally omitted: Phase 4 must always start open and
# without an attached ring.
PHASE_ENTRY_PARK_JOINT_NOISE_SCALES: dict[str, dict[str, float]] = {
    asset_name: {
        "psm_yaw_joint": 0.025,
        "psm_pitch_end_joint": 0.025,
        "psm_main_insertion_joint": 0.004,
        "psm_tool_roll_joint": 0.035,
        "psm_tool_pitch_joint": 0.025,
        "psm_tool_yaw_joint": 0.035,
    }
    for asset_name in PHASE_ENTRY_PARK_JOINT_POS
}


def configure_phase_entry_pose_reset(
    env_cfg,
    *,
    scale: float,
    sample_uniform_scale: bool,
    sample_frontier_width: float | None = None,
    neighborhood_probability: float = 0.0,
    neighborhood_scale: float = 1.0,
) -> None:
    """Configure physical phase-entry resets without changing observation zero."""

    if not 0.0 <= scale <= 1.0:
        raise ValueError("phase-entry pose scale must be in [0, 1]")

    from m_dVrk.hrl.curriculum import validate_phase_entry_neighborhood

    neighborhood_probability, neighborhood_scale = (
        validate_phase_entry_neighborhood(
            neighborhood_probability,
            neighborhood_scale,
        )
    )

    # Imported lazily to avoid a constants -> task config -> constants cycle.
    from m_dVrk.tasks.manager_based.m_dvrk import mdp

    for asset_name, joint_positions in PHASE_ENTRY_PARK_JOINT_POS.items():
        event_name = "reset_robot_right" if asset_name == "robot_right" else "reset_robot_left"
        event = getattr(env_cfg.events, event_name)
        asset_cfg = event.params["asset_cfg"]
        event.func = mdp.reset_joints_to_interpolated_pose
        event.params = {
            "asset_cfg": asset_cfg,
            "target_joint_positions": dict(joint_positions),
            "scale": float(scale),
            "sample_uniform_scale": bool(sample_uniform_scale),
            "sample_frontier_width": sample_frontier_width,
            # Event order is right then left; both arms share the same alpha.
            "sample_owner": asset_name == "robot_right",
            "neighborhood_joint_scales": dict(
                PHASE_ENTRY_PARK_JOINT_NOISE_SCALES[asset_name]
            ),
            "neighborhood_probability": neighborhood_probability,
            "neighborhood_scale": neighborhood_scale,
            # The right-arm reset samples the episode-level exact/noisy choice
            # and radius.  The left arm reuses both, with its own direction.
            "neighborhood_owner": asset_name == "robot_right",
        }


def apply_phase_entry_park_joint_pose(
    env_cfg,
    *,
    neighborhood_probability: float = 0.0,
    neighborhood_scale: float = 1.0,
) -> None:
    """Reset physically to the full-task park pose with canonical observations."""

    configure_phase_entry_pose_reset(
        env_cfg,
        scale=1.0,
        sample_uniform_scale=False,
        sample_frontier_width=None,
        neighborhood_probability=neighborhood_probability,
        neighborhood_scale=neighborhood_scale,
    )


def apply_phase_entry_empirical_bank(
    env_cfg,
    *,
    bank_path: str,
    empirical_probability: float = 0.8,
    empirical_sample_mode: str = "random",
    use_recorded_velocity: bool = False,
    task_phase: str = "phase_4",
) -> None:
    """Reset a phase from paired empirical robot/ring states."""

    if not 0.0 <= empirical_probability <= 1.0:
        raise ValueError("empirical_probability must be in [0, 1]")
    if empirical_sample_mode not in {"random", "sequential"}:
        raise ValueError("empirical_sample_mode must be 'random' or 'sequential'")

    from m_dVrk.tasks.manager_based.m_dvrk import mdp

    for asset_name, joint_positions in PHASE_ENTRY_PARK_JOINT_POS.items():
        event_name = (
            "reset_robot_right" if asset_name == "robot_right" else "reset_robot_left"
        )
        bank_side = "right" if asset_name == "robot_right" else "left"
        event = getattr(env_cfg.events, event_name)
        asset_cfg = event.params["asset_cfg"]
        event.func = mdp.reset_joints_from_empirical_bank
        event.params = {
            "asset_cfg": asset_cfg,
            "bank_path": bank_path,
            "bank_side": bank_side,
            "fallback_joint_positions": dict(joint_positions),
            "empirical_probability": float(empirical_probability),
            "empirical_sample_mode": empirical_sample_mode,
            "sample_owner": asset_name == "robot_right",
            "use_recorded_velocity": bool(use_recorded_velocity),
            "expected_task_phase": task_phase,
        }

# ---------------------------------------------------------------------------
# Default IDLE action tensor (created lazily on first access, CPU)
# ---------------------------------------------------------------------------

def make_idle_action() -> torch.Tensor:
    """Return a CPU tensor [verb_l=idle, tgt_l=None, verb_r=idle, tgt_r=None]."""
    return torch.tensor(
        [
            VERB_TO_ID["idle"], TARGET_TO_ID["None"],   # left arm
            VERB_TO_ID["idle"], TARGET_TO_ID["None"],   # right arm
        ],
        dtype=torch.long,
    )


IDLE_ACTION: torch.Tensor = make_idle_action()

# ---------------------------------------------------------------------------
# Observation dimensions (single source of truth — must match wrapper + BC)
# ---------------------------------------------------------------------------

STACK_SIZE: int = 3
EMB_DIM: int = 32
AUX_DIM: int = 62
GEOM_DIM: int = 86
OBS_DIM: int = STACK_SIZE * EMB_DIM + AUX_DIM + GEOM_DIM  # 244

# ---------------------------------------------------------------------------
# Reward / training hyperparameter defaults (overridable via CLI / YAML)
# ---------------------------------------------------------------------------

INVALID_COMMAND_PENALTY: float = 1.0
SUCCESS_BONUS: float = 100.0

# Episode CSV logging
EPISODE_CSV_FILENAME: str = "episode_progress.csv"

# Best-terminal-distance checkpoint callback
BEST_TERMINAL_DISTANCE_WINDOW_EPISODES: int = 10
BEST_TERMINAL_DISTANCE_PREFIX: str = "dvrk_ppo_best_terminal_distance"

# Success snapshot / video dump
SUCCESS_SNAPSHOT_PROB: float = 0.10    # 10%: save PNG on each success
SUCCESS_VIDEO_PROB: float = 0.02       # 2%:  save MP4 on each success
SUCCESS_DUMP_DIR: str = "success_snapshots"
SUCCESS_VIDEO_DIR: str = "success_videos"
SUCCESS_VIDEO_FRAME_SKIP: int = 5      # Accumulate 1 frame every N steps

# Reward debug image dump
REWARD_DEBUG_DUMP_INTERVAL: int = 500
REWARD_DEBUG_DUMP_ENV_IDS: tuple[int, ...] = (0,)
REWARD_DEBUG_DUMP_DIR: str = "reward_debug_frames"
REWARD_DEBUG_DUMP_MAX_PER_ENV: int = 200

# Allow project-level overrides via `configs/defaults.yaml` (optional)
try:
    import yaml
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    cfg_path = os.path.join(repo_root, "configs", "defaults.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        paths = cfg.get("paths", {})
        SUCCESS_DUMP_DIR = paths.get("success_dump_dir", SUCCESS_DUMP_DIR)
        SUCCESS_VIDEO_DIR = paths.get("success_video_dir", SUCCESS_VIDEO_DIR)
        REWARD_DEBUG_DUMP_DIR = paths.get("reward_debug_dump_dir", REWARD_DEBUG_DUMP_DIR)
except Exception:
    # PyYAML may not be installed in all environments; skip quietly
    pass

# PPO update schedule defaults
PPO_TARGET_TRANSITIONS_PER_UPDATE: int = 8192
PPO_PREFERRED_BATCH_SIZE: int = 2048
PPO_N_EPOCHS: int = 5
PPO_LEARNING_RATE: float = 3e-4
PPO_ENT_COEF: float = 0.015
