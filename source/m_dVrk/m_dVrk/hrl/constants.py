"""Single source of truth for action/entity names and shared constants.

Every script that sends HRL commands (training, collection, evaluation,
BC pretraining, offline diagnostics) must import from here instead of
redefining these mappings locally.
"""

from __future__ import annotations

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
