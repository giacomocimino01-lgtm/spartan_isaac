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
from m_dVrk.hrl.constants import (
    PEG_GREEN,
    PHASE1_DEFAULT_INITIAL_STATE,
    PHASE1_INITIAL_STATES,
    PHASE2_DEFAULT_INITIAL_STATE,
    PHASE2_INITIAL_STATES,
)

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
    "--phase2_initial_state",
    choices=PHASE2_INITIAL_STATES,
    default=PHASE2_DEFAULT_INITIAL_STATE,
    help="Phase-2 full-task or late-stage recovery reset layout.",
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
parser.add_argument(
    "--phase2_strategy",
    choices=("serial", "bimanual"),
    default="serial",
    help=(
        "Phase-2 expert strategy. 'serial' moves one ring at a time; "
        "'bimanual' reaches and grasps one red-stack and one blue-stack "
        "ring concurrently, then releases them at the centre sequentially."
    ),
)
parser.add_argument(
    "--phase3_strategy",
    choices=("serial", "bimanual"),
    default="serial",
    help=(
        "Phase-3 expert strategy. 'bimanual' pipelines the right source "
        "transfer after the left arm exposes the next centre-stack ring."
    ),
)
parser.add_argument(
    "--phase4_strategy",
    choices=("serial", "bimanual"),
    default="serial",
    help=(
        "Phase-4 expert strategy. 'bimanual' transfers one ring per arm "
        "concurrently into the two physically reachable outside regions."
    ),
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
    choices=["phase_0", "phase_1", "phase_2", "phase_3", "phase_4"],
    default="phase_0",
    help="Phase for the environment task.",
)
parser.add_argument(
    "--phase1_initial_state",
    choices=PHASE1_INITIAL_STATES,
    default=PHASE1_DEFAULT_INITIAL_STATE,
    help=(
        "Phase-1 reset state. Imbalanced states collect explicit correction "
        "demonstrations instead of complete green-stack demonstrations."
    ),
)
parser.add_argument(
    "--action_mask_mode",
    choices=("guided", "goal_agnostic", "minimal"),
    default="guided",
    help="Observation semantics used while collecting the expert trajectories.",
)
parser.add_argument(
    "--manipulability_scale",
    type=float,
    default=1.0,
    help=(
        "Observed manipulability of completed top rings for non-guided modes. "
        "Use 1.0 to collect a dataset natively matching the minimal mask."
    ),
)
parser.add_argument(
    "--phase_entry_park_reset",
    action="store_true",
    help=(
        "Start demonstrations from the full-task parked phase-entry joint pose."
    ),
)
parser.add_argument(
    "--phase_entry_park_neighborhood_probability",
    type=float,
    default=0.0,
    help=(
        "Per-episode probability of sampling a small joint-space perturbation "
        "around the parked pose. Requires --phase_entry_park_reset."
    ),
)
parser.add_argument(
    "--phase_entry_park_neighborhood_scale",
    type=float,
    default=1.0,
    help="Scale in [0,1] applied to the safe parked-pose perturbation ellipsoid.",
)
parser.add_argument(
    "--decision_only",
    action="store_true",
    help=(
        "Record one transition only when the scripted expert issues a new "
        "semantic command. Busy low-level State Machine steps are omitted."
    ),
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
    BOARD_CENTER, OUTSIDE_LEFT, OUTSIDE_RIGHT,
    PEG_RED, PEG_YELLOW, PEG_GREEN, PEG_BLUE, target_id_for_phase,
    ARM_ROOT_XY, MAX_RING_PLANAR_REACH,
    apply_phase_entry_park_joint_pose,
)
from m_dVrk.hrl.tcc import XIRLResnet18, load_tcc_model
from m_dVrk.hrl.wrapper import DVRKVisionHRLWrapper
from m_dVrk.hrl.expert_scheduling import (
    Phase2BimanualScheduler,
    Phase3BimanualScheduler,
    Phase4BimanualScheduler,
)
from m_dVrk.controllers.spartan_state_machine import SPARTANStateMachine
from m_dVrk.controllers.ring_sync import get_scene_entity_positions_w, sync_attached_and_frozen_rings


# ─── Helper classes & functions ───────────────────────────────────────────────

class EpisodeSlot:
    """Per-environment state used by the collection loop."""
    def __init__(self):
        self.obs_buffer: list = []
        self.actions_buffer: list = []
        self.decision_indices_buffer: list = []
        self.rgb_buffer: list = []
        self.step_idx: int = 0
        self.done: bool = False
        # High-level command queue: list of (verb, arm, target) triples
        self.commands: list = []
        self.command_idx: int = 0
        self.bimanual_scheduler: (
            Phase2BimanualScheduler
            | Phase3BimanualScheduler
            | Phase4BimanualScheduler
            | None
        ) = None


def _choose_arm(arm_mode: str, ring_idx: int) -> str:
    if arm_mode == "right":
        return "right_arm"
    if arm_mode == "left":
        return "left_arm"
    if arm_mode == "alternate":
        return "right_arm" if ring_idx % 2 == 0 else "left_arm"
    return random.choice(["right_arm", "left_arm"])


def _choose_reachable_arm(sm, env_id: int, ring_name: str, arm_mode: str, ring_idx: int) -> str:
    ring_xy_local = (
        sm.frozen_rings_pose[ring_name][env_id, :2]
        - sm.env.scene.env_origins[env_id, :2]
    )
    distances = {
        arm: torch.linalg.vector_norm(
            ring_xy_local - torch.tensor(root_xy, device=sm.device)
        ).item()
        for arm, root_xy in ARM_ROOT_XY.items()
    }
    reachable = [
        arm for arm, distance in distances.items()
        if distance <= MAX_RING_PLANAR_REACH
    ]
    preferred = _choose_arm(arm_mode, ring_idx)
    if preferred in reachable:
        return preferred
    if reachable:
        return random.choice(reachable)
    return min(distances, key=distances.get)


def _choose_phase2_arm(
    sm,
    env_id: int,
    ring_name: str,
    arm_mode: str,
    ring_idx: int,
) -> str:
    """Sample only empirically feasible phase-2 source/arm pairs."""
    source = sm.ring_support_peg[ring_name][env_id]
    if source == PEG_RED:
        return "left_arm"
    if source == PEG_BLUE:
        # Both arms completed grasp-and-release from the blue stack in the
        # scripted feasibility runs.  Preserve expert diversity here instead
        # of baking the old blue->right strategy into BC.
        return _choose_arm(arm_mode, ring_idx)
    raise RuntimeError(f"phase_2 ring {ring_name} has invalid source {source!r}")


def _choose_phase3_arm(
    sm,
    env_id: int,
    ring_name: str,
    destination: str,
    arm_mode: str,
    ring_idx: int,
) -> str:
    """Choose an arm that can reach both the centre stack and destination."""
    # Scripted smoke tests establish that the right PSM cannot complete the
    # centre->red transfer, while the left PSM can serve every coloured peg.
    if destination == PEG_RED:
        return "left_arm"
    origin_xy = sm.env.scene.env_origins[env_id, :2]
    ring_xy = sm.frozen_rings_pose[ring_name][env_id, :2] - origin_xy
    distances: dict[str, float] = {}
    feasible: list[str] = []
    for arm, root_xy in ARM_ROOT_XY.items():
        destination_xy = (
            sm.get_target_coordinates(destination, arm, env_id)[:2] - origin_xy
        )
        root = torch.tensor(root_xy, device=sm.device)
        worst_distance = max(
            torch.linalg.vector_norm(ring_xy - root).item(),
            torch.linalg.vector_norm(destination_xy - root).item(),
        )
        distances[arm] = worst_distance
        if worst_distance <= MAX_RING_PLANAR_REACH:
            feasible.append(arm)
    preferred = _choose_arm(arm_mode, ring_idx)
    if preferred in feasible:
        return preferred
    if feasible:
        return random.choice(feasible)
    return min(distances, key=distances.get)


def _phase1_transfer_plan(sm, env_id: int) -> tuple[list[str], list[str]]:
    """Return the rings and destinations needed to reach a 2-red/2-blue goal."""
    red_rings = sm.get_top_rings_on_peg(env_id, PEG_RED, num_rings=4)
    blue_rings = sm.get_top_rings_on_peg(env_id, PEG_BLUE, num_rings=4)
    green_rings = sm.get_top_rings_on_peg(env_id, PEG_GREEN, num_rings=4)

    if len(red_rings) > 2:
        return red_rings[: len(red_rings) - 2], [PEG_BLUE] * (len(red_rings) - 2)
    if len(blue_rings) > 2:
        return blue_rings[: len(blue_rings) - 2], [PEG_RED] * (len(blue_rings) - 2)

    destinations = [PEG_RED] * max(0, 2 - len(red_rings))
    destinations += [PEG_BLUE] * max(0, 2 - len(blue_rings))
    random.shuffle(destinations)
    if len(green_rings) != len(destinations):
        raise RuntimeError(
            "Unsupported Phase-1 layout for scripted completion: "
            f"red={len(red_rings)} blue={len(blue_rings)} green={len(green_rings)}"
        )
    return green_rings, destinations


def _choose_phase1_arm(
    sm,
    env_id: int,
    ring_name: str,
    arm_mode: str,
    ring_idx: int,
) -> str:
    """Choose a feasible Phase-1 source arm without constraining the policy."""
    source = sm.ring_support_peg[ring_name][env_id]
    if source == PEG_RED:
        # The same physical limitation measured for Phase 2 applies when a
        # Phase-1 recovery starts from the red stack.
        return "left_arm"
    return _choose_reachable_arm(sm, env_id, ring_name, arm_mode, ring_idx)


def _build_phase2_bimanual_scheduler(
    sm,
    env_id: int,
) -> Phase2BimanualScheduler:
    """Build the asynchronous two-arm Phase-2 expert scheduler."""
    red_rings = sm.get_top_rings_on_peg(env_id, PEG_RED, num_rings=4)
    blue_rings = sm.get_top_rings_on_peg(env_id, PEG_BLUE, num_rings=4)
    paired_count = min(len(red_rings), len(blue_rings))
    pairs = list(zip(red_rings[:paired_count], blue_rings[:paired_count]))
    remaining = red_rings[paired_count:] + blue_rings[paired_count:]
    serial_tail = []
    for ring_idx, ring in enumerate(remaining):
        arm = _choose_phase2_arm(sm, env_id, ring, args_cli.arm_mode, ring_idx)
        serial_tail += [
            ("reach", arm, ring),
            ("grasp", arm, ring),
            ("reach", arm, BOARD_CENTER),
            ("release", arm, BOARD_CENTER),
        ]
    return Phase2BimanualScheduler(pairs, serial_tail)


def _build_phase3_bimanual_scheduler(sm, env_id: int) -> Phase3BimanualScheduler:
    """Build two physically ordered, overlapping centre-stack pairs."""
    rings = sm.get_top_rings_on_peg(env_id, BOARD_CENTER, num_rings=4)
    if len(rings) != 4:
        raise RuntimeError(f"Phase-3 bimanual expert expected 4 centre rings, got {rings}")
    pairs = [
        (rings[0], PEG_RED, rings[1], PEG_GREEN),
        (rings[2], PEG_YELLOW, rings[3], PEG_BLUE),
    ]
    return Phase3BimanualScheduler(pairs)


def _phase4_arm_rings(sm, env_id: int) -> tuple[list[str], list[str]]:
    """Return deterministic, reachable left/right Phase-4 source rings."""
    left_rings = [
        sm.get_top_rings_on_peg(env_id, peg, num_rings=1)[0]
        for peg in (PEG_RED, PEG_BLUE)
    ]
    right_rings = [
        sm.get_top_rings_on_peg(env_id, peg, num_rings=1)[0]
        for peg in (PEG_YELLOW, PEG_GREEN)
    ]
    return left_rings, right_rings


def _build_phase4_bimanual_scheduler(sm, env_id: int) -> Phase4BimanualScheduler:
    left_rings, right_rings = _phase4_arm_rings(sm, env_id)
    pairs = [
        (left_ring, OUTSIDE_LEFT, right_ring, OUTSIDE_RIGHT)
        for left_ring, right_ring in zip(left_rings, right_rings)
    ]
    return Phase4BimanualScheduler(pairs)


def _build_commands(sm, env_id: int, task_phase: str, ring_order: str, arm_mode: str, target_peg: str, num_rings: int) -> list:
    """Build the ordered list of (verb, arm, target) triples for one episode."""
    if task_phase == "phase_1":
        rings, destinations = _phase1_transfer_plan(sm, env_id)
    elif task_phase == "phase_2":
        source_stacks = [
            sm.get_top_rings_on_peg(env_id, PEG_RED, num_rings=2),
            sm.get_top_rings_on_peg(env_id, PEG_BLUE, num_rings=2),
        ]
        rings = []
        if ring_order == "fixed":
            # Deterministically interleave red and blue source stacks.  This
            # gives BC a single expert label for an otherwise identical
            # Phase-2 observation; ``random`` remains available when expert
            # diversity is desired instead of exact-action supervision.
            while any(source_stacks):
                for stack in source_stacks:
                    if stack:
                        rings.append(stack.pop(0))
        else:
            while any(source_stacks):
                available = [idx for idx, stack in enumerate(source_stacks) if stack]
                source_idx = random.choice(available)
                rings.append(source_stacks[source_idx].pop(0))
        destinations = [BOARD_CENTER] * len(rings)
    elif task_phase == "phase_3":
        rings = sm.get_top_rings_on_peg(env_id, BOARD_CENTER, num_rings=4)
        destinations = list(PEG_NAMES)
        if ring_order == "random":
            random.shuffle(destinations)
    elif task_phase == "phase_4":
        left_rings, right_rings = _phase4_arm_rings(sm, env_id)
        rings = [left_rings[0], right_rings[0], left_rings[1], right_rings[1]]
        destinations = [
            OUTSIDE_LEFT,
            OUTSIDE_RIGHT,
            OUTSIDE_LEFT,
            OUTSIDE_RIGHT,
        ]
    else:
        rings = list(RING_NAMES)[:num_rings]
        if ring_order == "random":
            random.shuffle(rings)
        destinations = [target_peg] * len(rings)

    commands = []
    for i, ring in enumerate(rings):
        if task_phase == "phase_2":
            arm = _choose_phase2_arm(sm, env_id, ring, arm_mode, i)
        elif task_phase == "phase_3":
            arm = _choose_phase3_arm(
                sm, env_id, ring, destinations[i], arm_mode, i
            )
        elif task_phase == "phase_4":
            arm = "left_arm" if destinations[i] == OUTSIDE_LEFT else "right_arm"
        elif task_phase == "phase_1":
            arm = _choose_phase1_arm(sm, env_id, ring, arm_mode, i)
        else:
            arm = _choose_arm(arm_mode, i)
        destination = destinations[i]
        commands += [
            ("reach",   arm, ring),
            ("grasp",   arm, ring),
            ("reach",   arm, destination),
            ("release", arm, destination),
        ]
    return commands


def start_episode(isaac_env, sm, slots, env_id: int):
    """Reset an environment slot and arm the command queue for a new episode."""
    slot = slots[env_id]
    slot.obs_buffer = []
    slot.actions_buffer = []
    slot.decision_indices_buffer = []
    slot.rgb_buffer = []
    slot.step_idx = 0
    slot.done = False
    if args_cli.task_phase == "phase_2" and args_cli.phase2_strategy == "bimanual":
        slot.bimanual_scheduler = _build_phase2_bimanual_scheduler(sm, env_id)
        slot.commands = []
    elif args_cli.task_phase == "phase_3" and args_cli.phase3_strategy == "bimanual":
        slot.bimanual_scheduler = _build_phase3_bimanual_scheduler(sm, env_id)
        slot.commands = []
    elif args_cli.task_phase == "phase_4" and args_cli.phase4_strategy == "bimanual":
        slot.bimanual_scheduler = _build_phase4_bimanual_scheduler(sm, env_id)
        slot.commands = []
    else:
        slot.bimanual_scheduler = None
        slot.commands = _build_commands(
            sm,
            env_id,
            args_cli.task_phase,
            args_cli.ring_order,
            args_cli.arm_mode,
            args_cli.target_peg,
            args_cli.num_rings,
        )
    slot.command_idx = 0
    if args_cli.task_phase == "phase_1":
        print(
            f"[collector] Env {env_id} initial Phase-1 layout: "
            f"{sm.describe_phase1_layout(env_id)} | commands={slot.commands}",
            flush=True,
        )


_IDLE_VERB_ID   = VERB_TO_ID["idle"]
_NONE_TARGET_ID = TARGET_TO_ID["None"]


def get_triplet_action_for_env(sm, slot: EpisodeSlot, env_id: int):
    """Return the current joint action and whether a new command was issued."""
    issued_new_command = False
    if slot.bimanual_scheduler is not None:
        triplets = slot.bimanual_scheduler.issue_ready(
            left_idle=sm.arm_idle(env_id, "left_arm"),
            right_idle=sm.arm_idle(env_id, "right_arm"),
        )
        for verb_str, arm_str, tgt_str in triplets:
            sm.set_new_triplet(verb_str, arm_str, tgt_str, env_id)
        if triplets:
            slot.command_idx += 1
            issued_new_command = True
    # Serial experts advance only when the complete state machine is free.
    elif sm.all_idle(env_id) and slot.command_idx < len(slot.commands):
        command = slot.commands[slot.command_idx]
        slot.command_idx += 1
        # Serial plans store a single triplet directly. Bimanual plans store
        # one or two triplets inside the semantic joint decision.
        if len(command) == 3 and isinstance(command[0], str):
            triplets = (command,)
        else:
            triplets = command
        for verb_str, arm_str, tgt_str in triplets:
            sm.set_new_triplet(verb_str, arm_str, tgt_str, env_id)
        issued_new_command = True

    cmd_l = sm.current_triplet_l[env_id]
    cmd_r = sm.current_triplet_r[env_id]

    verb_l = VERB_TO_ID.get(cmd_l["verb"], _IDLE_VERB_ID)   if cmd_l else _IDLE_VERB_ID
    tgt_l  = target_id_for_phase(cmd_l["target"], args_cli.task_phase) if cmd_l else _NONE_TARGET_ID
    verb_r = VERB_TO_ID.get(cmd_r["verb"], _IDLE_VERB_ID)   if cmd_r else _IDLE_VERB_ID
    tgt_r  = target_id_for_phase(cmd_r["target"], args_cli.task_phase) if cmd_r else _NONE_TARGET_ID

    return (verb_l, tgt_l, verb_r, tgt_r), issued_new_command


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():

    if args_cli.task_phase in {"phase_1", "phase_2", "phase_3", "phase_4"}:
        args_cli.num_rings = 4
    if args_cli.phase2_strategy == "bimanual":
        if args_cli.task_phase != "phase_2":
            raise ValueError("--phase2_strategy bimanual requires --task_phase phase_2")
        if args_cli.ring_order != "fixed":
            raise ValueError("--phase2_strategy bimanual requires --ring_order fixed")
    if args_cli.phase3_strategy == "bimanual" and args_cli.task_phase != "phase_3":
        raise ValueError("--phase3_strategy bimanual requires --task_phase phase_3")
    if args_cli.phase4_strategy == "bimanual" and args_cli.task_phase != "phase_4":
        raise ValueError("--phase4_strategy bimanual requires --task_phase phase_4")
    if not 0.0 <= args_cli.manipulability_scale <= 1.0:
        raise ValueError("--manipulability_scale must be in [0, 1]")
    if not 0.0 <= args_cli.phase_entry_park_neighborhood_probability <= 1.0:
        raise ValueError("--phase_entry_park_neighborhood_probability must be in [0, 1]")
    if not 0.0 <= args_cli.phase_entry_park_neighborhood_scale <= 1.0:
        raise ValueError("--phase_entry_park_neighborhood_scale must be in [0, 1]")
    if (
        args_cli.phase_entry_park_neighborhood_probability > 0.0
        and not args_cli.phase_entry_park_reset
    ):
        raise ValueError(
            "--phase_entry_park_neighborhood_probability requires "
            "--phase_entry_park_reset"
        )
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    env_cfg = MDvrkEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.num_rerenders_on_reset = 2
    if args_cli.phase_entry_park_reset:
        apply_phase_entry_park_joint_pose(
            env_cfg,
            neighborhood_probability=(
                args_cli.phase_entry_park_neighborhood_probability
            ),
            neighborhood_scale=args_cli.phase_entry_park_neighborhood_scale,
        )
        print(
            "[collector] Arm reset mode: full-task parked phase-entry pose "
            f"(neighborhood_probability="
            f"{args_cli.phase_entry_park_neighborhood_probability:.2f}, "
            f"neighborhood_scale={args_cli.phase_entry_park_neighborhood_scale:.2f})"
        )
    env_cfg.events.reset_rings.params["randomize"] = bool(args_cli.randomize_rings)

    print(f"[collector] Starting environment on device {env_cfg.sim.device}...")
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    if args_cli.phase_entry_park_reset:
        isaac_env.phase_entry_pose_scale = 1.0
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

    rl_env = DVRKVisionHRLWrapper(
        isaac_env,
        sm,
        tcc,
        task_phase=args_cli.task_phase,
        action_mask_mode=args_cli.action_mask_mode,
        manipulability_curriculum_scale=args_cli.manipulability_scale,
        phase1_initial_state=args_cli.phase1_initial_state,
        phase2_initial_state=args_cli.phase2_initial_state,
    )

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
    all_episode_ids = []
    all_decision_indices = []

    # Reset environment to get initial observations
    obs = rl_env.reset()

    # Start initial episodes
    for env_id in range(args_cli.num_envs):
        start_episode(isaac_env, sm, slots, env_id)

    print(f"[collector] Starting collection of {args_cli.num_episodes} successful episodes...")

    while simulation_app.is_running() and successful_episodes_count < args_cli.num_episodes:
        # Determine high-level commands to step the environment
        step_actions = np.zeros((args_cli.num_envs, 4), dtype=np.int64)
        record_transition = np.zeros(args_cli.num_envs, dtype=bool)
        for env_id in range(args_cli.num_envs):
            slot = slots[env_id]
            if not slot.done:
                action, issued_new_command = get_triplet_action_for_env(sm, slot, env_id)
                step_actions[env_id] = action
                record_transition[env_id] = issued_new_command or not args_cli.decision_only

        # Save current observations and actions to buffer BEFORE stepping the environment.
        # This aligns the state observed at time t with the action executed at time t.
        # Note: the wrapper will modify continuous commands inside step_wait, but it keeps track
        # of the active high-level discrete actions in `prev_actions` which is what we need to record.
        for env_id in range(args_cli.num_envs):
            slot = slots[env_id]
            if not slot.done:
                if record_transition[env_id]:
                    slot.obs_buffer.append(obs[env_id].copy())
                    slot.decision_indices_buffer.append(slot.command_idx - 1)
                if args_cli.save_videos:
                    rgb_frame = rl_env.env.scene.sensors["camera"].data.output["rgb"][env_id, :, :, :3].cpu()
                    slot.rgb_buffer.append(rgb_frame)

        # Step the environment
        next_obs, rewards, dones, infos = rl_env.step(step_actions)

        # Retrieve the actual actions that were executed by the wrapper (handling arm busy overrides)
        executed_actions = rl_env.last_step_actions.cpu().numpy()

        for env_id in range(args_cli.num_envs):
            slot = slots[env_id]
            if not slot.done:
                if record_transition[env_id]:
                    slot.actions_buffer.append(executed_actions[env_id].copy())
                slot.step_idx += 1

                # Check if episode ended
                if dones[env_id]:
                    is_success = infos[env_id].get("is_success", False)
                    # Verify task logical completion
                    sm_success = sm.successful(env_id, phase=args_cli.task_phase)
                    
                    if is_success or sm_success:
                        if args_cli.decision_only:
                            expected_indices = list(range(slot.command_idx))
                            if slot.decision_indices_buffer != expected_indices:
                                print(
                                    "[collector] Episode reached success but the decision "
                                    "trace is incomplete; discarding it. "
                                    f"expected={expected_indices} "
                                    f"actual={slot.decision_indices_buffer}",
                                    flush=True,
                                )
                                start_episode(isaac_env, sm, slots, env_id)
                                continue
                        if successful_episodes_count < args_cli.num_episodes:
                            episode_id = successful_episodes_count
                            successful_episodes_count += 1
                            all_observations.extend(slot.obs_buffer)
                            all_actions.extend(slot.actions_buffer)
                            all_episode_ids.extend([episode_id] * len(slot.obs_buffer))
                            all_decision_indices.extend(slot.decision_indices_buffer)
                            unit = "decisions" if args_cli.decision_only else "steps"
                            print(
                                f"[collector] Episode SUCCESS! Saved {len(slot.obs_buffer)} {unit}. "
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
        episode_ids_arr = np.array(all_episode_ids, dtype=np.int64)
        decision_indices_arr = np.array(all_decision_indices, dtype=np.int64)
        if not (
            len(obs_arr)
            == len(acts_arr)
            == len(episode_ids_arr)
            == len(decision_indices_arr)
        ):
            raise RuntimeError("Collected dataset arrays are not aligned")

        # Metadata to help reproducibility and downstream checks
        metadata = {
            "obs_dim": int(obs_arr.shape[1]) if obs_arr.ndim > 1 else int(obs_arr.size),
            "emb_dim": int((obs_arr.shape[1] - 62 - 86) // 3) if obs_arr.ndim > 1 else None,
            "aux_dim": 62,
            "geom_dim": 86,
            "task_phase": args_cli.task_phase,
            "phase1_initial_state": args_cli.phase1_initial_state,
            "phase2_initial_state": args_cli.phase2_initial_state,
            "action_mask_mode": args_cli.action_mask_mode,
            "manipulability_scale": float(args_cli.manipulability_scale),
            "phase_entry_park_reset": bool(args_cli.phase_entry_park_reset),
            "phase_entry_park_neighborhood_probability": float(
                args_cli.phase_entry_park_neighborhood_probability
            ),
            "phase_entry_park_neighborhood_scale": float(
                args_cli.phase_entry_park_neighborhood_scale
            ),
            "ring_order": args_cli.ring_order,
            "arm_mode": args_cli.arm_mode,
            "phase2_strategy": args_cli.phase2_strategy,
            "num_rings": int(args_cli.num_rings),
            "seed": int(args_cli.seed),
            "record_mode": "decision_only" if args_cli.decision_only else "all_steps",
            "successful_episodes": int(successful_episodes_count),
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
            episode_ids=episode_ids_arr,
            decision_indices=decision_indices_arr,
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
