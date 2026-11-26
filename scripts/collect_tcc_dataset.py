"""Collect scripted camera trajectories for XIRL/TCC pretraining.

The output layout matches the XIRL dataset convention used in this project:

    <output_root>/<split>/<class_name>/<video_id>/<frame>.png

Example:

    python scripts/collect_tcc_dataset.py \
        --num_envs 2 \
        --num_videos 32 \
        --output_root /mnt/data/aiprah/data/sim_dataset_xirl_extra \
        --split train \
        --class_name phase_0 \
        --headless
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

from app_launcher_utils import pin_process_to_requested_cuda_device
from isaaclab.app import AppLauncher
from m_dVrk.hrl.constants import PEG_GREEN

parser = argparse.ArgumentParser(description="Collect scripted dVRK peg-and-ring videos for TCC.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel Isaac environments.")
parser.add_argument("--num_videos", type=int, default=25, help="Number of successful videos to collect.")
parser.add_argument(
    "--output_root",
    type=str,
    default="/mnt/data/aiprah/data/sim_dataset_xirl_extra",
    help="Dataset root. Frames are saved under <root>/<split>/<class_name>/<video_id>/.",
)
parser.add_argument("--split", type=str, default="train", help="Dataset split folder, usually train or valid.")
parser.add_argument("--class_name", type=str, default="phase_0", help="XIRL action-class folder.")
parser.add_argument("--start_index", type=int, default=None, help="First numeric video id. Defaults to next free id.")
parser.add_argument("--id_width", type=int, default=2, help="Zero-padding width for video folders.")
parser.add_argument("--frame_stride", type=int, default=5, help="Save one frame every N env steps.")
parser.add_argument("--max_steps_per_video", type=int, default=6500, help="Discard a video after this many RL steps.")
parser.add_argument("--max_attempts", type=int, default=0, help="Stop after N attempted videos. 0 means unlimited.")
parser.add_argument("--episode_length_s", type=float, default=700.0, help="Long collection episode timeout in seconds.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for scripted variations.")
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
    "--phase2_arm_assignment",
    choices=("same_side", "reverse", "left_only"),
    default="same_side",
    help=(
        "Scripted phase-2 source-to-arm assignment. 'same_side' uses "
        "red->left and blue->right; 'reverse' swaps the two arms; "
        "'left_only' assigns both stacks to the left arm."
    ),
)
parser.add_argument("--target_peg", type=str, default=PEG_GREEN, help="Peg where rings are placed.")
parser.add_argument(
    "--num_rings",
    type=int,
    default=4,
    choices=(1, 2, 3, 4),
    help="Number of rings to place in each video.",
)
parser.add_argument("--camera_width", type=int, default=0, help="Override camera width. 0 keeps env default.")
parser.add_argument("--camera_height", type=int, default=0, help="Override camera height. 0 keeps env default.")
parser.add_argument("--keep_failed", action="store_true", help="Keep failed videos under failed_<video_id> folders.")
parser.add_argument(
    "--initial_rerenders",
    type=int,
    default=4,
    help=(
        "Number of camera renders after applying a phase-specific initial "
        "state. Multiple renders are required to flush stale RTX sensor data."
    ),
)
parser.add_argument(
    "--start_frames_only",
    action="store_true",
    help=(
        "Save one fully rendered initial frame per episode and skip the "
        "scripted trajectory. Useful for phase-router prototypes."
    ),
)
parser.add_argument(
    "--start_settle_steps",
    type=int,
    default=3,
    help=(
        "Idle simulation steps before a start-only capture. This clears the "
        "renderer temporal history without advancing the scripted task."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The camera sensor requires Isaac rendering to be enabled. Make this script
# forgiving so the user does not have to remember the extra launcher flag.
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True

requested_device = getattr(args_cli, "device", None)
args_cli.device = pin_process_to_requested_cuda_device(requested_device)
if requested_device != args_cli.device and requested_device is not None and "cuda" in requested_device:
    print(
        f"[collector] remapped device {requested_device} -> {args_cli.device} "
        f"with CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torchvision.io as io

from isaaclab.envs import ManagerBasedRLEnv
from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg
from m_dVrk.hrl.constants import (
    ARM_ROOT_XY,
    BOARD_CENTER,
    OUTSIDE_LEFT,
    OUTSIDE_RIGHT,
    MAX_RING_PLANAR_REACH,
    RING_NAMES,
    PEG_NAMES,
    PEG_GREEN,
    PEG_RED,
    PEG_YELLOW,
    PEG_BLUE,
)
from m_dVrk.controllers.spartan_state_machine import SPARTANStateMachine
from m_dVrk.controllers.ring_sync import sync_attached_and_frozen_rings

ScriptedStateMachine = SPARTANStateMachine

class VideoSlot:
    def __init__(self):
        self.path: Path | None = None
        self.commands: list[tuple[str, str, str]] = []
        self.command_idx = 0
        self.frame_idx = 0
        self.step_idx = 0
        self.video_id: int | None = None
        self.done = True
        self.targeted_rings: list[str] = []


def next_video_index(dataset_dir: Path, start_index: int | None) -> int:
    if start_index is not None:
        return start_index
    max_idx = -1
    if dataset_dir.exists():
        for child in dataset_dir.iterdir():
            if child.is_dir() and child.name.isdigit():
                max_idx = max(max_idx, int(child.name))
    return max_idx + 1


def choose_arm(mode: str, ring_idx: int) -> str:
    if mode == "right":
        return "right_arm"
    if mode == "left":
        return "left_arm"
    if mode == "alternate":
        return "right_arm" if ring_idx % 2 == 0 else "left_arm"
    return random.choice(["right_arm", "left_arm"])


def choose_reachable_arm(sm, env_id: int, ring_name: str, mode: str, ring_idx: int) -> str:
    """Choose an expert arm that can reach the ring's current stack."""
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
    preferred = choose_arm(mode, ring_idx)
    if preferred in reachable:
        return preferred
    if reachable:
        return random.choice(reachable)
    return min(distances, key=distances.get)


def choose_phase2_arm(sm, env_id: int, ring_name: str, assignment: str) -> str:
    """Choose the scripted PSM for a phase-2 source stack."""
    if assignment == "left_only":
        return "left_arm"
    source = sm.ring_support_peg[ring_name][env_id]
    if source == PEG_RED:
        same_side_arm = "left_arm"
    elif source == PEG_BLUE:
        same_side_arm = "right_arm"
    else:
        raise RuntimeError(f"phase_2 ring {ring_name} has invalid source {source!r}")
    if assignment == "reverse":
        return "right_arm" if same_side_arm == "left_arm" else "left_arm"
    return same_side_arm


def choose_phase3_arm(
    sm,
    env_id: int,
    ring_name: str,
    destination: str,
    mode: str,
    ring_idx: int,
) -> str:
    """Choose an arm that can reach both the centre source and goal peg."""
    # Physical feasibility measured in the Phase-3 smoke test: the right PSM
    # times out on centre->red, whereas the left PSM reaches all four pegs.
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
    preferred = choose_arm(mode, ring_idx)
    if preferred in feasible:
        return preferred
    if feasible:
        return random.choice(feasible)
    return min(distances, key=distances.get)


def build_commands_for_env(sm, env_id: int, class_name: str, ring_order: str, arm_mode: str, target_peg: str, num_rings: int = 4) -> tuple[list[tuple[str, str, str]], list[str]]:
    if class_name == "phase_1":
        rings = sm.get_top_rings_on_peg(env_id, PEG_GREEN, num_rings=4)
        dest_pegs = [PEG_RED, PEG_RED, PEG_BLUE, PEG_BLUE]
        random.shuffle(dest_pegs)
        
        commands: list[tuple[str, str, str]] = []
        for ring_idx, ring_name in enumerate(rings):
            arm = choose_reachable_arm(
                sm, env_id, ring_name, arm_mode, ring_idx
            )
            dest_peg = dest_pegs[ring_idx]
            commands.extend(
                [
                    ("reach", arm, ring_name),
                    ("grasp", arm, ring_name),
                    ("reach", arm, dest_peg),
                    ("release", arm, dest_peg),
                ]
            )
        return commands, rings
    elif class_name == "phase_2":
        # Each source stack must be consumed top-first. Randomly interleave the
        # red and blue stacks to retain trajectory variety without generating
        # physically impossible attempts to grasp a covered ring.
        source_stacks = [
            sm.get_top_rings_on_peg(env_id, PEG_RED, num_rings=2),
            sm.get_top_rings_on_peg(env_id, PEG_BLUE, num_rings=2),
        ]
        rings: list[str] = []
        while any(source_stacks):
            available = [idx for idx, stack in enumerate(source_stacks) if stack]
            source_idx = random.choice(available)
            rings.append(source_stacks[source_idx].pop(0))

        commands = []
        for ring_idx, ring_name in enumerate(rings):
            arm = choose_phase2_arm(
                sm, env_id, ring_name, args_cli.phase2_arm_assignment
            )
            commands.extend(
                [
                    ("reach", arm, ring_name),
                    ("grasp", arm, ring_name),
                    ("reach", arm, BOARD_CENTER),
                    ("release", arm, BOARD_CENTER),
                ]
            )
        return commands, rings
    elif class_name == "phase_3":
        # The centre is a stack: consume it top-first, but randomise the
        # one-to-one ring-to-coloured-peg permutation across demonstrations.
        rings = sm.get_top_rings_on_peg(env_id, BOARD_CENTER, num_rings=4)
        dest_pegs = list(PEG_NAMES)
        if ring_order == "random":
            random.shuffle(dest_pegs)

        commands = []
        for ring_idx, (ring_name, dest_peg) in enumerate(zip(rings, dest_pegs)):
            arm = choose_phase3_arm(
                sm, env_id, ring_name, dest_peg, arm_mode, ring_idx
            )
            commands.extend(
                [
                    ("reach", arm, ring_name),
                    ("grasp", arm, ring_name),
                    ("reach", arm, dest_peg),
                    ("release", arm, dest_peg),
                ]
            )
        return commands, rings
    elif class_name == "phase_4":
        # Phase 4 starts with one ring on each coloured peg.  Use the arm
        # closest to each outside region and keep exactly two rings per side.
        plan = [
            (PEG_RED, "left_arm", OUTSIDE_LEFT),
            (PEG_YELLOW, "right_arm", OUTSIDE_RIGHT),
            (PEG_BLUE, "left_arm", OUTSIDE_LEFT),
            (PEG_GREEN, "right_arm", OUTSIDE_RIGHT),
        ]
        if ring_order == "random":
            random.shuffle(plan)
        commands = []
        rings = []
        for source_peg, arm, destination in plan:
            source_rings = sm.get_top_rings_on_peg(
                env_id, source_peg, num_rings=1
            )
            if len(source_rings) != 1:
                raise RuntimeError(
                    f"Phase-4 source {source_peg} expected one ring, got {source_rings}"
                )
            ring_name = source_rings[0]
            rings.append(ring_name)
            commands.extend(
                [
                    ("reach", arm, ring_name),
                    ("grasp", arm, ring_name),
                    ("reach", arm, destination),
                    ("release", arm, destination),
                ]
            )
        return commands, rings
    else:
        rings = RING_NAMES.copy()
        if ring_order == "random":
            random.shuffle(rings)
        rings = rings[:num_rings]

        commands: list[tuple[str, str, str]] = []
        for ring_idx, ring_name in enumerate(rings):
            arm = choose_arm(arm_mode, ring_idx)
            commands.extend(
                [
                    ("reach", arm, ring_name),
                    ("grasp", arm, ring_name),
                    ("reach", arm, target_peg),
                    ("release", arm, target_peg),
                ]
            )
        return commands, rings


def tensor_env_ids(env, ids: list[int]) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.long, device=env.device)


def save_camera_frame(env, env_id: int, slot: VideoSlot):
    if slot.path is None:
        return
    rgb = env.scene.sensors["camera"].data.output["rgb"][env_id, :, :, :3]
    img = rgb.detach().cpu()
    if img.dtype != torch.uint8:
        max_value = float(img.max()) if img.numel() else 1.0
        if max_value <= 1.0:
            img = (img.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        else:
            img = img.clamp(0.0, 255.0).round().to(torch.uint8)
    img = img.permute(2, 0, 1).contiguous()
    io.write_png(img, str(slot.path / f"{slot.frame_idx:06d}.png"))
    slot.frame_idx += 1


def refresh_camera_after_state_override(env) -> None:
    """Flush stale sensor frames after teleporting the frozen rings.

    ``sync_attached_and_frozen_rings`` writes object poses directly to PhysX.
    Camera output is buffered, so a single call to ``render`` can still expose
    the scene from before that write.  Refreshing transformations and rendering
    a few times makes frame zero correspond to the state machine's phase state.
    """
    import omni.physx

    physx = omni.physx.acquire_physx_interface()
    for _ in range(max(1, args_cli.initial_rerenders)):
        physx.update_transformations(True, True)
        env.sim.render()


def start_video(env, sm, slots, env_id: int, dataset_dir: Path, video_id: int):
    slot = slots[env_id]
    slot.path = dataset_dir / f"temp_env_{env_id}"
    if slot.path.exists():
        shutil.rmtree(slot.path)
    slot.path.mkdir(parents=True, exist_ok=False)

    # Reset state machine first to establish the stacked rings positions/order
    sm.reset_env(env_id, phase=args_cli.class_name)
    
    # Sync the initial state of this environment immediately to place rings in simulator
    sync_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    sync_mask[env_id] = True
    sync_attached_and_frozen_rings(env, sm, sync_mask)
    refresh_camera_after_state_override(env)

    slot.commands, slot.targeted_rings = build_commands_for_env(
        sm, env_id, args_cli.class_name, args_cli.ring_order, args_cli.arm_mode, args_cli.target_peg, args_cli.num_rings
    )
    slot.command_idx = 0
    slot.frame_idx = 0
    slot.step_idx = 0
    slot.video_id = video_id
    slot.done = False
    save_camera_frame(env, env_id, slot)
    print(f"[collector] env={env_id} started video={video_id:04d} commands={len(slot.commands)}")


def collect_start_frames_only(env, sm, dataset_dir: Path, next_id: int) -> int:
    """Collect independent, correctly rendered one-frame start episodes."""
    collected = 0
    while simulation_app.is_running() and collected < args_cli.num_videos:
        batch_size = min(env.num_envs, args_cli.num_videos - collected)
        env_ids = list(range(batch_size))
        env.reset(env_ids=tensor_env_ids(env, env_ids))

        sync_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for env_id in env_ids:
            sm.reset_env(env_id, phase=args_cli.class_name)
            sync_mask[env_id] = True
        sync_attached_and_frozen_rings(env, sm, sync_mask)

        for _ in range(max(0, args_cli.start_settle_steps)):
            env.step(sm.get_action())
            sync_attached_and_frozen_rings(env, sm, sync_mask)
        refresh_camera_after_state_override(env)

        for env_id in env_ids:
            video_id = next_id + collected
            video_path = dataset_dir / f"{video_id:0{max(1, args_cli.id_width)}d}"
            if video_path.exists():
                raise FileExistsError(f"Refusing to overwrite {video_path}")
            video_path.mkdir(parents=True)
            slot = VideoSlot()
            slot.path = video_path
            slot.video_id = video_id
            slot.done = False
            save_camera_frame(env, env_id, slot)
            collected += 1
            print(
                f"[collector] env={env_id} saved start-only "
                f"video={video_id:04d} frames=1"
            )

    return collected


def finish_video(
    slots,
    env_id: int,
    success: bool,
    reason: str,
    keep_failed: bool,
    target_path: Path | None = None,
):
    slot = slots[env_id]
    if slot.path is None:
        slot.done = True
        return

    if success:
        if target_path is not None and target_path != slot.path:
            if target_path.exists():
                shutil.rmtree(target_path)
            slot.path.rename(target_path)
            slot.path = target_path
            try:
                slot.video_id = int(target_path.name)
            except ValueError:
                pass
        print(
            f"[collector] env={env_id} video={slot.video_id:04d} saved "
            f"frames={slot.frame_idx} steps={slot.step_idx}"
        )
    else:
        print(f"[collector] env={env_id} video={slot.video_id:04d} failed: {reason}")
        if keep_failed:
            failed_path = slot.path.parent / f"failed_{slot.video_id:0{max(1, args_cli.id_width)}d}"
            if failed_path.exists():
                shutil.rmtree(failed_path)
            slot.path.rename(failed_path)
        else:
            shutil.rmtree(slot.path, ignore_errors=True)

    slot.path = None
    slot.commands = []
    slot.command_idx = 0
    slot.frame_idx = 0
    slot.step_idx = 0
    slot.video_id = None
    slot.done = True
    slot.targeted_rings = []


def maybe_issue_next_command(sm: ScriptedStateMachine, slots, env_id: int):
    slot = slots[env_id]
    if slot.done or not sm.all_idle(env_id):
        return
    if slot.command_idx > 0:
        previous_verb, previous_arm, previous_target = slot.commands[
            slot.command_idx - 1
        ]
        if previous_verb == "grasp":
            attached = (
                sm.attached_target_r[env_id]
                if previous_arm == "right_arm"
                else sm.attached_target_l[env_id]
            )
            print(
                f"[collector] env={env_id} grasp_result "
                f"{previous_arm} target={previous_target} attached={attached}"
            )
        elif previous_verb == "release":
            print(
                f"[collector] env={env_id} release_result "
                f"target={previous_target} "
                f"center={sm.ring_count_on_peg(env_id, BOARD_CENTER)}"
            )
    if slot.command_idx >= len(slot.commands):
        return
    verb, arm, target = slot.commands[slot.command_idx]
    slot.command_idx += 1
    sm.set_new_triplet(verb, arm, target, env_id)
    print(f"[collector] env={env_id} command={slot.command_idx:02d}/{len(slot.commands):02d} {arm} {verb}->{target}")


def main():
    if args_cli.class_name in {"phase_1", "phase_2", "phase_3", "phase_4"}:
        args_cli.num_rings = 4
    random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    dataset_dir = Path(args_cli.output_root) / args_cli.split / args_cli.class_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    next_id = next_video_index(dataset_dir, args_cli.start_index)
    save_id = next_id

    env_cfg = MDvrkEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.num_rerenders_on_reset = 2
    env_cfg.events.reset_rings.params["randomize"] = bool(args_cli.randomize_rings)
    if args_cli.camera_width > 0:
        env_cfg.scene.camera.width = args_cli.camera_width
    if args_cli.camera_height > 0:
        env_cfg.scene.camera.height = args_cli.camera_height

    print(f"[collector] writing to {dataset_dir}")
    print(f"[collector] ring reset mode: {'randomized' if args_cli.randomize_rings else 'fixed deterministic'}")
    print(f"[collector] simulation device: {env_cfg.sim.device}")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    sm = ScriptedStateMachine(env)
    env.reset()
    for env_id in range(env.num_envs):
        sm.reset_env(env_id, phase=args_cli.class_name)

    if args_cli.start_frames_only:
        successful_videos = collect_start_frames_only(env, sm, dataset_dir, next_id)
        print(
            f"[collector] complete: successful={successful_videos} "
            f"attempted={successful_videos} start_frames_only=True"
        )
        env.close()
        return

    slots = [VideoSlot() for _ in range(env.num_envs)]
    successful_videos = 0
    attempted_videos = 0

    max_initial = args_cli.num_videos
    if args_cli.max_attempts > 0:
        max_initial = min(max_initial, args_cli.max_attempts)
    initial_env_ids = list(range(min(env.num_envs, max_initial)))
    for env_id in initial_env_ids:
        start_video(env, sm, slots, env_id, dataset_dir, next_id)
        next_id += 1
        attempted_videos += 1

    while simulation_app.is_running() and successful_videos < args_cli.num_videos:
        active_ids = [i for i, slot in enumerate(slots) if not slot.done]
        if not active_ids:
            break

        for env_id in active_ids:
            maybe_issue_next_command(sm, slots, env_id)

        _, _, terminated, truncated, _ = env.step(sm.get_action())
        terminated = terminated.to(device=env.device, dtype=torch.bool)
        truncated = truncated.to(device=env.device, dtype=torch.bool)
        isaac_done = terminated | truncated
        active_env_mask = ~isaac_done
        sync_attached_and_frozen_rings(env, sm, active_env_mask)

        reset_ids: list[int] = []
        for env_id in active_ids:
            slot = slots[env_id]
            slot.step_idx += 1

            if bool(isaac_done[env_id].item()):
                reason = "terminated" if bool(terminated[env_id].item()) else "time_out"
                finish_video(slots, env_id, success=False, reason=reason, keep_failed=args_cli.keep_failed)
                sm.reset_env(env_id, phase=args_cli.class_name)
                reset_ids.append(env_id)
                continue

            if slot.step_idx % max(1, args_cli.frame_stride) == 0:
                save_camera_frame(env, env_id, slot)

            if slot.step_idx >= args_cli.max_steps_per_video:
                finish_video(slots, env_id, success=False, reason="max_steps", keep_failed=args_cli.keep_failed)
                reset_ids.append(env_id)
                continue

            if slot.command_idx >= len(slot.commands) and sm.all_idle(env_id):
                success = sm.successful(env_id, phase=args_cli.class_name)
                target_path = None
                if success:
                    save_camera_frame(env, env_id, slot)
                    target_path = dataset_dir / f"{save_id:0{max(1, args_cli.id_width)}d}"
                    save_id += 1
                    successful_videos += 1
                if args_cli.class_name == "phase_1":
                    failure_reason = (
                        f"red={sm.ring_count_on_peg(env_id, PEG_RED)} "
                        f"blue={sm.ring_count_on_peg(env_id, PEG_BLUE)}"
                    )
                elif args_cli.class_name == "phase_2":
                    failure_reason = (
                        f"center={sm.ring_count_on_peg(env_id, BOARD_CENTER)}"
                    )
                elif args_cli.class_name == "phase_3":
                    failure_reason = " ".join(
                        f"{peg}={sm.ring_count_on_peg(env_id, peg)}"
                        for peg in PEG_NAMES
                    )
                elif args_cli.class_name == "phase_4":
                    failure_reason = (
                        f"outside_right={sm.ring_count_on_peg(env_id, OUTSIDE_RIGHT)} "
                        f"outside_left={sm.ring_count_on_peg(env_id, OUTSIDE_LEFT)}"
                    )
                else:
                    failure_reason = f"rings_placed={len(slot.targeted_rings)}"
                finish_video(
                    slots,
                    env_id,
                    success=success,
                    reason=failure_reason,
                    keep_failed=args_cli.keep_failed,
                    target_path=target_path,
                )
                reset_ids.append(env_id)

        if reset_ids:
            env.reset(env_ids=tensor_env_ids(env, reset_ids))
            for env_id in reset_ids:
                sm.reset_env(env_id, phase=args_cli.class_name)

        free_ids = [i for i, slot in enumerate(slots) if slot.done]
        for env_id in free_ids:
            if successful_videos + sum(not slot.done for slot in slots) >= args_cli.num_videos:
                break
            if args_cli.max_attempts > 0 and attempted_videos >= args_cli.max_attempts:
                break
            start_video(env, sm, slots, env_id, dataset_dir, next_id)
            next_id += 1
            attempted_videos += 1

        if args_cli.max_attempts > 0 and attempted_videos >= args_cli.max_attempts:
            active_count = sum(not slot.done for slot in slots)
            if active_count == 0:
                break

    print(f"[collector] complete: successful={successful_videos} attempted={attempted_videos}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
