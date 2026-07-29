# """Collect scripted camera trajectories for XIRL/TCC pretraining.

# The output layout matches the XIRL dataset convention used in this project:

#     <output_root>/<split>/<class_name>/<video_id>/<frame>.png

# Example:

#     python scripts/collect_tcc_dataset.py \
#         --num_envs 8 \
#         --num_videos 32 \
#         --output_root /mnt/data/aiprah/data/sim_dataset_xirl_extra \
#         --split train \
#         --class_name phase_0 \
#         --headless
# """

# from __future__ import annotations

# import argparse
# import os
# import random
# import shutil
# from pathlib import Path

# from app_launcher_utils import pin_process_to_requested_cuda_device
# from isaaclab.app import AppLauncher

# parser = argparse.ArgumentParser(description="Collect scripted dVRK peg-and-ring videos for TCC.")
# parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel Isaac environments.")
# parser.add_argument("--num_videos", type=int, default=25, help="Number of successful videos to collect.")
# parser.add_argument(
#     "--output_root",
#     type=str,
#     default="/mnt/data/aiprah/data/sim_dataset_xirl_extra",
#     help="Dataset root. Frames are saved under <root>/<split>/<class_name>/<video_id>/.",
# )
# parser.add_argument("--split", type=str, default="train", help="Dataset split folder, usually train or valid.")
# parser.add_argument("--class_name", type=str, default="phase_0", help="XIRL action-class folder.")
# parser.add_argument("--start_index", type=int, default=None, help="First numeric video id. Defaults to next free id.")
# parser.add_argument("--id_width", type=int, default=2, help="Zero-padding width for video folders.")
# parser.add_argument("--frame_stride", type=int, default=5, help="Save one frame every N env steps.")
# parser.add_argument("--max_steps_per_video", type=int, default=6500, help="Discard a video after this many RL steps.")
# parser.add_argument("--max_attempts", type=int, default=0, help="Stop after N attempted videos. 0 means unlimited.")
# parser.add_argument("--episode_length_s", type=float, default=700.0, help="Long collection episode timeout in seconds.")
# parser.add_argument("--seed", type=int, default=42, help="Random seed for scripted variations.")
# parser.add_argument(
#     "--randomize_rings",
#     action="store_true",
#     help="Use the environment random ring reset instead of the fixed deterministic layout.",
# )
# parser.add_argument(
#     "--ring_order",
#     choices=("random", "fixed"),
#     default="random",
#     help="Ring order inside each scripted video.",
# )
# parser.add_argument(
#     "--arm_mode",
#     choices=("right", "left", "random", "alternate"),
#     default="random",
#     help="Arm selection for each ring.",
# )
# parser.add_argument("--target_peg", type=str, default="peg_green", help="Peg where rings are placed.")
# parser.add_argument("--camera_width", type=int, default=0, help="Override camera width. 0 keeps env default.")
# parser.add_argument("--camera_height", type=int, default=0, help="Override camera height. 0 keeps env default.")
# parser.add_argument("--keep_failed", action="store_true", help="Keep failed videos under failed_<video_id> folders.")
# AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()

# # The camera sensor requires Isaac rendering to be enabled. Make this script
# # forgiving so the user does not have to remember the extra launcher flag.
# if not getattr(args_cli, "enable_cameras", False):
#     args_cli.enable_cameras = True

# requested_device = getattr(args_cli, "device", None)
# args_cli.device = pin_process_to_requested_cuda_device(requested_device)
# if requested_device != args_cli.device and requested_device is not None and "cuda" in requested_device:
#     print(
#         f"[collector] remapped device {requested_device} -> {args_cli.device} "
#         f"with CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
#     )

# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# import torch
# import torchvision.io as io

# from isaaclab.envs import ManagerBasedRLEnv
# from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg
# from parallel_env import RING_NAMES, SPARTANStateMachine, sync_attached_and_frozen_rings

# ScriptedStateMachine = SPARTANStateMachine

# class VideoSlot:
#     def __init__(self):
#         self.path: Path | None = None
#         self.commands: list[tuple[str, str, str]] = []
#         self.command_idx = 0
#         self.frame_idx = 0
#         self.step_idx = 0
#         self.video_id: int | None = None
#         self.done = True


# def next_video_index(dataset_dir: Path, start_index: int | None) -> int:
#     if start_index is not None:
#         return start_index
#     max_idx = -1
#     if dataset_dir.exists():
#         for child in dataset_dir.iterdir():
#             if child.is_dir() and child.name.isdigit():
#                 max_idx = max(max_idx, int(child.name))
#     return max_idx + 1


# def choose_arm(mode: str, ring_idx: int) -> str:
#     if mode == "right":
#         return "right_arm"
#     if mode == "left":
#         return "left_arm"
#     if mode == "alternate":
#         return "right_arm" if ring_idx % 2 == 0 else "left_arm"
#     return random.choice(["right_arm", "left_arm"])


# def build_commands(ring_order: str, arm_mode: str, target_peg: str) -> list[tuple[str, str, str]]:
#     rings = RING_NAMES.copy()
#     if ring_order == "random":
#         random.shuffle(rings)

#     commands: list[tuple[str, str, str]] = []
#     for ring_idx, ring_name in enumerate(rings):
#         arm = choose_arm(arm_mode, ring_idx)
#         commands.extend(
#             [
#                 ("reach", arm, ring_name),
#                 ("grasp", arm, ring_name),
#                 ("reach", arm, target_peg),
#                 ("release", arm, target_peg),
#             ]
#         )
#     return commands


# def tensor_env_ids(env, ids: list[int]) -> torch.Tensor:
#     return torch.tensor(ids, dtype=torch.long, device=env.device)


# def save_camera_frame(env, env_id: int, slot: VideoSlot):
#     if slot.path is None:
#         return
#     rgb = env.scene.sensors["camera"].data.output["rgb"][env_id, :, :, :3]
#     img = rgb.detach().cpu()
#     if img.dtype != torch.uint8:
#         max_value = float(img.max()) if img.numel() else 1.0
#         if max_value <= 1.0:
#             img = (img.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
#         else:
#             img = img.clamp(0.0, 255.0).round().to(torch.uint8)
#     img = img.permute(2, 0, 1).contiguous()
#     io.write_png(img, str(slot.path / f"{slot.frame_idx:06d}.png"))
#     slot.frame_idx += 1


# def start_video(env, sm: ScriptedStateMachine, slots, env_id: int, dataset_dir: Path, video_id: int):
#     slot = slots[env_id]
#     slot.path = dataset_dir / f"{video_id:0{max(1, args_cli.id_width)}d}"
#     slot.path.mkdir(parents=True, exist_ok=False)
#     slot.commands = build_commands(args_cli.ring_order, args_cli.arm_mode, args_cli.target_peg)
#     slot.command_idx = 0
#     slot.frame_idx = 0
#     slot.step_idx = 0
#     slot.video_id = video_id
#     slot.done = False
#     sm.reset_env(env_id)
#     save_camera_frame(env, env_id, slot)
#     print(f"[collector] env={env_id} started video={video_id:04d} commands={len(slot.commands)}")


# def finish_video(slots, env_id: int, success: bool, reason: str, keep_failed: bool):
#     slot = slots[env_id]
#     if slot.path is None:
#         slot.done = True
#         return

#     if success:
#         print(
#             f"[collector] env={env_id} video={slot.video_id:04d} saved "
#             f"frames={slot.frame_idx} steps={slot.step_idx}"
#         )
#     else:
#         print(f"[collector] env={env_id} video={slot.video_id:04d} failed: {reason}")
#         if keep_failed:
#             failed_path = slot.path.with_name(f"failed_{slot.path.name}")
#             if failed_path.exists():
#                 shutil.rmtree(failed_path)
#             slot.path.rename(failed_path)
#         else:
#             shutil.rmtree(slot.path, ignore_errors=True)

#     slot.path = None
#     slot.commands = []
#     slot.command_idx = 0
#     slot.frame_idx = 0
#     slot.step_idx = 0
#     slot.video_id = None
#     slot.done = True


# def maybe_issue_next_command(sm: ScriptedStateMachine, slots, env_id: int):
#     slot = slots[env_id]
#     if slot.done or not sm.all_idle(env_id):
#         return
#     if slot.command_idx >= len(slot.commands):
#         return
#     verb, arm, target = slot.commands[slot.command_idx]
#     slot.command_idx += 1
#     sm.set_new_triplet(verb, arm, target, env_id)
#     print(f"[collector] env={env_id} command={slot.command_idx:02d}/{len(slot.commands):02d} {arm} {verb}->{target}")


# def main():
#     random.seed(args_cli.seed)
#     torch.manual_seed(args_cli.seed)

#     dataset_dir = Path(args_cli.output_root) / args_cli.split / args_cli.class_name
#     dataset_dir.mkdir(parents=True, exist_ok=True)
#     next_id = next_video_index(dataset_dir, args_cli.start_index)

#     env_cfg = MDvrkEnvCfg()
#     env_cfg.scene.num_envs = args_cli.num_envs
#     env_cfg.seed = args_cli.seed
#     env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
#     env_cfg.episode_length_s = args_cli.episode_length_s
#     env_cfg.num_rerenders_on_reset = 2
#     env_cfg.events.reset_rings.params["randomize"] = bool(args_cli.randomize_rings)
#     if args_cli.camera_width > 0:
#         env_cfg.scene.camera.width = args_cli.camera_width
#     if args_cli.camera_height > 0:
#         env_cfg.scene.camera.height = args_cli.camera_height

#     print(f"[collector] writing to {dataset_dir}")
#     print(f"[collector] ring reset mode: {'randomized' if args_cli.randomize_rings else 'fixed deterministic'}")
#     print(f"[collector] simulation device: {env_cfg.sim.device}")
#     env = ManagerBasedRLEnv(cfg=env_cfg)
#     sm = ScriptedStateMachine(env)
#     env.reset()
#     for env_id in range(env.num_envs):
#         sm.reset_env(env_id)

#     slots = [VideoSlot() for _ in range(env.num_envs)]
#     successful_videos = 0
#     attempted_videos = 0

#     max_initial = args_cli.num_videos
#     if args_cli.max_attempts > 0:
#         max_initial = min(max_initial, args_cli.max_attempts)
#     initial_env_ids = list(range(min(env.num_envs, max_initial)))
#     for env_id in initial_env_ids:
#         start_video(env, sm, slots, env_id, dataset_dir, next_id)
#         next_id += 1
#         attempted_videos += 1

#     while simulation_app.is_running() and successful_videos < args_cli.num_videos:
#         active_ids = [i for i, slot in enumerate(slots) if not slot.done]
#         if not active_ids:
#             break

#         for env_id in active_ids:
#             maybe_issue_next_command(sm, slots, env_id)

#         _, _, terminated, truncated, _ = env.step(sm.get_action())
#         terminated = terminated.to(device=env.device, dtype=torch.bool)
#         truncated = truncated.to(device=env.device, dtype=torch.bool)
#         isaac_done = terminated | truncated
#         active_env_mask = ~isaac_done
#         sync_attached_and_frozen_rings(env, sm, active_env_mask)

#         reset_ids: list[int] = []
#         for env_id in active_ids:
#             slot = slots[env_id]
#             slot.step_idx += 1

#             if bool(isaac_done[env_id].item()):
#                 reason = "terminated" if bool(terminated[env_id].item()) else "time_out"
#                 finish_video(slots, env_id, success=False, reason=reason, keep_failed=args_cli.keep_failed)
#                 sm.reset_env(env_id)
#                 reset_ids.append(env_id)
#                 continue

#             if slot.step_idx % max(1, args_cli.frame_stride) == 0:
#                 save_camera_frame(env, env_id, slot)

#             if slot.step_idx >= args_cli.max_steps_per_video:
#                 finish_video(slots, env_id, success=False, reason="max_steps", keep_failed=args_cli.keep_failed)
#                 reset_ids.append(env_id)
#                 continue

#             if slot.command_idx >= len(slot.commands) and sm.all_idle(env_id):
#                 success = sm.successful(env_id, args_cli.target_peg)
#                 if success:
#                     save_camera_frame(env, env_id, slot)
#                     successful_videos += 1
#                 finish_video(
#                     slots,
#                     env_id,
#                     success=success,
#                     reason=f"green_count={sm.green_peg_ring_count(env_id)}",
#                     keep_failed=args_cli.keep_failed,
#                 )
#                 reset_ids.append(env_id)

#         if reset_ids:
#             env.reset(env_ids=tensor_env_ids(env, reset_ids))
#             for env_id in reset_ids:
#                 sm.reset_env(env_id)

#         free_ids = [i for i, slot in enumerate(slots) if slot.done]
#         for env_id in free_ids:
#             if successful_videos + sum(not slot.done for slot in slots) >= args_cli.num_videos:
#                 break
#             if args_cli.max_attempts > 0 and attempted_videos >= args_cli.max_attempts:
#                 break
#             start_video(env, sm, slots, env_id, dataset_dir, next_id)
#             next_id += 1
#             attempted_videos += 1

#         if args_cli.max_attempts > 0 and attempted_videos >= args_cli.max_attempts:
#             active_count = sum(not slot.done for slot in slots)
#             if active_count == 0:
#                 break

#     print(f"[collector] complete: successful={successful_videos} attempted={attempted_videos}")
#     env.close()


# if __name__ == "__main__":
#     main()
#     simulation_app.close()

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
parser.add_argument("--target_peg", type=str, default="peg_green", help="Peg where rings are placed.")
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
from parallel_env import RING_NAMES, SPARTANStateMachine, sync_attached_and_frozen_rings

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


def build_commands_for_env(sm, env_id: int, class_name: str, ring_order: str, arm_mode: str, target_peg: str, num_rings: int = 4) -> tuple[list[tuple[str, str, str]], list[str]]:
    if class_name == "phase_1":
        rings = sm.get_top_rings_on_peg(env_id, "peg_green", num_rings=4)
        dest_pegs = ["peg_red", "peg_red", "peg_blue", "peg_blue"]
        random.shuffle(dest_pegs)
        
        commands: list[tuple[str, str, str]] = []
        for ring_idx, ring_name in enumerate(rings):
            arm = choose_arm(arm_mode, ring_idx)
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
    import omni.physx
    omni.physx.acquire_physx_interface().update_transformations(True, True)
    env.sim.render()

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
    if slot.command_idx >= len(slot.commands):
        return
    verb, arm, target = slot.commands[slot.command_idx]
    slot.command_idx += 1
    sm.set_new_triplet(verb, arm, target, env_id)
    print(f"[collector] env={env_id} command={slot.command_idx:02d}/{len(slot.commands):02d} {arm} {verb}->{target}")


def main():
    if args_cli.class_name == "phase_1":
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
                finish_video(
                    slots,
                    env_id,
                    success=success,
                    reason=(
                        f"red={sm.ring_count_on_peg(env_id, 'peg_red')} "
                        f"blue={sm.ring_count_on_peg(env_id, 'peg_blue')}"
                        if args_cli.class_name == "phase_1"
                        else f"rings_placed={len(slot.targeted_rings)}"
                    ),
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