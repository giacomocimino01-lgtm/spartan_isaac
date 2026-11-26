#!/usr/bin/env python3
"""Run deterministic multi-phase inference in one continuous scene."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.io as io
import torchvision.transforms as T
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--phase",
    nargs=4,
    action="append",
    required=True,
    metavar=("NAME", "POLICY", "TCC_CHECKPOINT", "GOAL_DATASET"),
    help="Repeat once per phase: name, PPO checkpoint, TCC checkpoint and dataset.",
)
parser.add_argument(
    "--start_embeddings",
    default=None,
    help="start_embeddings.pt produced by evaluate_phase_router.py.",
)
parser.add_argument(
    "--router_mode",
    choices=("raw", "calibrated", "shadow"),
    default="raw",
    help=(
        "'raw' routes after a separate completion event using nearest-start; "
        "'calibrated' evaluates the Section 5 transition score on every frame "
        "and lets it control the switch; 'shadow' evaluates and logs the same "
        "score while phase_success_mode remains in control."
    ),
)
parser.add_argument(
    "--calibrated_router_artifact",
    default=None,
    help="calibrated_router.pt produced by calibrate_phase_router.py.",
)
parser.add_argument(
    "--soft_end_reference_bank",
    default=None,
    help=(
        "Optional complete-video TCC reference bank. In calibrated mode this "
        "replaces the prototype end term for phases contained in the bank."
    ),
)
parser.add_argument("--soft_end_temperature", type=float, default=1.0)
parser.add_argument(
    "--soft_end_region_mode",
    choices=("fraction", "last_frames", "last_seconds"),
    default="fraction",
)
parser.add_argument("--soft_end_threshold", type=float, default=0.9)
parser.add_argument("--soft_end_last_frames", type=int, default=3)
parser.add_argument("--soft_end_last_seconds", type=float, default=1.0)
parser.add_argument("--soft_end_reference_capture_hz", type=float, default=2.0)
parser.add_argument(
    "--transition_threshold",
    nargs=2,
    action="append",
    default=[],
    metavar=("PHASE", "LOG_SCORE"),
    help=(
        "Per-current-phase threshold theta_k on the best transition log-score. "
        "Required for every non-terminal phase in calibrated mode."
    ),
)
parser.add_argument(
    "--transition_margin",
    type=float,
    default=0.5,
    help="Required best-versus-runner-up log-score margin m (default: 0.5).",
)
parser.add_argument(
    "--start_probability_threshold",
    type=float,
    default=0.5,
    help="Standalone P(start) floor in calibrated mode (default: 0.5).",
)
parser.add_argument(
    "--start_probability_threshold_phase",
    nargs=2,
    action="append",
    default=[],
    metavar=("PHASE", "PROBABILITY"),
    help="Override the standalone start-probability floor for one current phase.",
)
parser.add_argument(
    "--transition_consecutive_steps",
    type=int,
    default=3,
    help="Consecutive supported frames required to switch policy (default: 3).",
)
parser.add_argument(
    "--transition_consecutive_steps_phase",
    nargs=2,
    action="append",
    default=[],
    metavar=("PHASE", "STEPS"),
    help="Override transition confirmation length for one current phase.",
)
parser.add_argument(
    "--transition_prior",
    nargs=3,
    action="append",
    default=[],
    metavar=("FROM_PHASE", "TO_PHASE", "WEIGHT"),
    help=(
        "Optional sparse transition-prior weight. Rows are normalized; when a "
        "row is provided, omitted outgoing edges have probability zero. CLI "
        "rows override the prior stored in the calibrated artifact."
    ),
)
parser.add_argument(
    "--disable_transition_prior",
    action="store_true",
    help=(
        "Ignore both transition priors stored in the calibrated artifact and "
        "CLI transition-prior rows. Every phase except the current one is then "
        "scored with a zero log-prior contribution."
    ),
)
parser.add_argument(
    "--sequence",
    nargs="+",
    default=["phase_0", "phase_1", "phase_2"],
    help="Expected phase sequence used to score routing decisions.",
)
parser.add_argument(
    "--phase_success_mode",
    choices=("symbolic", "tcc"),
    default="symbolic",
    help=(
        "Signal that completes the active phase. 'symbolic' uses ring state; "
        "'tcc' uses the active encoder's raw squared distance to its goal."
    ),
)
parser.add_argument(
    "--shadow_post_symbolic_steps",
    type=int,
    default=0,
    help=(
        "In shadow+symbolic mode, keep the completed phase active for this many "
        "additional 10 Hz observations while holding the idle action. This "
        "records an uncensored post-boundary score tail before the baseline switch."
    ),
)
parser.add_argument(
    "--shadow_tail_action",
    choices=("idle", "policy"),
    default="idle",
    help=(
        "Action used during --shadow_post_symbolic_steps. 'idle' preserves the "
        "completed scene; 'policy' keeps executing the current phase policy to "
        "collect a deployment-counterfactual score tail."
    ),
)
parser.add_argument(
    "--shadow_policy_tail_phase",
    action="append",
    default=[],
    metavar="PHASE",
    help=(
        "When --shadow_tail_action=policy, restrict active-policy tails to the "
        "listed phase(s); other phase tails remain idle. If omitted, policy is "
        "used for every non-terminal phase."
    ),
)
parser.add_argument(
    "--shadow_stop_after_phase",
    default=None,
    metavar="PHASE",
    help=(
        "Stop each episode after recording the configured shadow tail for this "
        "phase, before parking or switching policy. Intended for targeted "
        "counterfactual calibration runs."
    ),
)
parser.add_argument(
    "--tcc_success_threshold",
    nargs=2,
    action="append",
    default=[],
    metavar=("PHASE", "SQUARED_DISTANCE"),
    help=(
        "Per-phase raw squared TCC goal-distance threshold. Repeat once for "
        "every phase when --phase_success_mode=tcc."
    ),
)
parser.add_argument(
    "--tcc_success_consecutive_steps",
    type=int,
    default=5,
    help="Number of consecutive below-threshold observations required for TCC success.",
)
parser.add_argument(
    "--tcc_success_min_steps",
    type=int,
    default=20,
    help="Minimum number of policy steps before TCC may complete a phase.",
)
parser.add_argument(
    "--disable_tcc_quiescence_guard",
    action="store_true",
    help=(
        "Allow TCC success while an arm is busy or carrying a ring. By "
        "default, below-threshold confirmation is accepted only with both "
        "arms idle and no attached ring; policy actions are held at idle "
        "while that confirmation streak is being collected."
    ),
)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_steps_per_phase", type=int, default=700)
parser.add_argument("--episode_length_s", type=float, default=240.0)
parser.add_argument("--output_dir", default="artifacts/evaluations/full_task_phase012")
parser.add_argument("--video_frame_stride", type=int, default=5)
parser.add_argument("--video_fps", type=int, default=15)
parser.add_argument("--disable_video", action="store_true")
parser.add_argument(
    "--capture_handoff",
    nargs=3,
    default=None,
    metavar=("FROM_PHASE", "TO_PHASE", "OUTPUT_DIR"),
    help=(
        "Capture a 10 Hz pre-parking RGB window around the first symbolic "
        "FROM_PHASE completion in each episode. The decoder is not used as "
        "the boundary label."
    ),
)
parser.add_argument(
    "--capture_handoff_pre_steps",
    type=int,
    default=40,
    help="Policy steps retained before and including the symbolic boundary.",
)
parser.add_argument(
    "--capture_handoff_post_steps",
    type=int,
    default=4,
    help="Additional pre-parking policy steps retained after the boundary.",
)
parser.add_argument(
    "--stop_after_handoff_capture",
    action="store_true",
    help=(
        "End the current episode immediately after its handoff window has "
        "been saved. This avoids unrelated rollout steps in calibration-only runs."
    ),
)
parser.add_argument("--transition_park_max_steps", type=int, default=150)
parser.add_argument("--transition_settle_steps", type=int, default=3)
parser.add_argument(
    "--disable_transition_parking",
    action="store_true",
    help="Route directly from the success frame without parking the arms first.",
)
parser.add_argument(
    "--home_before_phase",
    action="append",
    default=[],
    metavar="PHASE",
    help=(
        "After routing to PHASE, physically return both arms to the canonical "
        "reset end-effector pose before starting its policy. Repeat for more "
        "than one phase. Rings and task state are preserved."
    ),
)
parser.add_argument("--randomize_rings", action="store_true")
parser.add_argument("--ring_randomization_scale", type=float, default=1.0)
parser.add_argument(
    "--stochastic_actions",
    action="store_true",
    help="Sample policy actions instead of deterministic argmax actions.",
)
parser.add_argument(
    "--stochastic_phase",
    action="append",
    default=[],
    metavar="PHASE",
    help=(
        "Sample actions only for PHASE while keeping the other phase policies "
        "deterministic. Repeat for multiple phases."
    ),
)
parser.add_argument(
    "--action_mask_mode",
    choices=("guided", "goal_agnostic", "minimal"),
    default="guided",
    help="Select the historical goal-aware mask or the minimal physical mask.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv

from m_dVrk.controllers.spartan_state_machine import SPARTANStateMachine
from m_dVrk.controllers.ring_sync import sync_attached_and_frozen_rings
from m_dVrk.hrl.checkpoints import find_vecnormalize
from m_dVrk.hrl.calibrated_phase_router import (
    CalibratedPhaseRouter,
    CalibratedRouterArtifact,
    CalibratedRouterConfig,
    SoftAlignmentEndConfig,
    SoftAlignmentReferenceBank,
)
from m_dVrk.hrl.constants import IDLE_ACTION
from m_dVrk.hrl.masked_policy import masked_policy_custom_objects
from m_dVrk.hrl.phase_router import TCCSuccessGate, select_phase_by_nearest_start
from m_dVrk.hrl.rewards import tcc_distance
from m_dVrk.hrl.tcc import compute_average_goal_embedding, load_tcc_model
from m_dVrk.hrl.wrapper import DVRKVisionHRLWrapper
from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    policy_path: Path
    tcc_path: Path
    goal_dataset: Path


@dataclass(frozen=True)
class HandoffFrameRecord:
    frame: torch.Tensor
    phase_step: int
    total_step: int
    symbolic_complete: bool


def save_handoff_window(
    output_root: Path,
    episode: int,
    source_phase: str,
    target_phase: str,
    records: list[HandoffFrameRecord],
    boundary_index: int,
    step_seconds: float,
) -> Path:
    """Persist one complete, full-rate window without overwriting prior data."""

    episode_dir = output_root / f"episode_{episode:03d}"
    if episode_dir.exists():
        raise FileExistsError(f"Refusing to overwrite handoff capture: {episode_dir}")
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True)
    frame_rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        filename = f"{index:06d}.png"
        io.write_png(record.frame.permute(2, 0, 1), str(frames_dir / filename))
        relative_step = index - boundary_index
        frame_rows.append(
            {
                "index": index,
                "file": f"frames/{filename}",
                "relative_step": relative_step,
                "relative_time_s": relative_step * step_seconds,
                "phase_step": record.phase_step,
                "total_step": record.total_step,
                "symbolic_complete": record.symbolic_complete,
                "boundary": index == boundary_index,
            }
        )
    payload = {
        "format": "preparking_handoff_v1",
        "episode": episode,
        "source_phase": source_phase,
        "target_phase": target_phase,
        "capture_hz": 1.0 / step_seconds,
        "step_seconds": step_seconds,
        "boundary_index": boundary_index,
        "frames": frame_rows,
    }
    with (episode_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    return episode_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_phase_specs(raw_specs: list[list[str]]) -> dict[str, PhaseSpec]:
    specs: dict[str, PhaseSpec] = {}
    for name, policy, tcc_checkpoint, goal_dataset in raw_specs:
        if name in specs:
            raise ValueError(f"Duplicate phase specification: {name}")
        spec = PhaseSpec(
            name=name,
            policy_path=Path(policy).resolve(),
            tcc_path=Path(tcc_checkpoint).resolve(),
            goal_dataset=Path(goal_dataset).resolve(),
        )
        if not spec.policy_path.is_file():
            raise FileNotFoundError(f"Missing PPO checkpoint for {name}: {spec.policy_path}")
        if not spec.tcc_path.is_file():
            raise FileNotFoundError(f"Missing TCC checkpoint for {name}: {spec.tcc_path}")
        if not spec.goal_dataset.is_dir():
            raise FileNotFoundError(f"Missing goal dataset for {name}: {spec.goal_dataset}")
        specs[name] = spec
    return specs


def parse_tcc_success_thresholds(
    raw_thresholds: list[list[str]],
    sequence: list[str],
    success_mode: str,
) -> dict[str, float]:
    """Parse and validate repeated ``PHASE DISTANCE`` threshold arguments."""
    thresholds: dict[str, float] = {}
    for name, raw_value in raw_thresholds:
        if name in thresholds:
            raise ValueError(f"Duplicate TCC success threshold for {name}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid TCC success threshold for {name}: {raw_value!r}"
            ) from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"TCC success threshold for {name} must be finite and non-negative"
            )
        thresholds[name] = value

    unknown = sorted(set(thresholds) - set(sequence))
    if unknown:
        raise ValueError(f"TCC thresholds provided for unknown phases: {unknown}")
    if success_mode == "tcc":
        missing = [name for name in sequence if name not in thresholds]
        if missing:
            raise ValueError(
                "Missing --tcc_success_threshold for: " + ", ".join(missing)
            )
    return thresholds


def parse_transition_thresholds(
    raw_thresholds: list[list[str]], sequence: list[str], router_mode: str
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for name, raw_value in raw_thresholds:
        if name in thresholds:
            raise ValueError(f"Duplicate transition threshold for {name}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid transition threshold for {name}: {raw_value!r}"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"Transition threshold for {name} must be finite")
        thresholds[name] = value
    unknown = sorted(set(thresholds) - set(sequence))
    if unknown:
        raise ValueError(f"Transition thresholds provided for unknown phases: {unknown}")
    if router_mode in {"calibrated", "shadow"}:
        missing = [name for name in sequence[:-1] if name not in thresholds]
        if missing:
            raise ValueError(
                "Missing --transition_threshold for: " + ", ".join(missing)
            )
    return thresholds


def parse_start_probability_overrides(
    raw_values: list[list[str]], sequence: list[str]
) -> dict[str, float]:
    values: dict[str, float] = {}
    for phase, raw_value in raw_values:
        if phase in values:
            raise ValueError(f"Duplicate start-probability override for {phase}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid start-probability override for {phase}: {raw_value!r}"
            ) from exc
        if not 0.0 < value <= 1.0:
            raise ValueError("Start-probability overrides must lie in (0, 1]")
        values[phase] = value
    unknown = sorted(set(values) - set(sequence))
    if unknown:
        raise ValueError(f"Start-probability overrides use unknown phases: {unknown}")
    return values


def parse_consecutive_step_overrides(
    raw_values: list[list[str]], sequence: list[str]
) -> dict[str, int]:
    values: dict[str, int] = {}
    for phase, raw_value in raw_values:
        if phase in values:
            raise ValueError(f"Duplicate consecutive-step override for {phase}")
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid consecutive-step override for {phase}: {raw_value!r}"
            ) from exc
        if value <= 0:
            raise ValueError("Consecutive-step overrides must be positive")
        values[phase] = value
    unknown = sorted(set(values) - set(sequence))
    if unknown:
        raise ValueError(f"Consecutive-step overrides use unknown phases: {unknown}")
    return values


def parse_transition_priors(
    raw_priors: list[list[str]], sequence: list[str]
) -> dict[tuple[str, str], float]:
    priors: dict[tuple[str, str], float] = {}
    known = set(sequence)
    for source, target, raw_value in raw_priors:
        edge = (source, target)
        if edge in priors:
            raise ValueError(f"Duplicate transition prior for {source}->{target}")
        if source not in known or target not in known or source == target:
            raise ValueError(f"Invalid transition-prior edge: {source}->{target}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid transition prior for {source}->{target}: {raw_value!r}"
            ) from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Transition priors must be finite and positive")
        priors[edge] = value
    return priors


def load_calibrated_router_artifact(
    path: Path,
    phase_names: list[str],
    specs: dict[str, PhaseSpec],
) -> CalibratedRouterArtifact:
    if not path.is_file():
        raise FileNotFoundError(f"Missing calibrated-router artifact: {path}")
    payload = torch.load(path, map_location="cpu")
    artifact = CalibratedRouterArtifact.from_payload(payload)
    if set(artifact.phase_names) != set(phase_names):
        raise ValueError(
            "Calibrated artifact phases do not match --sequence: "
            f"{artifact.phase_names} != {tuple(phase_names)}"
        )
    checkpoints = artifact.metadata.get("checkpoints", {})
    if isinstance(checkpoints, dict):
        for name in phase_names:
            stored = checkpoints.get(name)
            if stored is not None and Path(str(stored)).resolve() != specs[name].tcc_path:
                raise ValueError(
                    f"TCC/calibrated-artifact mismatch for {name}: "
                    f"{specs[name].tcc_path} != {stored}"
                )
    return artifact


def load_soft_end_reference_bank(path: Path) -> SoftAlignmentReferenceBank:
    if not path.is_file():
        raise FileNotFoundError(f"Missing soft-end reference bank: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Soft-end reference-bank payload must be a mapping")
    return SoftAlignmentReferenceBank.from_payload(payload)


def load_start_embeddings(
    path: Path,
    phase_names: list[str],
    specs: dict[str, PhaseSpec],
) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing start-embedding artifact: {path}")
    payload = torch.load(path, map_location="cpu")
    if int(payload.get("prototype_frames", -1)) != 1:
        raise ValueError("End-to-end routing requires first-frame prototypes")
    starts = payload.get("start_embeddings", {})
    missing = [name for name in phase_names if name not in starts]
    if missing:
        raise ValueError(f"Missing start embeddings for: {', '.join(missing)}")

    prototype_stats = payload.get("stats", {})
    for name in phase_names:
        stored_checkpoint = prototype_stats.get(name, {}).get("checkpoint")
        if stored_checkpoint is None:
            continue
        if Path(stored_checkpoint).resolve() != specs[name].tcc_path:
            raise ValueError(
                f"TCC/start-prototype mismatch for {name}: "
                f"{specs[name].tcc_path} != {stored_checkpoint}"
            )
    return {name: torch.as_tensor(starts[name]).detach().cpu() for name in phase_names}


def camera_frame_u8(wrapper: DVRKVisionHRLWrapper) -> torch.Tensor:
    frame = wrapper.env.scene.sensors["camera"].data.output["rgb"][0, :, :, :3]
    frame = frame.detach().cpu()
    if frame.dtype != torch.uint8:
        if float(frame.max()) <= 1.5:
            frame = frame * 255.0
        frame = frame.round().clamp(0, 255).to(torch.uint8)
    return frame


def encode_router_frame(
    frame: torch.Tensor,
    models: dict[str, torch.nn.Module],
    preprocess: T.Compose,
) -> dict[str, torch.Tensor]:
    image = frame[:, :, :3].permute(2, 0, 1).unsqueeze(0).float() / 255.0
    encoded: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, model in models.items():
            device = next(model.parameters()).device
            encoded[name] = model(preprocess(image.to(device))).detach().cpu()
    return encoded


def annotate_frame(
    frame: torch.Tensor,
    phase: str,
    phase_step: int,
    total_step: int,
    route_note: str,
) -> torch.Tensor:
    image = frame.numpy().copy()
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 78), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, image, 0.50, 0, image)
    cv2.putText(
        image,
        f"PHASE: {phase.upper()}   PHASE STEP: {phase_step:03d}   TOTAL: {total_step:04d}",
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if route_note:
        cv2.putText(
            image,
            route_note,
            (12, 59),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 230, 255),
            2,
            cv2.LINE_AA,
        )
    return torch.from_numpy(image)


def park_arms_for_routing(
    wrapper: DVRKVisionHRLWrapper,
    max_steps: int,
    settle_steps: int,
) -> tuple[int, bool]:
    """Move both arms to a deterministic neutral pose without moving rings."""
    sm = wrapper.sm
    sm.reset_control_state(0)
    sm.set_new_triplet("park", "right_arm", "None", 0)
    sm.set_new_triplet("park", "left_arm", "None", 0)
    active_mask = torch.ones(
        wrapper.num_envs, dtype=torch.bool, device=wrapper.env.device
    )

    steps = 0
    for _ in range(max_steps):
        _, _, terminated, truncated, _ = wrapper.env.step(sm.get_action())
        steps += 1
        done = terminated.to(dtype=torch.bool) | truncated.to(dtype=torch.bool)
        sync_attached_and_frozen_rings(wrapper.env, sm, active_mask & ~done)
        if bool(done[0].item()):
            return steps, False
        if sm.all_idle(0):
            break
    else:
        return steps, False

    for _ in range(settle_steps):
        _, _, terminated, truncated, _ = wrapper.env.step(sm.get_action())
        steps += 1
        done = terminated.to(dtype=torch.bool) | truncated.to(dtype=torch.bool)
        sync_attached_and_frozen_rings(wrapper.env, sm, active_mask & ~done)
        if bool(done[0].item()):
            return steps, False
    return steps, True


def home_arms_for_phase_entry(
    wrapper: DVRKVisionHRLWrapper,
    max_steps: int,
    settle_steps: int,
) -> tuple[int, bool]:
    """Physically return both arms to the policy's canonical reset pose."""
    sm = wrapper.sm
    sm.reset_control_state(0)
    sm.set_new_triplet("home", "right_arm", "None", 0)
    sm.set_new_triplet("home", "left_arm", "None", 0)
    active_mask = torch.ones(
        wrapper.num_envs, dtype=torch.bool, device=wrapper.env.device
    )

    steps = 0
    for _ in range(max_steps):
        _, _, terminated, truncated, _ = wrapper.env.step(sm.get_action())
        steps += 1
        done = terminated.to(dtype=torch.bool) | truncated.to(dtype=torch.bool)
        sync_attached_and_frozen_rings(wrapper.env, sm, active_mask & ~done)
        if bool(done[0].item()):
            return steps, False
        if sm.all_idle(0):
            break
    else:
        return steps, False

    for _ in range(settle_steps):
        _, _, terminated, truncated, _ = wrapper.env.step(sm.get_action())
        steps += 1
        done = terminated.to(dtype=torch.bool) | truncated.to(dtype=torch.bool)
        sync_attached_and_frozen_rings(wrapper.env, sm, active_mask & ~done)
        if bool(done[0].item()):
            return steps, False
    return steps, True


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if args_cli.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args_cli.max_steps_per_phase <= 0:
        raise ValueError("--max_steps_per_phase must be positive")
    if args_cli.video_frame_stride <= 0 or args_cli.video_fps <= 0:
        raise ValueError("Video stride and FPS must be positive")
    if args_cli.transition_park_max_steps <= 0:
        raise ValueError("--transition_park_max_steps must be positive")
    if args_cli.transition_settle_steps < 0:
        raise ValueError("--transition_settle_steps cannot be negative")
    if args_cli.tcc_success_consecutive_steps <= 0:
        raise ValueError("--tcc_success_consecutive_steps must be positive")
    if args_cli.tcc_success_min_steps < 0:
        raise ValueError("--tcc_success_min_steps cannot be negative")
    if args_cli.transition_consecutive_steps <= 0:
        raise ValueError("--transition_consecutive_steps must be positive")
    if args_cli.shadow_post_symbolic_steps < 0:
        raise ValueError("--shadow_post_symbolic_steps cannot be negative")
    if args_cli.shadow_post_symbolic_steps > 0 and not (
        args_cli.router_mode == "shadow"
        and args_cli.phase_success_mode == "symbolic"
    ):
        raise ValueError(
            "--shadow_post_symbolic_steps requires "
            "--router_mode=shadow and --phase_success_mode=symbolic"
        )
    if args_cli.shadow_tail_action != "idle" and not (
        args_cli.router_mode == "shadow"
        and args_cli.phase_success_mode == "symbolic"
        and args_cli.shadow_post_symbolic_steps > 0
    ):
        raise ValueError(
            "--shadow_tail_action=policy requires --router_mode=shadow, "
            "--phase_success_mode=symbolic and --shadow_post_symbolic_steps > 0"
        )
    if not np.isfinite(args_cli.transition_margin) or args_cli.transition_margin < 0:
        raise ValueError("--transition_margin must be finite and non-negative")
    if not 0 < args_cli.start_probability_threshold <= 1:
        raise ValueError("--start_probability_threshold must be in (0, 1]")
    if not 0.0 <= args_cli.ring_randomization_scale <= 1.0:
        raise ValueError("--ring_randomization_scale must be in [0, 1]")
    if args_cli.capture_handoff_pre_steps < 0 or args_cli.capture_handoff_post_steps < 0:
        raise ValueError("Handoff capture step counts cannot be negative")

    specs = parse_phase_specs(args_cli.phase)
    sequence = list(args_cli.sequence)
    stochastic_phases = set(args_cli.stochastic_phase)
    unknown_stochastic_phases = sorted(stochastic_phases - set(sequence))
    if unknown_stochastic_phases:
        raise ValueError(
            "--stochastic_phase contains phases outside --sequence: "
            + ", ".join(unknown_stochastic_phases)
        )
    if len(sequence) < 2 or len(set(sequence)) != len(sequence):
        raise ValueError("--sequence must contain at least two unique phases")
    missing_specs = [name for name in sequence if name not in specs]
    if missing_specs:
        raise ValueError(f"Missing --phase specifications for: {', '.join(missing_specs)}")
    if args_cli.shadow_stop_after_phase is not None:
        if args_cli.shadow_stop_after_phase not in sequence[:-1]:
            raise ValueError(
                "--shadow_stop_after_phase must name a non-terminal phase in --sequence"
            )
        if not (
            args_cli.router_mode == "shadow"
            and args_cli.phase_success_mode == "symbolic"
            and args_cli.shadow_post_symbolic_steps > 0
        ):
            raise ValueError(
                "--shadow_stop_after_phase requires --router_mode=shadow, "
                "--phase_success_mode=symbolic and --shadow_post_symbolic_steps > 0"
            )
    unknown_policy_tail_phases = sorted(
        set(args_cli.shadow_policy_tail_phase) - set(sequence[:-1])
    )
    if unknown_policy_tail_phases:
        raise ValueError(
            "--shadow_policy_tail_phase contains terminal or unknown phases: "
            + ", ".join(unknown_policy_tail_phases)
        )
    if args_cli.shadow_policy_tail_phase and args_cli.shadow_tail_action != "policy":
        raise ValueError(
            "--shadow_policy_tail_phase requires --shadow_tail_action=policy"
        )
    unknown_home_phases = sorted(set(args_cli.home_before_phase) - set(sequence))
    if unknown_home_phases:
        raise ValueError(
            "--home_before_phase contains phases outside --sequence: "
            + ", ".join(unknown_home_phases)
        )
    tcc_success_thresholds = parse_tcc_success_thresholds(
        args_cli.tcc_success_threshold,
        sequence,
        (
            args_cli.phase_success_mode
            if args_cli.router_mode in {"raw", "shadow"}
            else "symbolic"
        ),
    )
    if (
        args_cli.router_mode == "calibrated"
        and args_cli.phase_success_mode == "tcc"
        and sequence[-1] not in tcc_success_thresholds
    ):
        raise ValueError(
            "Calibrated routing still needs --tcc_success_threshold for the final phase"
        )
    transition_thresholds = parse_transition_thresholds(
        args_cli.transition_threshold, sequence, args_cli.router_mode
    )
    start_probability_overrides = parse_start_probability_overrides(
        args_cli.start_probability_threshold_phase, sequence
    )
    consecutive_step_overrides = parse_consecutive_step_overrides(
        args_cli.transition_consecutive_steps_phase, sequence
    )
    transition_priors = parse_transition_priors(args_cli.transition_prior, sequence)
    if args_cli.disable_transition_prior and transition_priors:
        raise ValueError(
            "--disable_transition_prior cannot be combined with --transition_prior"
        )

    handoff_source: str | None = None
    handoff_target: str | None = None
    handoff_output_root: Path | None = None
    if args_cli.capture_handoff is not None:
        handoff_source, handoff_target, raw_capture_root = args_cli.capture_handoff
        if handoff_source not in sequence or handoff_target not in sequence:
            raise ValueError("Captured handoff phases must belong to --sequence")
        if sequence.index(handoff_target) != sequence.index(handoff_source) + 1:
            raise ValueError("--capture_handoff must name adjacent sequence phases")
        handoff_output_root = Path(raw_capture_root).resolve()
        handoff_output_root.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start_embeddings: dict[str, torch.Tensor] | None = None
    calibrated_artifact: CalibratedRouterArtifact | None = None
    calibrated_router: CalibratedPhaseRouter | None = None
    if args_cli.router_mode == "raw":
        if args_cli.start_embeddings is None:
            raise ValueError("--start_embeddings is required in raw router mode")
        start_embeddings = load_start_embeddings(
            Path(args_cli.start_embeddings).resolve(), sequence, specs
        )
    else:
        if args_cli.calibrated_router_artifact is None:
            raise ValueError(
                "--calibrated_router_artifact is required in calibrated or shadow mode"
            )
        calibrated_artifact = load_calibrated_router_artifact(
            Path(args_cli.calibrated_router_artifact).resolve(), sequence, specs
        )
        soft_end_bank = (
            load_soft_end_reference_bank(
                Path(args_cli.soft_end_reference_bank).resolve()
            )
            if args_cli.soft_end_reference_bank is not None
            else None
        )
        soft_end_config = (
            SoftAlignmentEndConfig(
                temperature=args_cli.soft_end_temperature,
                end_region_mode=args_cli.soft_end_region_mode,
                end_threshold=args_cli.soft_end_threshold,
                end_last_frames=args_cli.soft_end_last_frames,
                end_last_seconds=args_cli.soft_end_last_seconds,
                reference_capture_hz=args_cli.soft_end_reference_capture_hz,
            )
            if soft_end_bank is not None
            else None
        )
        if soft_end_config is not None:
            soft_end_config.validate()
        calibrated_router = CalibratedPhaseRouter(
            calibrated_artifact,
            CalibratedRouterConfig(
                transition_thresholds=transition_thresholds,
                margin=args_cli.transition_margin,
                start_log_threshold=float(
                    np.log(args_cli.start_probability_threshold)
                ),
                consecutive_steps=args_cli.transition_consecutive_steps,
                start_log_thresholds={
                    phase: float(np.log(probability))
                    for phase, probability in start_probability_overrides.items()
                },
                consecutive_steps_by_phase=consecutive_step_overrides,
            ),
            transition_priors=(
                {} if args_cli.disable_transition_prior else (transition_priors or None)
            ),
            end_reference_bank=soft_end_bank,
            soft_end_config=soft_end_config,
        )

    set_seed(args_cli.seed)
    env_cfg = MDvrkEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = float(args_cli.episode_length_s)
    env_cfg.num_rerenders_on_reset = 2
    env_cfg.wait_for_textures = False
    env_cfg.events.reset_rings.params["randomize"] = bool(args_cli.randomize_rings)
    env_cfg.events.reset_rings.params["randomization_scale"] = float(
        args_cli.ring_randomization_scale
    )

    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    state_machine = SPARTANStateMachine(isaac_env)
    preprocess = T.Compose([T.Resize((112, 112), antialias=True)])

    print("[full-task] Loading phase-specific TCC encoders and goals...", flush=True)
    tcc_models: dict[str, torch.nn.Module] = {}
    goal_embeddings: dict[str, torch.Tensor] = {}
    for name in sequence:
        spec = specs[name]
        model = load_tcc_model(str(spec.tcc_path), device=isaac_env.device)
        goal = compute_average_goal_embedding(
            model, preprocess, str(spec.goal_dataset), isaac_env.device
        )
        tcc_models[name] = model
        goal_embeddings[name] = goal

    start_phase = sequence[0]
    wrapper = DVRKVisionHRLWrapper(
        isaac_env,
        state_machine,
        tcc_models[start_phase],
        task_phase=start_phase,
        reward_debug_dump_interval=0,
        success_snapshot_prob=0.0,
        success_video_prob=0.0,
        auto_reset_on_success=False,
        action_mask_mode=args_cli.action_mask_mode,
    )
    wrapper.goal_embedding = goal_embeddings[start_phase]

    print("[full-task] Loading deterministic policies...", flush=True)
    policies: dict[str, PPO] = {}
    normalizers: dict[str, VecNormalize | None] = {}
    for name in sequence:
        spec = specs[name]
        policies[name] = PPO.load(
            str(spec.policy_path),
            device=args_cli.device,
            custom_objects=masked_policy_custom_objects(
                str(spec.policy_path), name, args_cli.action_mask_mode
            ),
        )
        vec_path = find_vecnormalize(spec.policy_path)
        if vec_path is None:
            normalizers[name] = None
            print(f"[full-task] {name}: raw observations (no VecNormalize file)")
            continue
        normalizer = VecNormalize.load(str(vec_path), wrapper)
        normalizer.training = False
        normalizer.norm_reward = False
        if normalizer.norm_obs:
            raise ValueError(
                f"{name} uses normalized observations, which are incompatible "
                "with the state-aware action mask"
            )
        normalizers[name] = normalizer
        print(f"[full-task] {name}: VecNormalize={vec_path} (norm_obs=False)")

    camera = wrapper.env.scene.sensors["camera"]
    del camera  # Access remains through wrapper; this validates the sensor exists.

    episode_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    completion_rows: list[dict[str, object]] = []
    distance_trace_rows: list[dict[str, object]] = []
    transition_score_rows: list[dict[str, object]] = []
    captured_handoff_dirs: list[str] = []
    phase_entry_observations: list[np.ndarray] = []
    phase_entry_episodes: list[int] = []
    phase_entry_phases: list[str] = []
    phase_entry_from_phases: list[str] = []
    episode_fieldnames = [
        "episode",
        "seed",
        "success",
        "failure_reason",
        "total_steps",
        *[f"steps_{name}" for name in sequence],
        "completed_phases",
    ]
    transition_fieldnames = [
        "episode",
        "router_mode",
        "from_phase",
        "expected_phase",
        "selected_phase",
        "correct",
        "parking_steps",
        "homing_steps",
        "trigger_goal_distance",
        "trigger_threshold",
        "trigger_streak",
        "symbolic_complete",
        "margin",
        "end_log_score",
        "start_log_probability",
        "transition_log_score",
        "passes_total_threshold",
        "passes_margin",
        "passes_start_floor",
        *[f"distance_{name}" for name in sequence],
        "frame",
    ]
    completion_fieldnames = [
        "episode",
        "phase",
        "success_mode",
        "phase_step",
        "total_step",
        "tcc_goal_distance",
        "tcc_threshold",
        "tcc_streak",
        "quiescent",
        "symbolic_complete",
    ]
    distance_trace_fieldnames = [
        "episode",
        "phase",
        "phase_step",
        "total_step",
        "tcc_goal_distance",
        "below_tcc_threshold",
        "tcc_streak",
        "quiescent",
        "symbolic_event",
        "symbolic_complete",
    ]
    transition_score_fieldnames = [
        "episode",
        "phase",
        "phase_step",
        "total_step",
        "selected_phase",
        "runner_up_phase",
        "end_log_score",
        "best_start_log_probability",
        "best_transition_log_score",
        "margin",
        "transition_threshold",
        "passes_total_threshold",
        "passes_margin",
        "passes_start_floor",
        "supports_transition",
        "confirmation_streak",
        "decision",
        "start_log_probabilities",
        "transition_log_scores",
    ]

    for episode in range(args_cli.episodes):
        wrapper.task_phase = start_phase
        wrapper.tcc_model = tcc_models[start_phase]
        wrapper.goal_embedding = goal_embeddings[start_phase]
        obs = wrapper.reset()
        current_phase = start_phase
        completed_phases: list[str] = []
        phase_steps = {name: 0 for name in sequence}
        total_steps = 0
        failure_reason = ""
        overall_success = False
        route_note = ""
        video_frames: list[torch.Tensor] = []
        handoff_pre_buffer: deque[HandoffFrameRecord] = deque(
            maxlen=args_cli.capture_handoff_pre_steps + 1
        )
        handoff_records: list[HandoffFrameRecord] | None = None
        handoff_boundary_index: int | None = None
        handoff_post_remaining = 0
        handoff_saved = False
        handoff_was_complete = False
        shadow_symbolic_boundary_step: int | None = None
        success_gate = (
            TCCSuccessGate(
                threshold=tcc_success_thresholds[current_phase],
                consecutive_steps=args_cli.tcc_success_consecutive_steps,
                min_steps=args_cli.tcc_success_min_steps,
            )
            if (
                args_cli.phase_success_mode == "tcc"
                and (
                    args_cli.router_mode in {"raw", "shadow"}
                    or current_phase == sequence[-1]
                )
            )
            else None
        )
        if calibrated_router is not None:
            calibrated_router.reset()

        print(f"[full-task] Episode {episode}: start {current_phase}", flush=True)
        while simulation_app.is_running():
            policy_obs = obs
            normalizer = normalizers[current_phase]
            if normalizer is not None:
                policy_obs = normalizer.normalize_obs(obs.copy())
            confirming_tcc_success = (
                success_gate is not None
                and (
                    args_cli.router_mode in {"raw", "shadow"}
                    or current_phase == sequence[-1]
                )
                and not args_cli.disable_tcc_quiescence_guard
                and success_gate.streak > 0
            )
            holding_shadow_tail = (
                args_cli.router_mode == "shadow"
                and args_cli.phase_success_mode == "symbolic"
                and current_phase != sequence[-1]
                and shadow_symbolic_boundary_step is not None
            )
            active_policy_shadow_tail = (
                holding_shadow_tail
                and args_cli.shadow_tail_action == "policy"
                and (
                    not args_cli.shadow_policy_tail_phase
                    or current_phase in args_cli.shadow_policy_tail_phase
                )
            )
            if confirming_tcc_success or (
                holding_shadow_tail and not active_policy_shadow_tail
            ):
                # Once a quiescent below-threshold frame has started the
                # debounce streak, keep the scene stationary while collecting
                # the remaining visual evidence.  Otherwise the policy could
                # immediately start another manipulation from an unseen
                # post-goal state and invalidate the confirmation window.
                action = IDLE_ACTION.cpu().numpy()[None, :]
            else:
                sample_current_policy = (
                    args_cli.stochastic_actions
                    or current_phase in stochastic_phases
                )
                action, _ = policies[current_phase].predict(
                    policy_obs, deterministic=not sample_current_policy
                )
            obs, _, dones, infos = wrapper.step(action)
            phase_steps[current_phase] += 1
            total_steps += 1

            if not args_cli.disable_video and total_steps % args_cli.video_frame_stride == 0:
                video_frames.append(
                    annotate_frame(
                        camera_frame_u8(wrapper),
                        current_phase,
                        phase_steps[current_phase],
                        total_steps,
                        route_note,
                    )
                )

            info = infos[0]
            current_goal_distance = float(
                tcc_distance(
                    wrapper.emb_buffer[:, -1, :], wrapper.goal_embedding
                )[0].item()
            )
            symbolic_event = bool(info.get("phase_success", False))
            symbolic_complete = state_machine.successful(
                0, phase=current_phase
            )
            if (
                handoff_output_root is not None
                and current_phase == handoff_source
                and not handoff_saved
            ):
                capture_record = HandoffFrameRecord(
                    frame=camera_frame_u8(wrapper).detach().cpu(),
                    phase_step=phase_steps[current_phase],
                    total_step=total_steps,
                    symbolic_complete=symbolic_complete,
                )
                if handoff_records is None:
                    handoff_pre_buffer.append(capture_record)
                    if symbolic_complete and not handoff_was_complete:
                        handoff_records = list(handoff_pre_buffer)
                        handoff_boundary_index = len(handoff_records) - 1
                        handoff_post_remaining = args_cli.capture_handoff_post_steps
                else:
                    handoff_records.append(capture_record)
                    handoff_post_remaining -= 1
                handoff_was_complete = symbolic_complete
                if (
                    handoff_records is not None
                    and handoff_boundary_index is not None
                    and handoff_post_remaining <= 0
                ):
                    capture_path = save_handoff_window(
                        handoff_output_root,
                        episode,
                        str(handoff_source),
                        str(handoff_target),
                        handoff_records,
                        handoff_boundary_index,
                        float(env_cfg.sim.dt * env_cfg.decimation),
                    )
                    captured_handoff_dirs.append(str(capture_path))
                    handoff_saved = True
                    print(
                        f"[full-task] Captured pre-parking handoff "
                        f"{handoff_source}->{handoff_target}: {capture_path}",
                        flush=True,
                    )
                    if args_cli.stop_after_handoff_capture:
                        failure_reason = "handoff_capture_complete"
                        break
            quiescent = (
                state_machine.all_idle(0)
                and state_machine.attached_target_l[0] is None
                and state_machine.attached_target_r[0] is None
            )
            if (
                args_cli.router_mode == "shadow"
                and args_cli.phase_success_mode == "symbolic"
                and current_phase != sequence[-1]
                and symbolic_complete
                and shadow_symbolic_boundary_step is None
            ):
                shadow_symbolic_boundary_step = total_steps
            configured_threshold = tcc_success_thresholds.get(current_phase)
            calibrated_evidence = None
            calibrated_selected_phase: str | None = None
            calibrated_frame: torch.Tensor | None = None
            calibrated_would_switch = False
            calibrated_trigger_streak = 0
            trigger_end_log_score: float | str = ""
            trigger_start_log_probability: float | str = ""
            trigger_transition_log_score: float | str = ""
            trigger_passes_total: bool | str = ""
            trigger_passes_margin: bool | str = ""
            trigger_passes_start: bool | str = ""
            scoring_calibrated_transition = (
                calibrated_router is not None and current_phase != sequence[-1]
            )
            if scoring_calibrated_transition:
                # The tutor-selected formulation evaluates all encoders and all
                # three report conditions at every frame. There is no E_k gate.
                calibrated_frame = camera_frame_u8(wrapper)
                encoded = encode_router_frame(calibrated_frame, tcc_models, preprocess)
                candidates = tuple(name for name in sequence if name != current_phase)
                calibrated_decision = calibrated_router.update(
                    current_phase, encoded, candidate_phases=candidates
                )
                calibrated_evidence = calibrated_decision.evidence
                scored_candidates = calibrated_evidence.candidate_phases
                best_index = int(calibrated_evidence.selected_indices[0].item())
                calibrated_selected_phase = calibrated_evidence.selected_phases[0]
                calibrated_would_switch = (
                    calibrated_decision.decisions[0] == "switch"
                )
                calibrated_trigger_streak = int(
                    calibrated_decision.confirmation_streaks[0].item()
                )
                trigger_end_log_score = float(
                    calibrated_evidence.end_log_scores[0].item()
                )
                trigger_start_log_probability = float(
                    calibrated_evidence.start_log_probabilities[0, best_index].item()
                )
                trigger_transition_log_score = float(
                    calibrated_evidence.transition_log_scores[0, best_index].item()
                )
                trigger_passes_total = bool(
                    calibrated_evidence.passes_total_threshold[0].item()
                )
                trigger_passes_margin = bool(
                    calibrated_evidence.passes_margin[0].item()
                )
                trigger_passes_start = bool(
                    calibrated_evidence.passes_start_floor[0].item()
                )
                transition_score_rows.append(
                    {
                        "episode": episode,
                        "phase": current_phase,
                        "phase_step": phase_steps[current_phase],
                        "total_step": total_steps,
                        "selected_phase": calibrated_selected_phase,
                        "runner_up_phase": calibrated_evidence.runner_up_phases[0] or "",
                        "end_log_score": trigger_end_log_score,
                        "best_start_log_probability": trigger_start_log_probability,
                        "best_transition_log_score": trigger_transition_log_score,
                        "margin": float(calibrated_evidence.margins[0].item()),
                        "transition_threshold": transition_thresholds[current_phase],
                        "passes_total_threshold": trigger_passes_total,
                        "passes_margin": trigger_passes_margin,
                        "passes_start_floor": trigger_passes_start,
                        "supports_transition": bool(
                            calibrated_evidence.supports_transition[0].item()
                        ),
                        "confirmation_streak": calibrated_trigger_streak,
                        "decision": calibrated_decision.decisions[0],
                        "start_log_probabilities": json.dumps(
                            {
                                name: float(
                                    calibrated_evidence.start_log_probabilities[0, index].item()
                                )
                                for index, name in enumerate(scored_candidates)
                            },
                            sort_keys=True,
                        ),
                        "transition_log_scores": json.dumps(
                            {
                                name: float(
                                    calibrated_evidence.transition_log_scores[0, index].item()
                                )
                                for index, name in enumerate(scored_candidates)
                            },
                            sort_keys=True,
                        ),
                    }
                )
            if (
                args_cli.router_mode == "calibrated"
                and scoring_calibrated_transition
            ):
                phase_complete = calibrated_would_switch
                trigger_streak = calibrated_trigger_streak
                trigger_threshold = transition_thresholds[current_phase]
            elif success_gate is None:
                if (
                    args_cli.router_mode == "shadow"
                    and args_cli.phase_success_mode == "symbolic"
                    and current_phase != sequence[-1]
                ):
                    phase_complete = (
                        shadow_symbolic_boundary_step is not None
                        and total_steps - shadow_symbolic_boundary_step
                        >= args_cli.shadow_post_symbolic_steps
                    )
                else:
                    phase_complete = symbolic_event
                trigger_streak = 0
                trigger_threshold = None
            else:
                phase_complete = success_gate.update(
                    current_goal_distance,
                    phase_steps[current_phase],
                    eligible=(quiescent or args_cli.disable_tcc_quiescence_guard),
                )
                trigger_streak = success_gate.streak
                trigger_threshold = success_gate.threshold

            distance_trace_rows.append(
                {
                    "episode": episode,
                    "phase": current_phase,
                    "phase_step": phase_steps[current_phase],
                    "total_step": total_steps,
                    "tcc_goal_distance": current_goal_distance,
                    "below_tcc_threshold": (
                        current_goal_distance <= configured_threshold
                        if configured_threshold is not None
                        else ""
                    ),
                    "tcc_streak": trigger_streak,
                    "quiescent": quiescent,
                    "symbolic_event": symbolic_event,
                    "symbolic_complete": symbolic_complete,
                }
            )

            if phase_complete:
                completed_phases.append(current_phase)
                completion_mode = (
                    "calibrated_transition"
                    if (
                        args_cli.router_mode == "calibrated"
                        and scoring_calibrated_transition
                    )
                    else (
                        f"shadow_{args_cli.phase_success_mode}"
                        if args_cli.router_mode == "shadow"
                        else args_cli.phase_success_mode
                    )
                )
                completion_rows.append(
                    {
                        "episode": episode,
                        "phase": current_phase,
                        "success_mode": completion_mode,
                        "phase_step": phase_steps[current_phase],
                        "total_step": total_steps,
                        "tcc_goal_distance": current_goal_distance,
                        "tcc_threshold": trigger_threshold,
                        "tcc_streak": trigger_streak,
                        "quiescent": quiescent,
                        "symbolic_complete": symbolic_complete,
                    }
                )
                print(
                    f"[full-task] Episode {episode}: {current_phase} success "
                    f"via {completion_mode} at "
                    f"phase_step={phase_steps[current_phase]}, total_step={total_steps}, "
                    f"tcc_distance={current_goal_distance:.6f}, "
                    f"symbolic_complete={symbolic_complete}",
                    flush=True,
                )
                if current_phase == args_cli.shadow_stop_after_phase:
                    failure_reason = f"shadow_tail_capture_complete:{current_phase}"
                    break
                if current_phase == sequence[-1]:
                    overall_success = True
                    break

                # Calibrated and shadow modes retain the exact scored frame.
                # Raw mode preserves its historical post-parking query.
                transition_frame = calibrated_frame
                parking_steps = 0
                if not args_cli.disable_transition_parking:
                    parking_steps, parked = park_arms_for_routing(
                        wrapper,
                        args_cli.transition_park_max_steps,
                        args_cli.transition_settle_steps,
                    )
                    if not parked:
                        failure_reason = f"transition_parking_failed:{current_phase}"
                        break
                    print(
                        f"[full-task] Parked arms for routing in "
                        f"{parking_steps} low-level steps",
                        flush=True,
                    )

                if transition_frame is None:
                    transition_frame = camera_frame_u8(wrapper)
                transition_path = output_dir / (
                    f"episode_{episode:03d}_{current_phase}_transition.png"
                )
                io.write_png(transition_frame.permute(2, 0, 1), str(transition_path))
                expected_phase = sequence[len(completed_phases)]
                if args_cli.router_mode == "calibrated":
                    if calibrated_selected_phase is None or calibrated_evidence is None:
                        raise RuntimeError("Confirmed transition lacks calibrated evidence")
                    selected_phase = calibrated_selected_phase
                    margin = float(calibrated_evidence.margins[0].item())
                    distances = [""] * len(sequence)
                elif args_cli.router_mode == "shadow":
                    if calibrated_evidence is None:
                        raise RuntimeError("Shadow transition lacks calibrated evidence")
                    # The calibrated decoder is observational only: task-order
                    # ground truth controls the actual policy handoff.
                    selected_phase = expected_phase
                    margin = float(calibrated_evidence.margins[0].item())
                    distances = [""] * len(sequence)
                else:
                    if start_embeddings is None:
                        raise RuntimeError("Raw routing lacks start embeddings")
                    encoded = encode_router_frame(
                        transition_frame, tcc_models, preprocess
                    )
                    selection = select_phase_by_nearest_start(
                        encoded, start_embeddings, phase_order=sequence
                    )
                    selected_phase = selection.selected_phases[0]
                    margin = float(selection.margins[0])
                    distances = selection.squared_distances[0].tolist()
                correct_route = selected_phase == expected_phase
                transition_row: dict[str, object] = {
                    "episode": episode,
                    "router_mode": args_cli.router_mode,
                    "from_phase": current_phase,
                    "expected_phase": expected_phase,
                    "selected_phase": selected_phase,
                    "correct": correct_route,
                    "parking_steps": parking_steps,
                    "homing_steps": 0,
                    "trigger_goal_distance": current_goal_distance,
                    "trigger_threshold": trigger_threshold,
                    "trigger_streak": trigger_streak,
                    "symbolic_complete": symbolic_complete,
                    "margin": margin,
                    "end_log_score": trigger_end_log_score,
                    "start_log_probability": trigger_start_log_probability,
                    "transition_log_score": trigger_transition_log_score,
                    "passes_total_threshold": trigger_passes_total,
                    "passes_margin": trigger_passes_margin,
                    "passes_start_floor": trigger_passes_start,
                    "frame": str(transition_path),
                }
                transition_row.update(
                    {
                        f"distance_{name}": (
                            float(distance) if distance != "" else ""
                        )
                        for name, distance in zip(sequence, distances, strict=True)
                    }
                )
                transition_rows.append(transition_row)
                route_note = (
                    f"ROUTER: {current_phase} -> {selected_phase} "
                    f"(expected {expected_phase}, margin {margin:.2f})"
                )
                print(f"[full-task] {route_note}", flush=True)
                if not correct_route:
                    failure_reason = (
                        f"routing_error:{current_phase}->{selected_phase};"
                        f"expected={expected_phase}"
                    )
                    break

                if selected_phase in args_cli.home_before_phase:
                    homing_steps, homed = home_arms_for_phase_entry(
                        wrapper,
                        args_cli.transition_park_max_steps,
                        args_cli.transition_settle_steps,
                    )
                    transition_row["homing_steps"] = homing_steps
                    if not homed:
                        failure_reason = f"phase_entry_homing_failed:{selected_phase}"
                        break
                    print(
                        f"[full-task] Homed arms before {selected_phase} in "
                        f"{homing_steps} low-level steps",
                        flush=True,
                    )

                previous_phase = current_phase
                current_phase = selected_phase
                obs = wrapper.transition_to_phase(
                    current_phase,
                    tcc_models[current_phase],
                    goal_embeddings[current_phase],
                )
                phase_entry_observations.append(
                    np.asarray(obs[0], dtype=np.float32).copy()
                )
                phase_entry_episodes.append(episode)
                phase_entry_phases.append(current_phase)
                phase_entry_from_phases.append(previous_phase)
                success_gate = (
                    TCCSuccessGate(
                        threshold=tcc_success_thresholds[current_phase],
                        consecutive_steps=args_cli.tcc_success_consecutive_steps,
                        min_steps=args_cli.tcc_success_min_steps,
                    )
                    if (
                        args_cli.phase_success_mode == "tcc"
                        and (
                            args_cli.router_mode in {"raw", "shadow"}
                            or current_phase == sequence[-1]
                        )
                    )
                    else None
                )
                if calibrated_router is not None:
                    calibrated_router.reset()
                shadow_symbolic_boundary_step = None

            if bool(dones[0]):
                failure_reason = "environment_timeout_or_termination"
                break
            if phase_steps[current_phase] >= args_cli.max_steps_per_phase:
                failure_reason = f"phase_timeout:{current_phase}"
                break

        if not failure_reason and not overall_success:
            failure_reason = "simulation_stopped"

        if not args_cli.disable_video and video_frames:
            outcome = "success" if overall_success else "fail"
            video_path = output_dir / f"episode_{episode:03d}_{outcome}.mp4"
            io.write_video(
                str(video_path),
                torch.stack(video_frames),
                fps=args_cli.video_fps,
            )
            print(f"[full-task] Video saved: {video_path}", flush=True)

        episode_rows.append(
            {
                "episode": episode,
                "seed": args_cli.seed,
                "success": overall_success,
                "failure_reason": failure_reason,
                "total_steps": total_steps,
                **{f"steps_{name}": phase_steps[name] for name in sequence},
                "completed_phases": ">".join(completed_phases),
            }
        )
        print(
            f"[full-task] Episode {episode}: success={overall_success} "
            f"reason={failure_reason or 'complete'} total_steps={total_steps}",
            flush=True,
        )

    write_rows(output_dir / "episodes.csv", episode_rows, episode_fieldnames)
    write_rows(output_dir / "transitions.csv", transition_rows, transition_fieldnames)
    write_rows(
        output_dir / "phase_completions.csv",
        completion_rows,
        completion_fieldnames,
    )
    write_rows(
        output_dir / "tcc_distance_trace.csv",
        distance_trace_rows,
        distance_trace_fieldnames,
    )
    write_rows(
        output_dir / "transition_score_trace.csv",
        transition_score_rows,
        transition_score_fieldnames,
    )
    if phase_entry_observations:
        np.savez_compressed(
            output_dir / "phase_entry_observations.npz",
            obs=np.stack(phase_entry_observations),
            episode=np.asarray(phase_entry_episodes, dtype=np.int64),
            phase=np.asarray(phase_entry_phases, dtype="<U16"),
            from_phase=np.asarray(phase_entry_from_phases, dtype="<U16"),
        )
    success_count = sum(bool(row["success"]) for row in episode_rows)
    summary = {
        "sequence": sequence,
        "episodes": len(episode_rows),
        "successes": success_count,
        "success_rate": success_count / len(episode_rows),
        "deterministic": not args_cli.stochastic_actions and not stochastic_phases,
        "stochastic_phases": sorted(stochastic_phases),
        "action_mask_mode": args_cli.action_mask_mode,
        "seed": args_cli.seed,
        "phase_success": {
            "mode": args_cli.phase_success_mode,
            "thresholds": tcc_success_thresholds,
            "consecutive_steps": args_cli.tcc_success_consecutive_steps,
            "min_steps": args_cli.tcc_success_min_steps,
            "quiescence_guard": not args_cli.disable_tcc_quiescence_guard,
            "completion_events": len(completion_rows),
            "symbolically_complete_at_trigger": sum(
                bool(row["symbolic_complete"]) for row in completion_rows
            ),
        },
        "router": {
            "mode": args_cli.router_mode,
            "start_embeddings": (
                str(Path(args_cli.start_embeddings).resolve())
                if args_cli.start_embeddings is not None
                else None
            ),
            "calibrated_artifact": (
                str(Path(args_cli.calibrated_router_artifact).resolve())
                if args_cli.calibrated_router_artifact is not None
                else None
            ),
            "soft_end_reference_bank": (
                str(Path(args_cli.soft_end_reference_bank).resolve())
                if args_cli.soft_end_reference_bank is not None
                else None
            ),
            "soft_end_config": (
                {
                    "temperature": args_cli.soft_end_temperature,
                    "end_region_mode": args_cli.soft_end_region_mode,
                    "end_threshold": args_cli.soft_end_threshold,
                    "end_last_frames": args_cli.soft_end_last_frames,
                    "end_last_seconds": args_cli.soft_end_last_seconds,
                    "reference_capture_hz": args_cli.soft_end_reference_capture_hz,
                }
                if args_cli.soft_end_reference_bank is not None
                else None
            ),
            "transition_thresholds": transition_thresholds,
            "transition_margin": args_cli.transition_margin,
            "start_probability_threshold": args_cli.start_probability_threshold,
            "start_probability_thresholds_by_phase": start_probability_overrides,
            "transition_consecutive_steps": args_cli.transition_consecutive_steps,
            "transition_consecutive_steps_by_phase": consecutive_step_overrides,
            "transition_prior_disabled": args_cli.disable_transition_prior,
            "transition_priors": {
                f"{source}->{target}": value
                for (source, target), value in (
                    calibrated_router.transition_priors.items()
                    if calibrated_router is not None
                    else transition_priors.items()
                )
            },
            "transition_parking": not args_cli.disable_transition_parking,
            "home_before_phases": list(args_cli.home_before_phase),
            "shadow": {
                "enabled": args_cli.router_mode == "shadow",
                "controls_switch": False,
                "post_symbolic_steps": args_cli.shadow_post_symbolic_steps,
                "tail_action": args_cli.shadow_tail_action,
                "policy_tail_phases": list(args_cli.shadow_policy_tail_phase),
                "stop_after_phase": args_cli.shadow_stop_after_phase,
            },
            "queries": len(transition_rows),
            "correct": sum(bool(row["correct"]) for row in transition_rows),
        },
        "handoff_capture": {
            "enabled": handoff_output_root is not None,
            "source_phase": handoff_source,
            "target_phase": handoff_target,
            "output_root": str(handoff_output_root) if handoff_output_root else None,
            "pre_steps": args_cli.capture_handoff_pre_steps,
            "post_steps": args_cli.capture_handoff_post_steps,
            "stop_after_capture": args_cli.stop_after_handoff_capture,
            "capture_hz": float(1.0 / (env_cfg.sim.dt * env_cfg.decimation)),
            "captured_episodes": len(captured_handoff_dirs),
            "episode_dirs": captured_handoff_dirs,
        },
        "phases": {
            name: {
                "policy": str(specs[name].policy_path),
                "tcc_checkpoint": str(specs[name].tcc_path),
                "goal_dataset": str(specs[name].goal_dataset),
            }
            for name in sequence
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if args_cli.episodes:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        raise
    if args_cli.episodes:
        # Isaac/Kit can block indefinitely during headless shutdown after all
        # CSV/JSON/video artifacts have already been closed.  Bounded batch
        # evaluation should release the GPU immediately for the next test.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    simulation_app.close()
