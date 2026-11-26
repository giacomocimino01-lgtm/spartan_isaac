#!/usr/bin/env python3
"""Evaluate a training-free TCC soft-alignment end score on handoff captures.

The script builds a variable-length reference bank from complete phase videos,
computes P(end), normalized progress, and temporal variance, and replays the
phase handoff with the existing calibrated start detector.  It keeps the
prototype end score as a directly comparable baseline.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
import torchvision.io as io
import torchvision.transforms as T


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "source" / "m_dVrk" / "m_dVrk"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_repo_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tcc_module = load_repo_module("soft_alignment_tcc", PACKAGE_ROOT / "hrl" / "tcc.py")
router_module = load_repo_module(
    "soft_alignment_router", PACKAGE_ROOT / "hrl" / "calibrated_phase_router.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router_artifact", required=True)
    parser.add_argument("--handoff_embeddings", required=True)
    parser.add_argument("--reference_dataset", required=True)
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument(
        "--reference_bank",
        default=None,
        help=(
            "Optional precomputed SoftAlignmentReferenceBank. When provided, "
            "reference images are not encoded again."
        ),
    )
    parser.add_argument("--source_phase", default="phase_1")
    parser.add_argument("--target_phase", default="phase_2")
    parser.add_argument("--reference_limit", type=int, default=None)
    parser.add_argument("--reference_capture_hz", type=float, default=2.0)
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--temperature_grid",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 5.0, 10.0],
    )
    parser.add_argument(
        "--end_threshold_grid",
        type=float,
        nargs="+",
        default=[0.9, 0.95, 0.97, 0.99],
    )
    parser.add_argument(
        "--end_last_frames_grid",
        type=int,
        nargs="*",
        default=[1, 2, 3, 5, 10],
    )
    parser.add_argument(
        "--end_last_seconds_grid",
        type=float,
        nargs="*",
        default=[0.5, 1.0, 1.5, 2.0, 3.0],
    )
    parser.add_argument(
        "--theta_grid",
        type=float,
        nargs="+",
        default=[-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25],
    )
    parser.add_argument(
        "--start_probability_grid",
        type=float,
        nargs="+",
        default=[0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.9],
    )
    parser.add_argument(
        "--consecutive_grid", type=int, nargs="+", default=[3, 4, 5]
    )
    parser.add_argument("--timing_tolerance_steps", type=int, default=5)
    parser.add_argument("--earliest_allowed_delay_steps", type=int, default=0)
    parser.add_argument(
        "--selection_policy",
        choices=("accuracy_first", "safety_first"),
        default="accuracy_first",
        help=(
            "Hyperparameter ranking on the fit split. safety_first minimizes "
            "early switches before maximizing timing hits."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/evaluations/phase_router_soft_alignment_end",
    )
    return parser.parse_args()


def natural_frame_key(path: Path) -> tuple[int, int | str]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name)


def reference_directories(root: Path, limit: int | None) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing reference dataset: {root}")
    directories = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name)
    if limit is not None:
        directories = directories[:limit]
    if not directories:
        raise ValueError(f"No reference videos found under {root}")
    return directories


def frame_paths(video: Path) -> list[Path]:
    paths = sorted(
        (path for path in video.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_frame_key,
    )
    if len(paths) < 2:
        raise ValueError(f"Reference video needs at least two frames: {video}")
    return paths


def encode_reference(
    paths: Sequence[Path],
    model: torch.nn.Module,
    preprocess: T.Compose,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for offset in range(0, len(paths), batch_size):
        batch_paths = paths[offset : offset + batch_size]
        images = torch.stack(
            [io.read_image(str(path))[:3].float() / 255.0 for path in batch_paths]
        )
        with torch.no_grad():
            outputs.append(model(preprocess(images).to(device)).detach().cpu())
    return torch.cat(outputs).float()


def load_artifact(path: Path) -> object:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Router artifact is not a mapping: {path}")
    return router_module.CalibratedRouterArtifact.from_payload(payload)


def load_handoff_trajectories(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = payload.get("trajectories")
    if not isinstance(raw, Mapping):
        raise ValueError(f"Handoff embedding payload lacks trajectories: {path}")
    trajectories = {
        str(name): router_module.HandoffTrajectory(
            source_embeddings=value["source_embeddings"],
            target_embeddings=value["target_embeddings"],
            boundary_index=int(value["boundary_index"]),
        )
        for name, value in raw.items()
    }
    for trajectory in trajectories.values():
        trajectory.validate()
    return trajectories, payload


def start_log_probabilities(
    artifact: object, trajectory: object, target_phase: str
) -> torch.Tensor:
    values = trajectory.target_embeddings.float()
    start = artifact.start_prototypes[target_phase].float()
    delta = torch.sum((values - start) ** 2, dim=1) / float(
        artifact.start_scales[target_phase]
    )
    logits = (
        delta * float(artifact.start_weights[target_phase])
        + float(artifact.start_biases[target_phase])
    )
    return F.logsigmoid(logits)


def prototype_end_log_scores(
    artifact: object, trajectory: object, source_phase: str
) -> torch.Tensor:
    values = trajectory.source_embeddings.float()
    goal = artifact.goal_prototypes[source_phase].float()
    return -torch.sum((values - goal) ** 2, dim=1) / float(
        artifact.end_scales[source_phase]
    )


def first_confirmed_index(support: torch.Tensor, consecutive_steps: int) -> int | None:
    streak = 0
    for index, value in enumerate(support.tolist()):
        streak = streak + 1 if bool(value) else 0
        if streak >= consecutive_steps:
            return index
    return None


def evaluate_scores(
    transition_scores: Mapping[str, torch.Tensor],
    start_scores: Mapping[str, torch.Tensor],
    trajectories: Mapping[str, object],
    trajectory_ids: Sequence[str],
    *,
    theta: float,
    start_probability: float,
    consecutive_steps: int,
    timing_tolerance: int,
    earliest_allowed_delay: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    counts = {"timing_hit": 0, "early_switch": 0, "late_switch": 0, "miss": 0}
    for trajectory_id in trajectory_ids:
        trajectory = trajectories[trajectory_id]
        support = (transition_scores[trajectory_id] > theta) & (
            start_scores[trajectory_id] > math.log(start_probability)
        )
        switch_index = first_confirmed_index(support, consecutive_steps)
        boundary = int(trajectory.boundary_index)
        delay = None if switch_index is None else switch_index - boundary
        if delay is None:
            outcome = "miss"
        elif delay < earliest_allowed_delay:
            outcome = "early_switch"
        elif delay <= timing_tolerance:
            outcome = "timing_hit"
        else:
            outcome = "late_switch"
        counts[outcome] += 1
        rows.append(
            {
                "trajectory": trajectory_id,
                "boundary_index": boundary,
                "switch_index": switch_index,
                "delay": delay,
                "outcome": outcome,
            }
        )
    switched_delays = [abs(int(row["delay"])) for row in rows if row["delay"] is not None]
    count = len(rows)
    return {
        "trajectories": count,
        "timing_hits": counts["timing_hit"],
        "early_switches": counts["early_switch"],
        "late_switches": counts["late_switch"],
        "misses": counts["miss"],
        "timing_hit_rate": counts["timing_hit"] / max(count, 1),
        "mean_absolute_delay": (
            sum(switched_delays) / len(switched_delays) if switched_delays else None
        ),
        "rows": rows,
    }


def selection_key(
    row: Mapping[str, object], selection_policy: str
) -> tuple[float, ...]:
    """Rank hyperparameters exclusively on the fit split."""
    metrics = row["fit"]
    mean_delay = metrics["mean_absolute_delay"]
    quality = (
        (
            -float(metrics["early_switches"]),
            float(metrics["timing_hits"]),
        )
        if selection_policy == "safety_first"
        else (
            float(metrics["timing_hits"]),
            -float(metrics["early_switches"]),
        )
    )
    return quality + (
        -float(metrics["misses"]),
        -float(metrics["late_switches"]),
        -float(mean_delay if mean_delay is not None else 1.0e9),
        float(row["theta"]),
        float(row["start_probability"]),
        float(row["consecutive_steps"]),
    )


def tune_decision_grid(
    transition_scores: Mapping[str, torch.Tensor],
    start_scores: Mapping[str, torch.Tensor],
    trajectories: Mapping[str, object],
    fit_ids: Sequence[str],
    validation_ids: Sequence[str],
    args: argparse.Namespace,
) -> tuple[dict[str, object], int]:
    rows: list[dict[str, object]] = []
    for theta, start_probability, consecutive_steps in itertools.product(
        args.theta_grid, args.start_probability_grid, args.consecutive_grid
    ):
        fit = evaluate_scores(
            transition_scores,
            start_scores,
            trajectories,
            fit_ids,
            theta=theta,
            start_probability=start_probability,
            consecutive_steps=consecutive_steps,
            timing_tolerance=args.timing_tolerance_steps,
            earliest_allowed_delay=args.earliest_allowed_delay_steps,
        )
        rows.append(
            {
                "theta": theta,
                "start_probability": start_probability,
                "consecutive_steps": consecutive_steps,
                "fit": fit,
            }
        )
    rows.sort(key=lambda row: selection_key(row, args.selection_policy), reverse=True)
    selected = rows[0]
    selected["validation"] = evaluate_scores(
        transition_scores,
        start_scores,
        trajectories,
        validation_ids,
        theta=float(selected["theta"]),
        start_probability=float(selected["start_probability"]),
        consecutive_steps=int(selected["consecutive_steps"]),
        timing_tolerance=args.timing_tolerance_steps,
        earliest_allowed_delay=args.earliest_allowed_delay_steps,
    )
    selected["all20_diagnostic"] = evaluate_scores(
        transition_scores,
        start_scores,
        trajectories,
        sorted(trajectories),
        theta=float(selected["theta"]),
        start_probability=float(selected["start_probability"]),
        consecutive_steps=int(selected["consecutive_steps"]),
        timing_tolerance=args.timing_tolerance_steps,
        earliest_allowed_delay=args.earliest_allowed_delay_steps,
    )
    return selected, len(rows)


def compact_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    return {
        key: metrics[key]
        for key in (
            "trajectories",
            "timing_hits",
            "early_switches",
            "late_switches",
            "misses",
            "timing_hit_rate",
            "mean_absolute_delay",
        )
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.embedding_size <= 0:
        raise ValueError("batch_size and embedding_size must be positive")
    if args.reference_limit is not None and args.reference_limit <= 0:
        raise ValueError("reference_limit must be positive")
    if args.reference_capture_hz <= 0:
        raise ValueError("reference_capture_hz must be positive")
    if any(value <= 0 or not math.isfinite(value) for value in args.temperature_grid):
        raise ValueError("temperature_grid values must be finite and positive")
    if any(not 0.0 <= value < 1.0 for value in args.end_threshold_grid):
        raise ValueError("end_threshold_grid values must lie in [0, 1)")
    if any(value <= 0 for value in args.end_last_frames_grid):
        raise ValueError("end_last_frames_grid values must be positive")
    if any(
        value <= 0 or not math.isfinite(value)
        for value in args.end_last_seconds_grid
    ):
        raise ValueError("end_last_seconds_grid values must be finite and positive")
    if any(not 0.0 < value <= 1.0 for value in args.start_probability_grid):
        raise ValueError("start_probability_grid values must lie in (0, 1]")
    if any(value <= 0 for value in args.consecutive_grid):
        raise ValueError("consecutive_grid values must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = Path(args.router_artifact).resolve()
    handoff_path = Path(args.handoff_embeddings).resolve()
    reference_root = Path(args.reference_dataset).resolve()
    checkpoint_path = Path(args.source_checkpoint).resolve()

    artifact = load_artifact(artifact_path)
    trajectories, handoff_payload = load_handoff_trajectories(handoff_path)
    fit_ids = tuple(str(value) for value in handoff_payload["fit_ids"])
    validation_ids = tuple(str(value) for value in handoff_payload["validation_ids"])
    if args.reference_bank is None:
        model = tcc_module.load_tcc_model(
            str(checkpoint_path), device=device, embedding_size=args.embedding_size
        )
        preprocess = T.Compose([T.Resize((112, 112), antialias=True)])
        directories = reference_directories(reference_root, args.reference_limit)
        references: list[torch.Tensor] = []
        reference_rows: list[dict[str, object]] = []
        for index, directory in enumerate(directories, start=1):
            paths = frame_paths(directory)
            print(
                f"[soft-end] Encoding reference {index}/{len(directories)}: "
                f"{directory.name} ({len(paths)} frames)",
                flush=True,
            )
            embedding = encode_reference(
                paths, model, preprocess, device, args.batch_size
            )
            references.append(embedding)
            reference_rows.append(
                {
                    "trajectory": directory.name,
                    "frames": len(paths),
                    "duration_seconds": (len(paths) - 1)
                    / args.reference_capture_hz,
                }
            )
        bank = router_module.SoftAlignmentReferenceBank(
            phase_names=(args.source_phase,),
            references_by_phase={args.source_phase: tuple(references)},
            metadata={
                "reference_dataset": str(reference_root),
                "source_checkpoint": str(checkpoint_path),
                "reference_capture_hz": args.reference_capture_hz,
                "reference_videos": reference_rows,
            },
        )
        bank.validate()
    else:
        source_bank_path = Path(args.reference_bank).resolve()
        payload = torch.load(source_bank_path, map_location="cpu", weights_only=False)
        bank = router_module.SoftAlignmentReferenceBank.from_payload(payload)
        bank.validate()
        if args.source_phase not in bank.references_by_phase:
            raise ValueError(
                f"Reference bank does not contain source phase {args.source_phase!r}"
            )
        references = list(bank.references_by_phase[args.source_phase])
        raw_rows = bank.metadata.get("reference_videos", [])
        reference_rows = list(raw_rows) if isinstance(raw_rows, Sequence) else []
    bank_path = output_dir / "soft_alignment_reference_bank.pt"
    torch.save(bank.to_payload(), bank_path)

    start_scores = {
        name: start_log_probabilities(artifact, trajectory, args.target_phase)
        for name, trajectory in trajectories.items()
    }
    prototype_end = {
        name: prototype_end_log_scores(artifact, trajectory, args.source_phase)
        for name, trajectory in trajectories.items()
    }
    prototype_transition = {
        name: prototype_end[name] + start_scores[name] for name in trajectories
    }
    baseline, evaluated_decisions = tune_decision_grid(
        prototype_transition,
        start_scores,
        trajectories,
        fit_ids,
        validation_ids,
        args,
    )

    region_specs: list[dict[str, object]] = [
        {"end_region_mode": "fraction", "end_threshold": value}
        for value in args.end_threshold_grid
    ]
    region_specs.extend(
        {"end_region_mode": "last_frames", "end_last_frames": value}
        for value in args.end_last_frames_grid
    )
    region_specs.extend(
        {"end_region_mode": "last_seconds", "end_last_seconds": value}
        for value in args.end_last_seconds_grid
    )
    candidates: list[dict[str, object]] = []
    cached_evidence: dict[tuple[float, str, float], dict[str, object]] = {}
    for temperature, region_spec in itertools.product(
        args.temperature_grid, region_specs
    ):
        mode = str(region_spec["end_region_mode"])
        region_value = float(
            region_spec.get(
                "end_threshold",
                region_spec.get("end_last_frames", region_spec.get("end_last_seconds")),
            )
        )
        config = router_module.SoftAlignmentEndConfig(
            temperature=temperature,
            end_region_mode=mode,
            end_threshold=float(region_spec.get("end_threshold", 0.9)),
            end_last_frames=int(region_spec.get("end_last_frames", 3)),
            end_last_seconds=float(region_spec.get("end_last_seconds", 1.0)),
            reference_capture_hz=args.reference_capture_hz,
        )
        evidence_by_trajectory = {
            name: router_module.soft_alignment_end_evidence(
                trajectory.source_embeddings, references, config
            )
            for name, trajectory in trajectories.items()
        }
        cached_evidence[(temperature, mode, region_value)] = evidence_by_trajectory
        transition_scores = {
            name: evidence.log_end_probabilities + start_scores[name]
            for name, evidence in evidence_by_trajectory.items()
        }
        selected, evaluated = tune_decision_grid(
            transition_scores,
            start_scores,
            trajectories,
            fit_ids,
            validation_ids,
            args,
        )
        boundary_values = []
        for name in validation_ids:
            boundary = int(trajectories[name].boundary_index)
            evidence = evidence_by_trajectory[name]
            boundary_values.append(
                {
                    "trajectory": name,
                    "p_end": float(evidence.end_probabilities[boundary]),
                    "progress": float(evidence.progress[boundary]),
                    "total_variance": float(evidence.total_variance[boundary]),
                }
            )
        candidates.append(
            {
                "temperature": temperature,
                **region_spec,
                "region_value": region_value,
                "selected_decision": selected,
                "evaluated_decision_configurations": evaluated,
                "boundary_values": boundary_values,
            }
        )

    candidates.sort(
        key=lambda row: selection_key(
            row["selected_decision"], args.selection_policy
        ),
        reverse=True,
    )
    selected_soft = candidates[0]
    selected_key = (
        float(selected_soft["temperature"]),
        str(selected_soft["end_region_mode"]),
        float(selected_soft["region_value"]),
    )
    selected_evidence = cached_evidence[selected_key]

    csv_path = output_dir / "selected_soft_alignment_frame_scores.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = (
            "trajectory",
            "frame_index",
            "boundary_index",
            "delay",
            "p_end",
            "log_p_end",
            "progress",
            "within_variance",
            "between_variance",
            "total_variance",
            "start_probability",
            "transition_log_score",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name in sorted(trajectories):
            trajectory = trajectories[name]
            boundary = int(trajectory.boundary_index)
            evidence = selected_evidence[name]
            for index in range(trajectory.source_embeddings.shape[0]):
                writer.writerow(
                    {
                        "trajectory": name,
                        "frame_index": index,
                        "boundary_index": boundary,
                        "delay": index - boundary,
                        "p_end": float(evidence.end_probabilities[index]),
                        "log_p_end": float(evidence.log_end_probabilities[index]),
                        "progress": float(evidence.progress[index]),
                        "within_variance": float(
                            evidence.within_reference_variance[index]
                        ),
                        "between_variance": float(
                            evidence.between_reference_variance[index]
                        ),
                        "total_variance": float(evidence.total_variance[index]),
                        "start_probability": float(torch.exp(start_scores[name][index])),
                        "transition_log_score": float(
                            evidence.log_end_probabilities[index]
                            + start_scores[name][index]
                        ),
                    }
                )

    def compact_selection(selected: Mapping[str, object]) -> dict[str, object]:
        return {
            "theta": selected["theta"],
            "start_probability": selected["start_probability"],
            "consecutive_steps": selected["consecutive_steps"],
            "validation": compact_metrics(selected["validation"]),
            "fit": compact_metrics(selected["fit"]),
            "all20_diagnostic": compact_metrics(selected["all20_diagnostic"]),
        }

    summary = {
        "format": "soft_alignment_end_evaluation_v1",
        "status": "experimental_not_promoted",
        "router_artifact": str(artifact_path),
        "handoff_embeddings": str(handoff_path),
        "reference_bank": str(bank_path),
        "reference_dataset": str(reference_root),
        "reference_capture_hz": args.reference_capture_hz,
        "references": reference_rows,
        "fit_ids": list(fit_ids),
        "validation_ids": list(validation_ids),
        "strict_timing_window": {
            "earliest_allowed_delay_steps": args.earliest_allowed_delay_steps,
            "latest_allowed_delay_steps": args.timing_tolerance_steps,
        },
        "selection_policy": args.selection_policy,
        "decision_grid": {
            "theta": args.theta_grid,
            "start_probability": args.start_probability_grid,
            "consecutive_steps": args.consecutive_grid,
            "end_threshold": args.end_threshold_grid,
            "end_last_frames": args.end_last_frames_grid,
            "end_last_seconds": args.end_last_seconds_grid,
            "configurations_per_end_model": evaluated_decisions,
        },
        "prototype_baseline": compact_selection(baseline),
        "selected_soft_alignment": {
            "temperature": selected_soft["temperature"],
            "end_region_mode": selected_soft["end_region_mode"],
            "region_value": selected_soft["region_value"],
            **compact_selection(selected_soft["selected_decision"]),
            "boundary_values": selected_soft["boundary_values"],
        },
        "soft_alignment_candidates": [
            {
                "temperature": row["temperature"],
                "end_region_mode": row["end_region_mode"],
                "region_value": row["region_value"],
                **compact_selection(row["selected_decision"]),
            }
            for row in candidates
        ],
        "selected_frame_scores": str(csv_path),
        "note": (
            "Hyperparameters are selected on the 15 fit trajectories and measured "
            "once on five held-out validation trajectories; all-20 metrics are "
            "diagnostic and include the fit trajectories."
        ),
    }
    summary_path = output_dir / "soft_alignment_end_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "reference_bank": str(bank_path),
                "prototype_baseline": summary["prototype_baseline"],
                "selected_soft_alignment": summary["selected_soft_alignment"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
