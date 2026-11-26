#!/usr/bin/env python3
"""Fit the calibrated TCC transition router from phase-segment datasets.

Each ``--phase`` supplies one TCC encoder plus calibration and validation
directories.  A directory contains one subdirectory per demonstration segment,
and each segment contains ordered RGB frames.  Every frame is encoded by every
phase encoder, as required for foreign-phase negatives.

The resulting ``calibrated_router.pt`` contains the prototypes, leave-one-video-
out scales, and one class-balanced monotone logistic start detector per phase.
Decision thresholds are intentionally not fitted on the calibration data; they
must be selected from validation replay or closed-loop rollouts.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence

import torch
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


tcc_module = load_repo_module("calibrated_router_tcc", PACKAGE_ROOT / "hrl" / "tcc.py")
router_module = load_repo_module(
    "calibrated_router_core", PACKAGE_ROOT / "hrl" / "calibrated_phase_router.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        nargs=4,
        action="append",
        required=True,
        metavar=("NAME", "CHECKPOINT", "CALIB_DATASET", "VALID_DATASET"),
        help=(
            "Repeat for every phase. Dataset directories contain one frame "
            "directory per demonstration segment."
        ),
    )
    parser.add_argument(
        "--endpoint_frames",
        type=int,
        default=5,
        help="First/last L frames used as start/end windows (default: 5).",
    )
    parser.add_argument(
        "--guard_frames",
        type=int,
        default=10,
        help="Frames excluded at the end of declared predecessor segments (default: 10).",
    )
    parser.add_argument(
        "--guard_transition",
        nargs=2,
        action="append",
        default=[],
        metavar=("FROM_PHASE", "TO_PHASE"),
        help=(
            "Exclude the last G frames of FROM_PHASE segments from the negative "
            "examples of TO_PHASE. Repeat for every annotated transition."
        ),
    )
    parser.add_argument(
        "--transition_count",
        nargs=3,
        action="append",
        default=[],
        metavar=("FROM_PHASE", "TO_PHASE", "COUNT"),
        help=(
            "Annotated FROM->TO transition frequency. Repeating an edge is "
            "not allowed. Missing edges receive prior probability zero."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_calib_videos", type=int, default=None)
    parser.add_argument("--max_valid_videos", type=int, default=None)
    parser.add_argument("--min_scale", type=float, default=1.0e-8)
    parser.add_argument("--logistic_l2", type=float, default=1.0e-3)
    parser.add_argument(
        "--theta_grid",
        type=float,
        nargs="+",
        default=[-4.0, -3.0, -2.0, -1.5, -1.0],
        help="Shared transition-log-score thresholds tried on validation replay.",
    )
    parser.add_argument(
        "--margin_grid", type=float, nargs="+", default=[0.5, 1.0, 2.0]
    )
    parser.add_argument(
        "--start_probability_grid",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
    )
    parser.add_argument(
        "--consecutive_grid", type=int, nargs="+", default=[3, 5]
    )
    parser.add_argument(
        "--timing_tolerance",
        type=int,
        default=10,
        help="Boundary tolerance w in validation frames (default: 10).",
    )
    parser.add_argument(
        "--skip_grid_search",
        action="store_true",
        help="Skip sequential validation replay and parameter selection.",
    )
    parser.add_argument(
        "--output_dir", default="artifacts/evaluations/phase_router_calibrated"
    )
    parser.add_argument(
        "--save_embedding_cache",
        action="store_true",
        help="Also save fit/validation precomputed embedding payloads.",
    )
    return parser.parse_args()


def video_directories(dataset: Path, limit: int | None) -> list[Path]:
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")
    videos = sorted(path for path in dataset.iterdir() if path.is_dir())
    return videos if limit is None else videos[:limit]


def frame_paths(video: Path) -> list[Path]:
    return sorted(
        path for path in video.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_image_batch(
    paths: Sequence[Path], preprocess: T.Compose, device: torch.device
) -> torch.Tensor:
    images = [io.read_image(str(path))[:3].float() / 255.0 for path in paths]
    if not images:
        raise ValueError("Cannot encode an empty image batch")
    return preprocess(torch.stack(images)).to(device)


def encode_paths_with_all_models(
    models: Mapping[str, torch.nn.Module],
    paths: Sequence[Path],
    preprocess: T.Compose,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    output: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    for offset in range(0, len(paths), batch_size):
        images = load_image_batch(paths[offset : offset + batch_size], preprocess, device)
        with torch.no_grad():
            for name, model in models.items():
                output[name].append(model(images).detach().cpu())
    if not paths:
        raise ValueError("Cannot encode a segment without frames")
    return {name: torch.cat(parts) for name, parts in output.items()}


def build_router_dataset(
    models: Mapping[str, torch.nn.Module],
    datasets: Mapping[str, Path],
    *,
    endpoint_frames: int,
    guard_frames: int,
    guard_transitions: set[tuple[str, str]],
    video_limit: int | None,
    preprocess: T.Compose,
    device: torch.device,
    batch_size: int,
) -> object:
    embeddings: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    labels: list[str] = []
    trajectory_ids: list[str] = []
    starts: list[bool] = []
    ends: list[bool] = []
    negative_masks: dict[str, list[bool]] = {name: [] for name in models}

    for source_phase, dataset in datasets.items():
        for video in video_directories(dataset, video_limit):
            paths = frame_paths(video)
            if not paths:
                continue
            encoded = encode_paths_with_all_models(
                models, paths, preprocess, device, batch_size
            )
            count = len(paths)
            start_count = min(endpoint_frames, count)
            end_start = max(count - endpoint_frames, 0)
            for name in models:
                embeddings[name].append(encoded[name])
            labels.extend([source_phase] * count)
            # Equal names across phase folders are treated as segments from the
            # same original demonstration, preserving video-level grouping.
            trajectory_ids.extend([video.name] * count)
            starts.extend(index < start_count for index in range(count))
            ends.extend(index >= end_start for index in range(count))
            for target_phase in models:
                exclude_guard = (source_phase, target_phase) in guard_transitions
                negative_masks[target_phase].extend(
                    not (exclude_guard and index >= max(count - guard_frames, 0))
                    for index in range(count)
                )

    if not labels:
        raise ValueError("No frames were found in the supplied datasets")
    data = router_module.CalibratedRouterDataset(
        embeddings_by_phase={
            name: torch.cat(parts, dim=0) for name, parts in embeddings.items()
        },
        phase_labels=tuple(labels),
        trajectory_ids=tuple(trajectory_ids),
        start_mask=torch.tensor(starts, dtype=torch.bool),
        end_mask=torch.tensor(ends, dtype=torch.bool),
        start_negative_masks={
            name: torch.tensor(mask, dtype=torch.bool)
            for name, mask in negative_masks.items()
        },
    )
    data.validate(tuple(models))
    return data


def detector_values(
    artifact: object, data: object, phase: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.tensor(
        [value == phase for value in data.phase_labels], dtype=torch.bool
    )
    positives = labels & data.start_mask
    allowed = data.start_negative_masks[phase]
    negatives = (~labels) & allowed
    selected = positives | negatives
    values = data.embeddings_by_phase[phase]
    delta = torch.sum(
        (values - artifact.start_prototypes[phase]) ** 2, dim=1
    ) / artifact.start_scales[phase]
    probabilities = torch.sigmoid(
        delta * artifact.start_weights[phase] + artifact.start_biases[phase]
    )
    return probabilities[selected], positives[selected], delta[selected]


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    positives = scores[labels]
    negatives = scores[~labels]
    if not positives.numel() or not negatives.numel():
        return float("nan")
    # Pairwise definition handles ties exactly and is inexpensive for endpoint
    # windows.  Chunk negatives to keep peak memory bounded for long datasets.
    favorable = 0.0
    comparisons = 0
    for offset in range(0, negatives.numel(), 4096):
        chunk = negatives[offset : offset + 4096]
        favorable += float((positives[:, None] > chunk[None, :]).sum().item())
        favorable += 0.5 * float((positives[:, None] == chunk[None, :]).sum().item())
        comparisons += positives.numel() * chunk.numel()
    return favorable / comparisons


def validation_start_report(
    artifact: object,
    data: object,
    output_dir: Path,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    confusion = {
        phase: Counter({candidate: 0 for candidate in artifact.phase_names})
        for phase in artifact.phase_names
    }
    start_indices = data.start_mask.nonzero(as_tuple=False).flatten().tolist()
    for index in start_indices:
        true_phase = data.phase_labels[index]
        log_probabilities: dict[str, float] = {}
        for candidate in artifact.phase_names:
            value = data.embeddings_by_phase[candidate][index]
            delta = float(
                torch.sum((value - artifact.start_prototypes[candidate]) ** 2).item()
                / artifact.start_scales[candidate]
            )
            logit = (
                artifact.start_weights[candidate] * delta
                + artifact.start_biases[candidate]
            )
            log_probabilities[candidate] = float(
                torch.nn.functional.logsigmoid(torch.tensor(logit)).item()
            )
        predicted = max(log_probabilities, key=log_probabilities.get)
        confusion[true_phase][predicted] += 1
        rows.append(
            {
                "trajectory": data.trajectory_ids[index],
                "true_phase": true_phase,
                "predicted_phase": predicted,
                "correct": predicted == true_phase,
                **{
                    f"log_p_start_{name}": log_probabilities[name]
                    for name in artifact.phase_names
                },
            }
        )

    fieldnames = [
        "trajectory",
        "true_phase",
        "predicted_phase",
        "correct",
        *[f"log_p_start_{name}" for name in artifact.phase_names],
    ]
    with (output_dir / "validation_start_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "start_frames": len(rows),
        "choice_accuracy": correct / len(rows) if rows else 0.0,
        "confusion": {
            true: dict(counts) for true, counts in confusion.items()
        },
    }


def replay_validation(
    artifact: object,
    data: object,
    *,
    theta: float,
    margin: float,
    start_probability: float,
    consecutive_steps: int,
    timing_tolerance: int,
) -> dict[str, float | int]:
    """Replay the online state machine and score boundaries independently."""

    config = router_module.CalibratedRouterConfig(
        transition_thresholds={name: theta for name in artifact.phase_names},
        margin=margin,
        start_log_threshold=math.log(start_probability),
        consecutive_steps=consecutive_steps,
    )
    decoder = router_module.CalibratedPhaseRouter(artifact, config)
    trajectory_order = tuple(dict.fromkeys(str(value) for value in data.trajectory_ids))
    boundary_count = 0
    timing_hits = 0
    choice_hits = 0
    joint_hits = 0
    misses = 0
    false_switches = 0
    absolute_delays: list[int] = []

    for trajectory in trajectory_order:
        indices = [
            index
            for index, value in enumerate(data.trajectory_ids)
            if str(value) == trajectory
        ]
        if not indices:
            continue
        labels = [data.phase_labels[index] for index in indices]
        boundaries: list[tuple[int, str, str]] = []
        for position in range(1, len(labels)):
            if labels[position] != labels[position - 1]:
                boundaries.append((position, labels[position - 1], labels[position]))
        boundary_count += len(boundaries)

        predicted_events: list[tuple[int, str, str]] = []
        current_phase = labels[0]
        decoder.reset()
        for position, index in enumerate(indices):
            embeddings = {
                name: data.embeddings_by_phase[name][index]
                for name in artifact.phase_names
            }
            candidates = tuple(
                name for name in artifact.phase_names if name != current_phase
            )
            decision = decoder.update(
                current_phase, embeddings, candidate_phases=candidates
            )
            if decision.decisions[0] != "switch":
                continue
            selected = decision.selected_phases[0]
            if selected is None:
                continue
            predicted_events.append((position, current_phase, selected))
            current_phase = selected
            decoder.reset()

        used_events: set[int] = set()
        for boundary_position, true_from, true_to in boundaries:
            candidates = [
                (event_index, event)
                for event_index, event in enumerate(predicted_events)
                if event_index not in used_events
                and event[1] == true_from
                and abs(event[0] - boundary_position) <= timing_tolerance
            ]
            if not candidates:
                misses += 1
                continue
            event_index, event = min(
                candidates, key=lambda item: abs(item[1][0] - boundary_position)
            )
            used_events.add(event_index)
            timing_hits += 1
            delay = event[0] - boundary_position
            absolute_delays.append(abs(delay))
            if event[2] == true_to:
                choice_hits += 1
                joint_hits += 1
        false_switches += len(predicted_events) - len(used_events)

    denominator = max(boundary_count, 1)
    return {
        "boundaries": boundary_count,
        "timing_hits": timing_hits,
        "choice_hits": choice_hits,
        "joint_hits": joint_hits,
        "misses": misses,
        "false_switches": false_switches,
        "timing_hit_rate": timing_hits / denominator,
        "choice_accuracy": choice_hits / max(timing_hits, 1),
        "joint_success_rate": joint_hits / denominator,
        "miss_rate": misses / denominator,
        "mean_absolute_delay": (
            sum(absolute_delays) / len(absolute_delays)
            if absolute_delays
            else float("nan")
        ),
    }


def tune_decision_parameters(
    artifact: object,
    data: object,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for theta, margin, start_probability, consecutive_steps in itertools.product(
        args.theta_grid,
        args.margin_grid,
        args.start_probability_grid,
        args.consecutive_grid,
    ):
        metrics = replay_validation(
            artifact,
            data,
            theta=theta,
            margin=margin,
            start_probability=start_probability,
            consecutive_steps=consecutive_steps,
            timing_tolerance=args.timing_tolerance,
        )
        rows.append(
            {
                "theta": theta,
                "margin": margin,
                "start_probability": start_probability,
                "consecutive_steps": consecutive_steps,
                **metrics,
            }
        )
    if not rows:
        raise ValueError("Decision-parameter grid is empty")
    if not any(int(row["boundaries"]) > 0 for row in rows):
        return {
            "status": "unavailable",
            "reason": (
                "No cross-phase boundaries could be reconstructed. Validation "
                "phase folders must reuse the same video-directory names, or "
                "thresholds must be tuned from a separately prepared full-trajectory replay."
            ),
            "evaluated_configurations": 0,
        }
    rows.sort(
        key=lambda row: (
            float(row["joint_success_rate"]),
            -int(row["false_switches"]),
            -float(row["miss_rate"]),
            -float(row["mean_absolute_delay"])
            if math.isfinite(float(row["mean_absolute_delay"]))
            else -float("inf"),
        ),
        reverse=True,
    )
    with (output_dir / "validation_grid.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = rows[0]
    return {
        "selection_rule": (
            "maximize joint timing-and-choice success, then minimize false "
            "switches, misses, and absolute delay"
        ),
        "timing_tolerance": args.timing_tolerance,
        "best": best,
        "evaluated_configurations": len(rows),
    }


def main() -> None:
    args = parse_args()
    if len(args.phase) < 2:
        raise ValueError("At least two --phase specifications are required")
    if args.endpoint_frames <= 0 or args.guard_frames < 0 or args.batch_size <= 0:
        raise ValueError("endpoint_frames/batch_size must be positive; guard_frames non-negative")
    if args.timing_tolerance < 0:
        raise ValueError("timing_tolerance must be non-negative")
    if any(not math.isfinite(value) for value in args.theta_grid):
        raise ValueError("theta_grid values must be finite")
    if any(not math.isfinite(value) or value < 0 for value in args.margin_grid):
        raise ValueError("margin_grid values must be finite and non-negative")
    if any(not 0 < value <= 1 for value in args.start_probability_grid):
        raise ValueError("start_probability_grid values must be in (0, 1]")
    if any(value <= 0 for value in args.consecutive_grid):
        raise ValueError("consecutive_grid values must be positive")
    phase_names = tuple(spec[0] for spec in args.phase)
    if len(phase_names) != len(set(phase_names)):
        raise ValueError("Phase names must be unique")
    guards = {(source, target) for source, target in args.guard_transition}
    unknown_guard_phases = sorted(
        {name for edge in guards for name in edge} - set(phase_names)
    )
    if unknown_guard_phases:
        raise ValueError(f"Guard transitions contain unknown phases: {unknown_guard_phases}")
    transition_counts: dict[tuple[str, str], int] = {}
    for source, target, raw_count in args.transition_count:
        edge = (source, target)
        if edge in transition_counts:
            raise ValueError(f"Duplicate transition count for {source}->{target}")
        if source not in phase_names or target not in phase_names or source == target:
            raise ValueError(f"Invalid transition-count edge: {source}->{target}")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(
                f"Invalid transition count for {source}->{target}: {raw_count!r}"
            ) from exc
        if count <= 0:
            raise ValueError("Transition counts must be positive integers")
        transition_counts[edge] = count

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")
    preprocess = T.Compose([T.Resize((112, 112), antialias=True)])
    models: dict[str, torch.nn.Module] = {}
    calib_datasets: dict[str, Path] = {}
    valid_datasets: dict[str, Path] = {}
    checkpoints: dict[str, str] = {}
    for name, checkpoint, calib_dataset, valid_dataset in args.phase:
        checkpoint_path = Path(checkpoint).resolve()
        calib_path = Path(calib_dataset).resolve()
        valid_path = Path(valid_dataset).resolve()
        print(f"[calibration] Loading {name}: {checkpoint_path}", flush=True)
        models[name] = tcc_module.load_tcc_model(
            str(checkpoint_path), device=device, embedding_size=args.embedding_size
        )
        calib_datasets[name] = calib_path
        valid_datasets[name] = valid_path
        checkpoints[name] = str(checkpoint_path)

    print("[calibration] Encoding calibration frames with every encoder...", flush=True)
    fit_data = build_router_dataset(
        models,
        calib_datasets,
        endpoint_frames=args.endpoint_frames,
        guard_frames=args.guard_frames,
        guard_transitions=guards,
        video_limit=args.max_calib_videos,
        preprocess=preprocess,
        device=device,
        batch_size=args.batch_size,
    )
    artifact = router_module.fit_calibrated_router(
        fit_data,
        phase_names,
        min_scale=args.min_scale,
        logistic_l2=args.logistic_l2,
        transition_counts=transition_counts,
        metadata={
            "endpoint_frames": args.endpoint_frames,
            "guard_frames": args.guard_frames,
            "guard_transitions": sorted([list(edge) for edge in guards]),
            "checkpoints": checkpoints,
            "calibration_datasets": {
                name: str(path) for name, path in calib_datasets.items()
            },
            "validation_datasets": {
                name: str(path) for name, path in valid_datasets.items()
            },
        },
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "calibrated_router.pt"
    torch.save(artifact.to_payload(), artifact_path)
    print(f"[calibration] Saved artifact: {artifact_path}", flush=True)

    detector_summary: dict[str, dict[str, float]] = {}
    for phase in phase_names:
        probabilities, labels, delta = detector_values(artifact, fit_data, phase)
        detector_summary[phase] = {
            "weight": artifact.start_weights[phase],
            "bias": artifact.start_biases[phase],
            "tau_start": artifact.start_scales[phase],
            "tau_end": artifact.end_scales[phase],
            "fit_auc": binary_auc(probabilities, labels),
            "positive_delta_median": float(torch.median(delta[labels]).item()),
            "negative_delta_median": float(torch.median(delta[~labels]).item()),
        }

    print("[calibration] Encoding validation frames with every encoder...", flush=True)
    validation_data = build_router_dataset(
        models,
        valid_datasets,
        endpoint_frames=args.endpoint_frames,
        guard_frames=args.guard_frames,
        guard_transitions=guards,
        video_limit=args.max_valid_videos,
        preprocess=preprocess,
        device=device,
        batch_size=args.batch_size,
    )
    validation_summary = validation_start_report(artifact, validation_data, output_dir)
    decision_tuning = (
        None
        if args.skip_grid_search
        else tune_decision_parameters(artifact, validation_data, args, output_dir)
    )
    summary = {
        "artifact": str(artifact_path),
        "phase_names": list(phase_names),
        "endpoint_frames": args.endpoint_frames,
        "guard_frames": args.guard_frames,
        "guard_transitions": sorted([list(edge) for edge in guards]),
        "transition_counts": {
            f"{source}->{target}": count
            for (source, target), count in artifact.transition_counts.items()
        },
        "transition_priors": {
            f"{source}->{target}": probability
            for (source, target), probability in artifact.transition_priors.items()
        },
        "detectors": detector_summary,
        "validation": validation_summary,
        "decision_tuning": decision_tuning,
        "note": (
            "Grid-search results are offline replay starting points. Confirm the "
            "selected theta/m/start floor/n on closed-loop policy rollouts."
        ),
    }
    with (output_dir / "calibration_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    if args.save_embedding_cache:
        torch.save(fit_data.to_payload(), output_dir / "fit_embeddings.pt")
        torch.save(validation_data.to_payload(), output_dir / "validation_embeddings.pt")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
