#!/usr/bin/env python3
"""Recalibrate one TCC router edge from true pre-parking policy handoffs."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence

import torch
import torchvision.io as io
import torchvision.transforms as T


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "source" / "m_dVrk" / "m_dVrk"


def load_repo_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tcc_module = load_repo_module("handoff_router_tcc", PACKAGE_ROOT / "hrl" / "tcc.py")
router_module = load_repo_module(
    "handoff_router_core", PACKAGE_ROOT / "hrl" / "calibrated_phase_router.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_artifact", required=True)
    parser.add_argument("--capture_root", required=True)
    parser.add_argument("--source_phase", required=True)
    parser.add_argument("--target_phase", required=True)
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--target_checkpoint", required=True)
    parser.add_argument("--endpoint_frames", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--end_prototype_frames", type=int, default=None)
    parser.add_argument("--end_scale_frames", type=int, default=None)
    parser.add_argument("--start_prototype_frames", type=int, default=None)
    parser.add_argument("--start_scale_frames", type=int, default=None)
    parser.add_argument("--guard_steps", type=int, default=10)
    parser.add_argument("--validation_fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min_scale", type=float, default=1.0e-8)
    parser.add_argument("--logistic_l2", type=float, default=1.0e-3)
    parser.add_argument("--transition_threshold", type=float, default=-2.0)
    parser.add_argument("--start_probability_threshold", type=float, default=0.3)
    parser.add_argument("--consecutive_steps", type=int, default=3)
    parser.add_argument(
        "--theta_grid",
        type=float,
        nargs="+",
        default=[-3.0, -2.5, -2.0, -1.5, -1.0, -0.75, -0.5],
    )
    parser.add_argument(
        "--start_probability_grid",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
    )
    parser.add_argument(
        "--consecutive_grid", type=int, nargs="+", default=[3, 5]
    )
    parser.add_argument("--timing_tolerance_steps", type=int, default=5)
    parser.add_argument(
        "--earliest_allowed_delay_steps",
        type=int,
        default=0,
        help="Smallest accepted switch delay relative to the symbolic boundary.",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/evaluations/phase_router_handoff_recalibrated",
    )
    return parser.parse_args()


def load_base_artifact(path: Path) -> object:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("Router artifact must contain a mapping")
    return router_module.CalibratedRouterArtifact.from_payload(payload)


def capture_episodes(
    root: Path, source_phase: str, target_phase: str
) -> dict[str, tuple[list[Path], int, float]]:
    episodes: dict[str, tuple[list[Path], int, float]] = {}
    for episode_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata_path = episode_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("format") != "preparking_handoff_v1":
            raise ValueError(f"Unsupported handoff capture: {metadata_path}")
        if payload.get("source_phase") != source_phase:
            continue
        if payload.get("target_phase") != target_phase:
            continue
        rows = payload.get("frames")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Capture contains no frames: {metadata_path}")
        paths = [episode_dir / str(row["file"]) for row in rows]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"Capture has missing frame files: {metadata_path}")
        boundary = int(payload["boundary_index"])
        capture_hz = float(payload["capture_hz"])
        if not 0 <= boundary < len(paths) or capture_hz <= 0:
            raise ValueError(f"Invalid boundary or capture rate: {metadata_path}")
        episodes[episode_dir.name] = (paths, boundary, capture_hz)
    if len(episodes) < 3:
        raise ValueError("At least three complete captured episodes are required")
    return episodes


def encode_paths(
    paths: Sequence[Path],
    models: Mapping[str, torch.nn.Module],
    preprocess: T.Compose,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    outputs: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    for offset in range(0, len(paths), batch_size):
        batch_paths = paths[offset : offset + batch_size]
        images = torch.stack(
            [io.read_image(str(path))[:3].float() / 255.0 for path in batch_paths]
        )
        images = preprocess(images).to(device)
        with torch.no_grad():
            for name, model in models.items():
                outputs[name].append(model(images).detach().cpu())
    return {name: torch.cat(parts) for name, parts in outputs.items()}


def encode_captures(
    episodes: Mapping[str, tuple[list[Path], int, float]],
    source_phase: str,
    target_phase: str,
    source_checkpoint: Path,
    target_checkpoint: Path,
    device: torch.device,
    embedding_size: int,
    batch_size: int,
) -> tuple[dict[str, object], float]:
    models = {
        source_phase: tcc_module.load_tcc_model(
            str(source_checkpoint), device=device, embedding_size=embedding_size
        ),
        target_phase: tcc_module.load_tcc_model(
            str(target_checkpoint), device=device, embedding_size=embedding_size
        ),
    }
    preprocess = T.Compose([T.Resize((112, 112), antialias=True)])
    trajectories: dict[str, object] = {}
    rates: set[float] = set()
    for episode_id, (paths, boundary, capture_hz) in episodes.items():
        encoded = encode_paths(paths, models, preprocess, device, batch_size)
        trajectories[episode_id] = router_module.HandoffTrajectory(
            source_embeddings=encoded[source_phase],
            target_embeddings=encoded[target_phase],
            boundary_index=boundary,
        )
        rates.add(capture_hz)
    if len(rates) != 1:
        raise ValueError(f"Capture episodes use inconsistent rates: {sorted(rates)}")
    return trajectories, rates.pop()


def split_trajectories(
    trajectories: Mapping[str, object], validation_fraction: float, seed: int
) -> tuple[dict[str, object], dict[str, object]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    names = sorted(trajectories)
    random.Random(seed).shuffle(names)
    validation_count = max(1, int(round(len(names) * validation_fraction)))
    validation_count = min(validation_count, len(names) - 2)
    validation_names = set(names[:validation_count])
    fit = {name: value for name, value in trajectories.items() if name not in validation_names}
    validation = {name: trajectories[name] for name in names if name in validation_names}
    return fit, validation


def mean_absolute_delay(metrics: Mapping[str, object]) -> float:
    delays = [
        abs(int(row["delay"]))
        for row in metrics["rows"]
        if row["delay"] is not None
    ]
    return sum(delays) / len(delays) if delays else float("inf")


def tune_decision_parameters(
    artifact: object,
    trajectories: Mapping[str, object],
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for theta, start_probability, consecutive_steps in itertools.product(
        args.theta_grid,
        args.start_probability_grid,
        args.consecutive_grid,
    ):
        metrics = router_module.replay_handoff_trajectories(
            artifact,
            trajectories,
            args.source_phase,
            args.target_phase,
            transition_threshold=theta,
            start_probability_threshold=start_probability,
            consecutive_steps=consecutive_steps,
            timing_tolerance=args.timing_tolerance_steps,
            earliest_allowed_delay=args.earliest_allowed_delay_steps,
        )
        rows.append(
            {
                "theta": theta,
                "start_probability": start_probability,
                "consecutive_steps": consecutive_steps,
                "metrics": metrics,
                "mean_absolute_delay": mean_absolute_delay(metrics),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["metrics"]["timing_hit_rate"]),
            -int(row["metrics"]["early_switches"]),
            -int(row["metrics"]["misses"]),
            -float(row["mean_absolute_delay"]),
            float(row["theta"]),
            float(row["start_probability"]),
            int(row["consecutive_steps"]),
        ),
        reverse=True,
    )
    return rows[0], rows


def main() -> None:
    args = parse_args()
    if args.source_phase == args.target_phase:
        raise ValueError("source_phase and target_phase must differ")
    if not args.endpoint_frames or any(value <= 0 for value in args.endpoint_frames):
        raise ValueError("endpoint_frames must contain positive values")
    if len(args.endpoint_frames) != len(set(args.endpoint_frames)):
        raise ValueError("endpoint_frames values must be unique")
    explicit_windows = (
        args.end_prototype_frames,
        args.end_scale_frames,
        args.start_prototype_frames,
        args.start_scale_frames,
    )
    if any(value is not None and value <= 0 for value in explicit_windows):
        raise ValueError("Explicit prototype and scale windows must be positive")
    if args.guard_steps < 0 or args.batch_size <= 0:
        raise ValueError("guard_steps must be non-negative and batch_size positive")
    if not args.theta_grid or any(not math.isfinite(value) for value in args.theta_grid):
        raise ValueError("theta_grid must contain finite values")
    if not args.start_probability_grid or any(
        not 0.0 < value <= 1.0 for value in args.start_probability_grid
    ):
        raise ValueError("start_probability_grid values must lie in (0, 1]")
    if not args.consecutive_grid or any(value <= 0 for value in args.consecutive_grid):
        raise ValueError("consecutive_grid values must be positive")
    capture_root = Path(args.capture_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_artifact = load_base_artifact(Path(args.base_artifact).resolve())
    episodes = capture_episodes(capture_root, args.source_phase, args.target_phase)
    trajectories, capture_hz = encode_captures(
        episodes,
        args.source_phase,
        args.target_phase,
        Path(args.source_checkpoint).resolve(),
        Path(args.target_checkpoint).resolve(),
        torch.device(args.device),
        args.embedding_size,
        args.batch_size,
    )
    fit, validation = split_trajectories(
        trajectories, args.validation_fraction, args.seed
    )
    torch.save(
        {
            "format": "preparking_handoff_embeddings_v1",
            "source_phase": args.source_phase,
            "target_phase": args.target_phase,
            "capture_hz": capture_hz,
            "trajectories": {
                name: {
                    "source_embeddings": value.source_embeddings,
                    "target_embeddings": value.target_embeddings,
                    "boundary_index": value.boundary_index,
                }
                for name, value in trajectories.items()
            },
            "fit_ids": sorted(fit),
            "validation_ids": sorted(validation),
        },
        output_dir / "handoff_embeddings.pt",
    )

    baseline = router_module.replay_handoff_trajectories(
        base_artifact,
        validation,
        args.source_phase,
        args.target_phase,
        transition_threshold=args.transition_threshold,
        start_probability_threshold=args.start_probability_threshold,
        consecutive_steps=args.consecutive_steps,
        timing_tolerance=args.timing_tolerance_steps,
        earliest_allowed_delay=args.earliest_allowed_delay_steps,
    )
    results: list[dict[str, object]] = []
    for endpoint_frames in sorted(args.endpoint_frames):
        artifact, stats = router_module.fit_handoff_calibration(
            base_artifact,
            fit,
            args.source_phase,
            args.target_phase,
            endpoint_frames=endpoint_frames,
            guard_frames=args.guard_steps,
            end_prototype_frames=args.end_prototype_frames,
            end_scale_frames=args.end_scale_frames,
            start_prototype_frames=args.start_prototype_frames,
            start_scale_frames=args.start_scale_frames,
            min_scale=args.min_scale,
            logistic_l2=args.logistic_l2,
            metadata={
                "capture_root": str(capture_root),
                "capture_hz": capture_hz,
                "fit_ids": sorted(fit),
                "validation_ids": sorted(validation),
            },
        )
        best_decision, decision_grid = tune_decision_parameters(
            artifact,
            validation,
            args,
        )
        fit_metrics = router_module.replay_handoff_trajectories(
            artifact,
            fit,
            args.source_phase,
            args.target_phase,
            transition_threshold=float(best_decision["theta"]),
            start_probability_threshold=float(best_decision["start_probability"]),
            consecutive_steps=int(best_decision["consecutive_steps"]),
            timing_tolerance=args.timing_tolerance_steps,
            earliest_allowed_delay=args.earliest_allowed_delay_steps,
        )
        validation_metrics = best_decision["metrics"]
        artifact_path = output_dir / f"calibrated_router_handoff_L{endpoint_frames}.pt"
        torch.save(artifact.to_payload(), artifact_path)
        results.append(
            {
                "endpoint_frames": endpoint_frames,
                "window_span_seconds": (endpoint_frames - 1) / capture_hz,
                "artifact": str(artifact_path),
                "stats": asdict(stats),
                "tau_end_source": artifact.end_scales[args.source_phase],
                "tau_start_target": artifact.start_scales[args.target_phase],
                "start_weight_target": artifact.start_weights[args.target_phase],
                "start_bias_target": artifact.start_biases[args.target_phase],
                "selected_decision": {
                    "theta": best_decision["theta"],
                    "start_probability": best_decision["start_probability"],
                    "consecutive_steps": best_decision["consecutive_steps"],
                    "selection_metrics": validation_metrics,
                    "mean_absolute_delay": best_decision["mean_absolute_delay"],
                },
                "evaluated_decision_configurations": len(decision_grid),
                "fit": fit_metrics,
                "validation": validation_metrics,
                "validation_mean_absolute_delay": mean_absolute_delay(validation_metrics),
            }
        )

    results.sort(
        key=lambda row: (
            float(row["validation"]["timing_hit_rate"]),
            -int(row["validation"]["early_switches"]),
            -int(row["validation"]["misses"]),
            -float(row["validation_mean_absolute_delay"]),
            -int(row["endpoint_frames"]),
        ),
        reverse=True,
    )
    summary = {
        "format": "handoff_router_recalibration_v1",
        "base_artifact": str(Path(args.base_artifact).resolve()),
        "capture_root": str(capture_root),
        "source_phase": args.source_phase,
        "target_phase": args.target_phase,
        "capture_hz": capture_hz,
        "fit_ids": sorted(fit),
        "validation_ids": sorted(validation),
        "decision_parameters": {
            "baseline_transition_threshold": args.transition_threshold,
            "baseline_start_probability_threshold": args.start_probability_threshold,
            "baseline_consecutive_steps": args.consecutive_steps,
            "theta_grid": args.theta_grid,
            "start_probability_grid": args.start_probability_grid,
            "consecutive_grid": args.consecutive_grid,
            "timing_tolerance_steps": args.timing_tolerance_steps,
            "earliest_allowed_delay_steps": args.earliest_allowed_delay_steps,
        },
        "baseline_validation": baseline,
        "selection_rule": (
            "maximize held-out timing hits, then minimize early switches, misses, "
            "absolute delay, and endpoint window length"
        ),
        "selected": results[0],
        "candidates": results,
    }
    summary_path = output_dir / "handoff_recalibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
