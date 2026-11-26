#!/usr/bin/env python3
"""Evaluate raw nearest-start routing across phase-specific TCC encoders."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import torch
import torchvision.io as io
import torchvision.transforms as T


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "source" / "m_dVrk" / "m_dVrk"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_repo_module(name: str, path: Path) -> ModuleType:
    """Load a leaf module without importing m_dVrk's IsaacLab entrypoint."""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tcc_module = load_repo_module("m_dvrk_tcc_standalone", PACKAGE_ROOT / "hrl" / "tcc.py")
router_module = load_repo_module(
    "m_dvrk_phase_router_standalone", PACKAGE_ROOT / "hrl" / "phase_router.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        nargs=4,
        action="append",
        required=True,
        metavar=("NAME", "CHECKPOINT", "TRAIN_DATASET", "VALID_DATASET"),
        help=(
            "Phase specification. Repeat once per phase: name, TCC checkpoint, "
            "training dataset used for the start mean, and validation dataset."
        ),
    )
    parser.add_argument(
        "--prototype_frames",
        type=int,
        default=1,
        help="Number of initial frames per training video used for the start mean (default: first frame only).",
    )
    parser.add_argument(
        "--query_frames",
        type=int,
        default=1,
        help="Number of initial frames per validation video evaluated independently (default: first frame only).",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max_train_videos",
        type=int,
        default=None,
        help="Optional deterministic subset size for quick experiments.",
    )
    parser.add_argument(
        "--max_valid_videos",
        type=int,
        default=None,
        help="Optional deterministic subset size for quick experiments.",
    )
    parser.add_argument(
        "--external_query",
        nargs=2,
        action="append",
        default=[],
        metavar=("EXPECTED_PHASE", "IMAGE"),
        help=(
            "Evaluate an additional image, such as a real online transition "
            "frame. May be repeated."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/evaluations/phase_router_raw",
    )
    return parser.parse_args()


def video_directories(dataset: Path, limit: int | None) -> list[Path]:
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")
    videos = sorted(path for path in dataset.iterdir() if path.is_dir())
    return videos if limit is None else videos[:limit]


def frame_paths(video: Path, count: int) -> list[Path]:
    frames = sorted(
        path for path in video.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    return frames[:count]


def load_image_batch(paths: list[Path], preprocess: T.Compose, device: torch.device) -> torch.Tensor:
    images = []
    for path in paths:
        image = io.read_image(str(path))[:3].float() / 255.0
        images.append(image)
    if not images:
        raise ValueError("Cannot load an empty image batch")
    return preprocess(torch.stack(images, dim=0)).to(device)


def encode_paths(
    model: torch.nn.Module,
    paths: list[Path],
    preprocess: T.Compose,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for offset in range(0, len(paths), batch_size):
        batch_paths = paths[offset : offset + batch_size]
        batch = load_image_batch(batch_paths, preprocess, device)
        with torch.no_grad():
            outputs.append(model(batch).detach().cpu())
    if not outputs:
        raise ValueError("No frames were available for encoding")
    return torch.cat(outputs, dim=0)


def compute_start_embedding(
    model: torch.nn.Module,
    dataset: Path,
    frame_count: int,
    video_limit: int | None,
    preprocess: T.Compose,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, int, int]:
    per_video: list[torch.Tensor] = []
    total_frames = 0
    for video in video_directories(dataset, video_limit):
        paths = frame_paths(video, frame_count)
        if not paths:
            continue
        embeddings = encode_paths(model, paths, preprocess, device, batch_size)
        per_video.append(embeddings)
        total_frames += len(paths)
    prototype = router_module.mean_start_embedding(per_video)
    return prototype, len(per_video), total_frames


def main() -> None:
    args = parse_args()
    if len(args.phase) < 2:
        raise ValueError("At least two --phase specifications are required")
    if args.prototype_frames <= 0 or args.query_frames <= 0:
        raise ValueError("Frame counts must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    phase_names = [spec[0] for spec in args.phase]
    if len(set(phase_names)) != len(phase_names):
        raise ValueError("Phase names must be unique")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")

    preprocess = T.Compose([T.Resize((112, 112), antialias=True)])
    models: dict[str, torch.nn.Module] = {}
    starts: dict[str, torch.Tensor] = {}
    datasets: dict[str, tuple[Path, Path]] = {}
    prototype_stats: dict[str, dict[str, int | str]] = {}

    for name, checkpoint, train_dataset, valid_dataset in args.phase:
        print(f"[router] Loading {name} encoder: {checkpoint}", flush=True)
        model = tcc_module.load_tcc_model(
            checkpoint, device=device, embedding_size=args.embedding_size
        )
        train_path = Path(train_dataset).resolve()
        valid_path = Path(valid_dataset).resolve()
        start, videos, frames = compute_start_embedding(
            model,
            train_path,
            args.prototype_frames,
            args.max_train_videos,
            preprocess,
            device,
            args.batch_size,
        )
        models[name] = model
        starts[name] = start
        datasets[name] = (train_path, valid_path)
        prototype_stats[name] = {
            "checkpoint": str(Path(checkpoint).resolve()),
            "train_dataset": str(train_path),
            "videos": videos,
            "frames": frames,
        }
        print(
            f"[router] {name} start mean: {videos} videos, {frames} frames",
            flush=True,
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "phase_names": phase_names,
            "start_embeddings": starts,
            "prototype_frames": args.prototype_frames,
            "stats": prototype_stats,
        },
        output_dir / "start_embeddings.pt",
    )

    rows: list[dict[str, object]] = []
    confusion: dict[str, Counter[str]] = {
        name: Counter({candidate: 0 for candidate in phase_names})
        for name in phase_names
    }

    for true_phase in phase_names:
        valid_path = datasets[true_phase][1]
        for video in video_directories(valid_path, args.max_valid_videos):
            paths = frame_paths(video, args.query_frames)
            for offset in range(0, len(paths), args.batch_size):
                batch_paths = paths[offset : offset + args.batch_size]
                images = load_image_batch(batch_paths, preprocess, device)
                encoded: dict[str, torch.Tensor] = {}
                with torch.no_grad():
                    for candidate in phase_names:
                        encoded[candidate] = models[candidate](images).detach().cpu()

                selection = router_module.select_phase_by_nearest_start(
                    encoded, starts, phase_order=phase_names
                )
                for index, path in enumerate(batch_paths):
                    predicted = selection.selected_phases[index]
                    confusion[true_phase][predicted] += 1
                    row: dict[str, object] = {
                        "frame": str(path),
                        "video": video.name,
                        "true_phase": true_phase,
                        "predicted_phase": predicted,
                        "correct": predicted == true_phase,
                        "margin": float(selection.margins[index]),
                    }
                    for phase_index, candidate in enumerate(phase_names):
                        row[f"distance_{candidate}"] = float(
                            selection.squared_distances[index, phase_index]
                        )
                    rows.append(row)

    if not rows:
        raise ValueError("No validation frames were found")

    with (output_dir / "frame_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    external_rows: list[dict[str, object]] = []
    for expected_phase, image_path_raw in args.external_query:
        if expected_phase not in phase_names:
            raise ValueError(
                f"Unknown external-query phase {expected_phase!r}; "
                f"expected one of {phase_names}"
            )
        image_path = Path(image_path_raw).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"External query does not exist: {image_path}")
        images = load_image_batch([image_path], preprocess, device)
        with torch.no_grad():
            encoded = {
                candidate: models[candidate](images).detach().cpu()
                for candidate in phase_names
            }
        selection = router_module.select_phase_by_nearest_start(
            encoded, starts, phase_order=phase_names
        )
        external_row: dict[str, object] = {
            "frame": str(image_path),
            "expected_phase": expected_phase,
            "predicted_phase": selection.selected_phases[0],
            "correct": selection.selected_phases[0] == expected_phase,
            "margin": float(selection.margins[0]),
        }
        for phase_index, candidate in enumerate(phase_names):
            external_row[f"distance_{candidate}"] = float(
                selection.squared_distances[0, phase_index]
            )
        external_rows.append(external_row)

    if external_rows:
        with (output_dir / "external_predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(external_rows[0]))
            writer.writeheader()
            writer.writerows(external_rows)

    correct = sum(bool(row["correct"]) for row in rows)
    per_phase_accuracy = {}
    for name in phase_names:
        total = sum(confusion[name].values())
        per_phase_accuracy[name] = confusion[name][name] / total if total else 0.0

    metrics = {
        "method": "raw_squared_euclidean_nearest_start",
        "phase_names": phase_names,
        "prototype_frames": args.prototype_frames,
        "query_frames": args.query_frames,
        "num_queries": len(rows),
        "accuracy": correct / len(rows),
        "per_phase_accuracy": per_phase_accuracy,
        "confusion_matrix": {
            true_phase: dict(confusion[true_phase]) for true_phase in phase_names
        },
        "prototype_stats": prototype_stats,
        "external_queries": external_rows,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)

    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[router] Results written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
