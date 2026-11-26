#!/usr/bin/env python3
"""Compare TCC checkpoints on held-out simulation and real trajectories.

The evaluation deliberately separates the two visual domains. For each model
it reports temporal nearest-neighbour alignment within simulation, within the
real domain, and across domains. Goal-distance quality is evaluated twice:
with a simulated-training goal prototype and with a real-training goal
prototype. This separates within-domain representation quality from actual
sim-to-real transfer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau, rankdata, spearmanr
import torch
from torchvision.io import read_image
from torchvision.transforms import functional as vision_functional


def load_repo_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REPO_ROOT = Path(__file__).resolve().parents[1]
TCC_MODULE = load_repo_module(
    "m_dvrk_tcc_sim_real_analysis",
    REPO_ROOT / "source" / "m_dVrk" / "m_dVrk" / "hrl" / "tcc.py",
)
load_tcc_model = TCC_MODULE.load_tcc_model


def numeric_key(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Expected a numeric video/frame name: {path}") from exc


def video_dirs(root: Path, *, minimum_id: int | None = None) -> list[Path]:
    videos = sorted((path for path in root.iterdir() if path.is_dir()), key=numeric_key)
    if minimum_id is not None:
        videos = [path for path in videos if numeric_key(path) >= minimum_id]
    if not videos:
        raise ValueError(f"No matching videos found below {root}")
    return videos


def frame_paths(video: Path) -> list[Path]:
    frames = sorted(
        (path for path in video.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        key=numeric_key,
    )
    if not frames:
        raise ValueError(f"No frames found below {video}")
    return frames


def uniform_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        raise ValueError("Cannot sample an empty sequence")
    if count <= 0 or length <= count:
        return list(range(length))
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]


def load_frame(path: Path, image_size: int) -> torch.Tensor:
    image = read_image(str(path))[:3].float() / 255.0
    return vision_functional.resize(
        image,
        [image_size, image_size],
        antialias=True,
    )


def load_video_tensors(
    videos: Sequence[Path],
    frames_per_video: int,
    image_size: int,
) -> list[torch.Tensor]:
    loaded: list[torch.Tensor] = []
    for index, video in enumerate(videos, start=1):
        paths = frame_paths(video)
        selected = [paths[i] for i in uniform_indices(len(paths), frames_per_video)]
        loaded.append(torch.stack([load_frame(path, image_size) for path in selected]))
        print(f"[data] {index:03d}/{len(videos):03d} {video} ({len(selected)} frames)")
    return loaded


def load_goal_frames(videos: Sequence[Path], image_size: int) -> torch.Tensor:
    return torch.stack([load_frame(frame_paths(video)[-1], image_size) for video in videos])


@torch.inference_mode()
def embed_tensor(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        outputs.append(model(images[start : start + batch_size].to(device)).cpu())
    return torch.cat(outputs, dim=0).numpy()


def embed_videos(
    model: torch.nn.Module,
    videos: Sequence[torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> list[np.ndarray]:
    return [embed_tensor(model, frames, device, batch_size) for frames in videos]


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if clean.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "count": 0}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "count": int(clean.size),
    }


def directed_alignment(
    queries: Sequence[np.ndarray],
    candidates: Sequence[np.ndarray],
    *,
    exclude_identity: bool,
) -> tuple[list[float], list[float]]:
    taus: list[float] = []
    cycles: list[float] = []
    for query_index, query in enumerate(queries):
        for candidate_index, candidate in enumerate(candidates):
            if exclude_identity and query_index == candidate_index:
                continue
            distances = cdist(query, candidate, metric="sqeuclidean")
            nearest_candidate = np.argmin(distances, axis=1)
            tau = kendalltau(np.arange(len(query)), nearest_candidate).correlation
            if tau is not None and np.isfinite(tau):
                taus.append(float(tau))

            nearest_query = np.argmin(distances, axis=0)
            reconstructed = nearest_query[nearest_candidate]
            cycles.append(float(np.mean(reconstructed == np.arange(len(query)))))
    return taus, cycles


def alignment_metrics(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray] | None = None,
) -> dict[str, dict[str, float | int]]:
    if right is None:
        taus, cycles = directed_alignment(left, left, exclude_identity=True)
    else:
        forward_taus, forward_cycles = directed_alignment(left, right, exclude_identity=False)
        reverse_taus, reverse_cycles = directed_alignment(right, left, exclude_identity=False)
        taus = forward_taus + reverse_taus
        cycles = forward_cycles + reverse_cycles
    return {"kendall_tau": summarize(taus), "two_way_cycle": summarize(cycles)}


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(np.sum(labels))
    negatives = int(np.sum(~labels))
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    positive_rank_sum = float(np.sum(ranks[labels]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def goal_metrics(
    videos: Sequence[np.ndarray],
    goal_embedding: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    progress_correlations: list[float] = []
    endpoint_aucs: list[float] = []
    final_rank_percentiles: list[float] = []
    goal_gaps: list[float] = []
    final_distances: list[float] = []

    for embeddings in videos:
        distances = np.sum((embeddings - goal_embedding[None, :]) ** 2, axis=1)
        progress = np.linspace(0.0, 1.0, len(distances))
        correlation = spearmanr(progress, -distances).correlation
        if correlation is not None and np.isfinite(correlation):
            progress_correlations.append(float(correlation))

        goal_count = max(2, math.ceil(0.10 * len(distances)))
        negative_count = max(2, math.floor(0.80 * len(distances)))
        selected_scores = np.concatenate((-distances[:negative_count], -distances[-goal_count:]))
        selected_labels = np.concatenate(
            (np.zeros(negative_count, dtype=np.int64), np.ones(goal_count, dtype=np.int64))
        )
        endpoint_aucs.append(binary_auc(selected_labels, selected_scores))

        ranks = rankdata(distances, method="average")
        denominator = max(1, len(distances) - 1)
        final_rank_percentiles.append(float((ranks[-1] - 1.0) / denominator))
        goal_gaps.append(float(np.median(distances[:negative_count]) - np.median(distances[-goal_count:])))
        final_distances.append(float(distances[-1]))

    return {
        "progress_spearman": summarize(progress_correlations),
        "endpoint_auc": summarize(endpoint_aucs),
        "final_rank_percentile": summarize(final_rank_percentiles),
        "early_minus_goal_distance": summarize(goal_gaps),
        "final_squared_distance": summarize(final_distances),
    }


def checkpoint_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Checkpoints must use LABEL=/path/to/checkpoint.ckpt")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not label:
        raise argparse.ArgumentTypeError("Checkpoint label cannot be empty")
    return label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", type=checkpoint_argument, required=True)
    parser.add_argument("--sim-train", type=Path, required=True)
    parser.add_argument("--sim-valid", type=Path, required=True)
    parser.add_argument("--mixed-train", type=Path, required=True)
    parser.add_argument("--mixed-valid", type=Path, required=True)
    parser.add_argument("--real-video-offset", type=int, default=100_000)
    parser.add_argument("--frames-per-video", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def metric(result: dict, *keys: str) -> float:
    current = result
    for key in keys:
        current = current[key]
    return float(current)


def write_markdown(results: dict[str, dict], output: Path) -> None:
    lines = [
        "# TCC Phase 0 domain comparison",
        "",
        "The real and simulated validation results are reported separately.",
        "For AUC and progress correlation, higher is better. For final-rank percentile, lower is better.",
        "`within-real` uses a goal prototype made from real training videos; `sim->real` uses the simulated goal prototype on real validation videos.",
        "",
        "## Temporal alignment",
        "",
        "| checkpoint | sim tau | real tau | cross tau | sim cycle | real cycle | cross cycle |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, result in results.items():
        values = [
            metric(result, "alignment", "sim", "kendall_tau", "mean"),
            metric(result, "alignment", "real", "kendall_tau", "mean"),
            metric(result, "alignment", "cross", "kendall_tau", "mean"),
            metric(result, "alignment", "sim", "two_way_cycle", "mean"),
            metric(result, "alignment", "real", "two_way_cycle", "mean"),
            metric(result, "alignment", "cross", "two_way_cycle", "mean"),
        ]
        lines.append(f"| {label} | " + " | ".join(f"{value:.4f}" for value in values) + " |")

    lines.extend(
        [
            "",
            "## Goal-distance metrics",
            "",
            "| checkpoint | within-sim AUC | within-sim progress | within-sim final rank | within-real AUC | within-real progress | within-real final rank | sim->real AUC | sim->real progress | sim->real final rank | real->sim AUC | real->sim progress | real->sim final rank |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, result in results.items():
        values = []
        for evaluation in ("within_sim", "within_real", "sim_to_real", "real_to_sim"):
            values.extend(
                [
                    metric(result, "goal", evaluation, "endpoint_auc", "mean"),
                    metric(result, "goal", evaluation, "progress_spearman", "mean"),
                    metric(result, "goal", evaluation, "final_rank_percentile", "mean"),
                ]
            )
        lines.append(f"| {label} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    sim_train_dirs = video_dirs(args.sim_train)
    sim_valid_dirs = video_dirs(args.sim_valid)
    real_train_dirs = video_dirs(args.mixed_train, minimum_id=args.real_video_offset)
    real_valid_dirs = video_dirs(args.mixed_valid, minimum_id=args.real_video_offset)

    print("[data] Loading simulated goal frames")
    sim_goal_images = load_goal_frames(sim_train_dirs, args.image_size)
    print("[data] Loading real goal frames")
    real_goal_images = load_goal_frames(real_train_dirs, args.image_size)
    print("[data] Loading simulated validation trajectories")
    sim_videos = load_video_tensors(sim_valid_dirs, args.frames_per_video, args.image_size)
    print("[data] Loading real validation trajectories")
    real_videos = load_video_tensors(real_valid_dirs, args.frames_per_video, args.image_size)

    results: dict[str, dict] = {}
    for label, checkpoint in args.checkpoint:
        print(f"[model] Evaluating {label}: {checkpoint}")
        model = load_tcc_model(str(checkpoint), device=device)
        sim_goal_embeddings = embed_tensor(model, sim_goal_images, device, args.batch_size)
        real_goal_embeddings = embed_tensor(model, real_goal_images, device, args.batch_size)
        sim_goal_embedding = np.mean(sim_goal_embeddings, axis=0)
        real_goal_embedding = np.mean(real_goal_embeddings, axis=0)
        sim_embeddings = embed_videos(model, sim_videos, device, args.batch_size)
        real_embeddings = embed_videos(model, real_videos, device, args.batch_size)

        results[label] = {
            "checkpoint": str(checkpoint),
            "alignment": {
                "sim": alignment_metrics(sim_embeddings),
                "real": alignment_metrics(real_embeddings),
                "cross": alignment_metrics(sim_embeddings, real_embeddings),
            },
            "goal": {
                "within_sim": goal_metrics(sim_embeddings, sim_goal_embedding),
                "within_real": goal_metrics(real_embeddings, real_goal_embedding),
                "sim_to_real": goal_metrics(real_embeddings, sim_goal_embedding),
                "real_to_sim": goal_metrics(sim_embeddings, real_goal_embedding),
            },
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_markdown(results, args.output_markdown)
    print(f"[done] JSON: {args.output_json}")
    print(f"[done] Markdown: {args.output_markdown}")


if __name__ == "__main__":
    main()
