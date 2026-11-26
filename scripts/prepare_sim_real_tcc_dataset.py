#!/usr/bin/env python3
"""Build a phase-specific XIRL dataset that mixes simulation and real videos.

The builder does not copy image data. Simulation trajectories are linked as
whole directories, while annotated real-video frames are linked into one
phase-specific trajectory per source video. This keeps the original datasets
immutable and makes the experiment cheap to reproduce.

Expected layouts::

    SIM_ROOT/{train,valid}/phase_K/VIDEO_ID/*.png
    REAL_ROOT/{train,val}/phase_0/VIDEO_ID/*.png
    ANNOTATIONS_ROOT/videoXX_phases.txt  # "frame_index phase_id"

The real dataset contains complete task recordings. Labels 0..4 correspond to
the five SPARTAN phases; label 5 is the post-task terminal segment and is not
included by default.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


DEFAULT_PHASES = (0, 1, 2, 3, 4)


def numeric_key(path: Path) -> int:
    """Return a numeric sort key for XIRL-compatible file/directory names."""

    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Expected a numeric name, got: {path}") from exc


def create_symlink(source: Path, destination: Path, *, is_directory: bool) -> None:
    """Create an absolute symlink, accepting an identical existing link."""

    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    if os.path.lexists(destination):
        if destination.is_symlink() and destination.resolve() == source:
            return
        raise FileExistsError(f"Refusing to replace existing path: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=is_directory)


def read_phase_annotations(path: Path) -> dict[int, list[int]]:
    """Read ``frame phase`` annotations grouped by phase."""

    grouped: dict[int, list[int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(f"Malformed annotation {path}:{line_number}: {line!r}")
            frame_index, phase_id = map(int, fields)
            grouped.setdefault(phase_id, []).append(frame_index)
    return grouped


def subsample_with_endpoints(indices: Iterable[int], stride: int) -> list[int]:
    """Subsample an ordered segment while retaining its first and last frame."""

    ordered = list(indices)
    if not ordered:
        return []
    selected = ordered[::stride]
    if selected[-1] != ordered[-1]:
        selected.append(ordered[-1])
    return selected


def limit_uniformly(indices: list[int], max_count: int) -> list[int]:
    """Uniformly limit an ordered sequence, preserving both endpoints."""

    if max_count == 0 or len(indices) <= max_count:
        return indices
    if max_count < 2:
        raise ValueError("A non-zero frame limit must be at least 2")
    last = len(indices) - 1
    positions = [round(i * last / (max_count - 1)) for i in range(max_count)]
    return [indices[position] for position in positions]


def discover_real_videos(
    real_root: Path,
    source_video_root: Path,
) -> tuple[dict[int, tuple[str, Path, str]], list[str]]:
    """Map extracted directory ordinals back to their source-video stems.

    The historical extractor named output directories from the position of a
    video in the lexicographically sorted ``*.mp4`` list, not from the numeric
    portion of its filename. Reconstructing that ordering matters because the
    source directory contains ``video23_WRONG_EXEC.mp4`` and all later folder
    ordinals are consequently shifted by one.
    """

    source_videos = sorted(source_video_root.glob("*.mp4"), key=lambda path: path.name)
    if not source_videos:
        raise FileNotFoundError(f"No source MP4 videos found below {source_video_root}")

    videos: dict[int, tuple[str, Path, str]] = {}
    notes: list[str] = []
    for source_split, target_split in (("train", "train"), ("valid", "valid"), ("val", "valid")):
        phase_dir = real_root / source_split / "phase_0"
        if not phase_dir.is_dir():
            continue
        for video_dir in sorted((p for p in phase_dir.iterdir() if p.is_dir()), key=numeric_key):
            video_id = numeric_key(video_dir)
            if video_id in videos:
                previous = videos[video_id][1]
                raise ValueError(f"Real video {video_id} occurs twice: {previous}, {video_dir}")
            if video_id >= len(source_videos):
                raise ValueError(
                    f"Extracted directory {video_dir} has ordinal {video_id}, but only "
                    f"{len(source_videos)} source videos exist"
                )
            videos[video_id] = (target_split, video_dir, source_videos[video_id].stem)
    if not videos:
        notes.append(f"No real videos found below {real_root}")
    return videos, notes


def link_simulation_videos(
    sim_root: Path,
    output_root: Path,
    phases: tuple[int, ...],
) -> dict[str, dict[str, int]]:
    """Link each phase-specific simulation trajectory into the mixed dataset."""

    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "valid"):
        counts[split] = {}
        for phase_id in phases:
            source_dir = sim_root / split / f"phase_{phase_id}"
            if not source_dir.is_dir():
                raise FileNotFoundError(source_dir)
            videos = sorted((p for p in source_dir.iterdir() if p.is_dir()), key=numeric_key)
            destination_dir = output_root / split / f"phase_{phase_id}"
            destination_dir.mkdir(parents=True, exist_ok=True)
            for output_id, video_dir in enumerate(videos):
                create_symlink(
                    video_dir,
                    destination_dir / f"{output_id:06d}",
                    is_directory=True,
                )
            counts[split][f"phase_{phase_id}"] = len(videos)
    return counts


def link_real_phase_segments(
    real_root: Path,
    annotations_root: Path,
    output_root: Path,
    phases: tuple[int, ...],
    frame_stride: int,
    max_frames_per_phase: int,
    video_offset: int,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], list[str]]:
    """Split complete real recordings into phase trajectories using annotations."""

    videos, notes = discover_real_videos(real_root, annotations_root)
    video_counts = {split: {f"phase_{p}": 0 for p in phases} for split in ("train", "valid")}
    frame_counts = {split: {f"phase_{p}": 0 for p in phases} for split in ("train", "valid")}

    for video_id, (split, frame_dir, source_stem) in sorted(videos.items()):
        annotation_path = annotations_root / f"{source_stem}_phases.txt"
        if not annotation_path.is_file():
            notes.append(
                f"Skipped real folder {video_id:02d} ({source_stem}): "
                f"missing {annotation_path.name}"
            )
            continue

        annotations = read_phase_annotations(annotation_path)
        for phase_id in phases:
            selected = subsample_with_endpoints(annotations.get(phase_id, ()), frame_stride)
            selected = limit_uniformly(selected, max_frames_per_phase)
            if not selected:
                notes.append(
                    f"Skipped folder {video_id:02d} ({source_stem}) "
                    f"phase_{phase_id}: no annotations"
                )
                continue

            destination_dir = (
                output_root
                / split
                / f"phase_{phase_id}"
                / f"{video_offset + video_id:06d}"
            )
            destination_dir.mkdir(parents=True, exist_ok=True)
            for output_frame, source_frame in enumerate(selected):
                source_path = frame_dir / f"{source_frame:05d}.png"
                create_symlink(
                    source_path,
                    destination_dir / f"{output_frame:06d}.png",
                    is_directory=False,
                )

            key = f"phase_{phase_id}"
            video_counts[split][key] += 1
            frame_counts[split][key] += len(selected)

    return video_counts, frame_counts, notes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phases", type=int, nargs="+", default=DEFAULT_PHASES)
    parser.add_argument(
        "--real-frame-stride",
        type=int,
        default=5,
        help="Keep every Nth real frame within a phase, always retaining endpoints.",
    )
    parser.add_argument(
        "--real-video-offset",
        type=int,
        default=100_000,
        help="Numeric ID offset used to avoid collisions with simulation videos.",
    )
    parser.add_argument(
        "--real-frames-per-phase",
        type=int,
        default=0,
        help=(
            "Uniformly cap each real phase segment after stride sampling. "
            "Zero keeps every selected frame; values must otherwise be >= 2."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.real_frame_stride <= 0:
        raise ValueError("--real-frame-stride must be positive")
    if args.real_frames_per_phase == 1 or args.real_frames_per_phase < 0:
        raise ValueError("--real-frames-per-phase must be zero or at least 2")
    phases = tuple(dict.fromkeys(args.phases))
    if not phases or any(phase < 0 for phase in phases):
        raise ValueError("--phases must contain non-negative phase IDs")

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. "
            "Use a new path so experiments remain immutable."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    sim_counts = link_simulation_videos(args.sim_root, output_root, phases)
    real_video_counts, real_frame_counts, notes = link_real_phase_segments(
        args.real_root,
        args.annotations_root,
        output_root,
        phases,
        args.real_frame_stride,
        args.real_frames_per_phase,
        args.real_video_offset,
    )

    manifest = {
        "sim_root": str(args.sim_root.resolve()),
        "real_root": str(args.real_root.resolve()),
        "annotations_root": str(args.annotations_root.resolve()),
        "output_root": str(output_root),
        "phases": list(phases),
        "real_frame_stride": args.real_frame_stride,
        "real_frames_per_phase": args.real_frames_per_phase,
        "real_video_offset": args.real_video_offset,
        "simulation_video_counts": sim_counts,
        "real_video_counts": real_video_counts,
        "real_linked_frame_counts": real_frame_counts,
        "notes": notes,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"Mixed dataset ready: {output_root}")


if __name__ == "__main__":
    main()
