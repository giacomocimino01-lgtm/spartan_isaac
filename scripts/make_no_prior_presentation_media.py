#!/usr/bin/env python3
"""Generate synchronized videos, filmstrips, and diagnostic plots for the no-prior run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PHASES = [f"phase_{i}" for i in range(5)]
FROM_PHASES = PHASES[:-1]
EDGE_LABELS = ["0→1", "1→2", "2→3", "3→4"]
COLORS = {
    "phase_0": "#2878B5",
    "phase_1": "#54A24B",
    "phase_2": "#D95319",
    "phase_3": "#2A9D8F",
    "phase_4": "#7B2CBF",
}
CONFIRMATION = {"phase_0": 4, "phase_1": 5, "phase_2": 4, "phase_3": 3}
CONTROL_HZ = 10.0
VIDEO_STRIDE = 5
OUTPUT_FPS = 10.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=260, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 14,
            "axes.labelsize": 11.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.alpha": 0.25,
        }
    )


def switch_delays(root: Path) -> dict[str, list[float]]:
    trace = read_csv(root / "transition_score_trace.csv")
    tcc = read_csv(root / "tcc_distance_trace.csv")
    symbolic: dict[tuple[int, str], int] = {}
    for row in tcc:
        if row["symbolic_complete"] == "True":
            key = (int(row["episode"]), row["phase"])
            symbolic.setdefault(key, int(row["total_step"]))
    result: dict[str, list[float]] = defaultdict(list)
    for row in trace:
        if row["decision"] != "switch":
            continue
        key = (int(row["episode"]), row["phase"])
        if key not in symbolic:
            continue
        delay = (int(row["total_step"]) - symbolic[key]) / CONTROL_HZ
        result[row["phase"]].append(delay)
    return result


def plot_decoder_delay(with_prior: Path, no_prior: Path, output: Path) -> None:
    datasets = [switch_delays(with_prior), switch_delays(no_prior)]
    labels = ["Sparse prior", "No prior"]
    colors = ["#8D99AE", "#2878B5"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    base = np.arange(4)
    offsets = [-0.18, 0.18]
    rng = np.random.default_rng(14)
    for data, label, color, offset in zip(datasets, labels, colors, offsets):
        groups = [data[phase] for phase in FROM_PHASES]
        positions = base + offset
        bp = ax.boxplot(groups, positions=positions, widths=0.28, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
            patch.set_edgecolor(color)
        for median in bp["medians"]:
            median.set_color(color)
            median.set_linewidth(2)
        for x, values in zip(positions, groups):
            jitter = rng.uniform(-0.055, 0.055, len(values))
            ax.scatter(x + jitter, values, color=color, s=26, alpha=0.78, zorder=3)
            ax.text(x, max(values) + 0.16, f"n={len(values)}", ha="center", color=color, fontsize=9)
        ax.plot([], [], color=color, linewidth=6, alpha=0.45, label=label)
    ax.axhline(0, color="#D62828", linestyle="--", linewidth=1.3, label="premature boundary")
    ax.set_xticks(base, EDGE_LABELS)
    ax.set_xlabel("Transition edge")
    ax.set_ylabel(r"Pure decoder delay $t_{trigger}-t_{symbolic}$ (s)")
    ax.set_title("Decoder reaction time after symbolic completion")
    ax.set_ylim(-0.2, 3.8)
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.text(0.99, 0.02, "All observed delays are non-negative", transform=ax.transAxes, ha="right", color="#2A9D8F", fontweight="bold")
    save_figure(fig, output, "05_pure_decoder_delay_by_edge")


def plot_end_start_space(no_prior: Path, output: Path) -> None:
    rows = [row for row in read_csv(no_prior / "transition_score_trace.csv") if row["decision"] == "switch"]
    fig, ax = plt.subplots(figsize=(9.4, 6.1))
    for phase, edge in zip(FROM_PHASES, EDGE_LABELS):
        group = [row for row in rows if row["phase"] == phase]
        x = np.asarray([float(row["end_log_score"]) for row in group])
        y = np.asarray([float(row["best_start_log_probability"]) for row in group])
        ax.scatter(x, y, s=54, color=COLORS[phase], alpha=0.82, edgecolor="white", linewidth=0.6, label=edge)
    all_x = np.asarray([float(row["end_log_score"]) for row in rows])
    xmin, xmax = all_x.min() - 0.18, all_x.max() + 0.18
    line_x = np.linspace(xmin, xmax, 300)
    ax.plot(line_x, -2.0 - line_x, color="#D62828", linestyle="--", linewidth=1.8, label=r"total threshold: $x+y=-2$")
    start_floor = math.log(0.6)
    ax.axhline(start_floor, color="#111111", linestyle=":", linewidth=1.7, label=r"start floor: $y=\log(0.6)$")
    ax.fill_between(line_x, np.maximum(-2.0 - line_x, start_floor), 0.05, color="#2A9D8F", alpha=0.08)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(min(start_floor - 0.12, -0.68), 0.03)
    ax.set_xlabel(r"End evidence $\log E_k(x)$")
    ax.set_ylabel(r"Start evidence $\log P(\mathrm{start}_j\mid x)$")
    ax.set_title("End/start decomposition at no-prior policy switches")
    ax.grid()
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.text(0.98, 0.96, "accepted region", transform=ax.transAxes, ha="right", va="top", color="#2A9D8F", fontweight="bold")
    save_figure(fig, output, "06_end_start_decision_space")


def replay_margin(no_prior: Path, output: Path) -> None:
    rows = read_csv(no_prior / "transition_score_trace.csv")
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["episode"]), row["phase"])].append(row)
    switch_keys = [key for key, group in grouped.items() if any(row["decision"] == "switch" for row in group)]

    margins = np.round(np.linspace(0.0, 1.0, 41), 3)
    records: list[dict[str, float]] = []
    for required_margin in margins:
        confirmed: dict[tuple[int, str], bool] = {}
        for key in switch_keys:
            streak = 0
            previous_candidate: str | None = None
            accepted = False
            for row in grouped[key]:
                support = (
                    float(row["best_transition_log_score"]) > float(row["transition_threshold"])
                    and float(row["margin"]) > required_margin
                    and float(row["best_start_log_probability"]) > math.log(0.6)
                )
                candidate = row["selected_phase"]
                if support:
                    streak = streak + 1 if candidate == previous_candidate else 1
                    previous_candidate = candidate
                else:
                    streak = 0
                    previous_candidate = None
                if streak >= CONFIRMATION[key[1]]:
                    accepted = True
                    break
            confirmed[key] = accepted
        overall = 100.0 * sum(confirmed.values()) / len(confirmed)
        phase1_values = [value for key, value in confirmed.items() if key[1] == "phase_1"]
        phase1 = 100.0 * sum(phase1_values) / len(phase1_values)
        records.append({"margin": float(required_margin), "overall": overall, "phase1": phase1})

    with (output / "07_margin_replay_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["margin", "overall_confirmed_percent", "phase1_confirmed_percent"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "margin": record["margin"],
                    "overall_confirmed_percent": record["overall"],
                    "phase1_confirmed_percent": record["phase1"],
                }
            )

    x = np.asarray([record["margin"] for record in records])
    overall = np.asarray([record["overall"] for record in records])
    phase1 = np.asarray([record["phase1"] for record in records])
    last_full = x[(overall == 100.0) & (phase1 == 100.0)].max()
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.plot(x, overall, color="#2878B5", linewidth=2.5, marker="o", markersize=4, label="All transitions retained")
    ax.plot(x, phase1, color="#D95319", linewidth=2.5, marker="s", markersize=4, label="Phase 1→2 retained")
    ax.axvline(last_full, color="#2A9D8F", linestyle="--", linewidth=1.8)
    ax.annotate(
        f"largest tested margin with 63/63 retained: m={last_full:.3f}",
        xy=(last_full, 100),
        xytext=(0.31, 91),
        arrowprops={"arrowstyle": "->", "color": "#2A9D8F"},
        color="#2A9D8F",
        fontweight="bold",
    )
    ax.set_xlabel(r"Required winner–runner-up margin $m$")
    ax.set_ylabel("Transitions confirmed within recorded trace (%)")
    ax.set_title("Offline margin replay on the no-prior traces")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-2, 104)
    ax.grid()
    ax.legend(frameon=False, loc="lower left")
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.5,
        0.025,
        "Trace ends at the original switch: non-confirmed cases are right-censored, not proven misses.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    save_figure(fig, output, "07_offline_margin_tradeoff")


def load_video_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"Could not read video {path}")
    return frames


def frame_for_step(frames: list[np.ndarray], total_step: int) -> np.ndarray:
    index = max(0, min(len(frames) - 1, (max(total_step, 1) - 1) // VIDEO_STRIDE))
    return frames[index]


def trace_context(root: Path, episode: int, phase: str) -> tuple[list[dict[str, str]], int | None, int | None]:
    rows = [
        row
        for row in read_csv(root / "transition_score_trace.csv")
        if int(row["episode"]) == episode and row["phase"] == phase
    ]
    tcc = [
        row
        for row in read_csv(root / "tcc_distance_trace.csv")
        if int(row["episode"]) == episode and row["phase"] == phase
    ]
    symbolic = next((int(row["total_step"]) for row in tcc if row["symbolic_complete"] == "True"), None)
    trigger = next((int(row["total_step"]) for row in rows if row["decision"] == "switch"), None)
    return rows, symbolic, trigger


def scores_from_rows(rows: list[dict[str, str]], candidates: list[str]) -> dict[str, np.ndarray]:
    result = {candidate: [] for candidate in candidates}
    for row in rows:
        scores = json.loads(row["transition_log_scores"])
        for candidate in candidates:
            result[candidate].append(float(scores[candidate]))
    return {candidate: np.asarray(values) for candidate, values in result.items()}


def render_chart_frame(
    rows: list[dict[str, str]],
    candidates: list[str],
    clip_start: int,
    clip_end: int,
    cursor_step: int,
    symbolic_step: int | None,
    trigger_step: int | None,
    y_limits: tuple[float, float],
    title: str,
    timeout_step: int | None = None,
) -> np.ndarray:
    window = [row for row in rows if clip_start <= int(row["total_step"]) <= clip_end]
    x = np.asarray([(int(row["total_step"]) - clip_start) / CONTROL_HZ for row in window])
    scores = scores_from_rows(window, candidates)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=100)
    for candidate in candidates:
        ax.plot(x, scores[candidate], color=COLORS[candidate], linewidth=2.0, label=candidate)
    ax.axhline(-2.0, color="#D62828", linestyle="--", linewidth=1.5, label=r"$\theta=-2$")
    if symbolic_step is not None:
        symbolic_x = (symbolic_step - clip_start) / CONTROL_HZ
        ax.axvspan(symbolic_x, (clip_end - clip_start) / CONTROL_HZ, color="#2A9D8F", alpha=0.10)
        ax.axvline(symbolic_x, color="#2A9D8F", linestyle=":", linewidth=1.5, label="symbolic complete")
    if trigger_step is not None:
        ax.axvline((trigger_step - clip_start) / CONTROL_HZ, color="#7B2CBF", linestyle=":", linewidth=1.8, label="policy switch")
    if timeout_step is not None:
        ax.axvline((timeout_step - clip_start) / CONTROL_HZ, color="#D62828", linestyle=":", linewidth=1.8, label="timeout")
    cursor_x = (cursor_step - clip_start) / CONTROL_HZ
    ax.axvline(cursor_x, color="#111111", linewidth=2.0)
    nearest = min(rows, key=lambda row: abs(int(row["total_step"]) - cursor_step))
    symbolic_now = symbolic_step is not None and cursor_step >= symbolic_step
    status = (
        f"best={nearest['selected_phase']}  score={float(nearest['best_transition_log_score']):.2f}  "
        f"margin={float(nearest['margin']):.2f}  streak={nearest['confirmation_streak']}  "
        f"symbolic={symbolic_now}  decision={nearest['decision']}"
    )
    ax.text(0.01, 0.98, status, transform=ax.transAxes, va="top", fontsize=9.3, bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"})
    ax.set_xlim(0, (clip_end - clip_start) / CONTROL_HZ)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Real control time within clip (s)")
    ax.set_ylabel("Transition log-score")
    ax.set_title(title)
    ax.grid()
    ax.legend(frameon=False, ncol=3, loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return image


def make_synchronized_clip(
    root: Path,
    output: Path,
    episode: int,
    phase: str,
    video_name: str,
    clip_start: int,
    clip_end: int,
    y_limits: tuple[float, float],
    stem: str,
    title: str,
    timeout_step: int | None = None,
) -> None:
    rows, symbolic, trigger = trace_context(root, episode, phase)
    candidates = [candidate for candidate in PHASES if candidate != phase]
    source_frames = load_video_frames(root / video_name)
    raw_path = output / f"{stem}_raw.mp4"
    final_path = output / f"{stem}.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), OUTPUT_FPS, (1440, 480))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {raw_path}")
    for total_step in range(clip_start, clip_end):
        scene = frame_for_step(source_frames, total_step)
        scene = cv2.resize(scene, (640, 480), interpolation=cv2.INTER_AREA)
        label = f"Episode {episode:03d} | {phase} | control step {total_step}"
        cv2.rectangle(scene, (0, 0), (640, 34), (0, 0, 0), -1)
        cv2.putText(scene, label, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        chart = render_chart_frame(
            rows,
            candidates,
            clip_start,
            clip_end,
            total_step,
            symbolic,
            trigger,
            y_limits,
            title,
            timeout_step,
        )
        chart = cv2.resize(chart, (800, 480), interpolation=cv2.INTER_AREA)
        writer.write(np.hstack([scene, chart]))
    writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw_path),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(final_path),
        ],
        check=True,
    )
    raw_path.unlink()


def make_filmstrip(
    root: Path,
    output: Path,
    episode: int,
    phase: str,
    video_name: str,
    clip_start: int,
    clip_end: int,
    sample_steps: list[int],
    sample_labels: list[str],
    y_limits: tuple[float, float],
    stem: str,
    title: str,
    timeout_step: int | None = None,
) -> None:
    rows, symbolic, trigger = trace_context(root, episode, phase)
    candidates = [candidate for candidate in PHASES if candidate != phase]
    frames = load_video_frames(root / video_name)
    window = [row for row in rows if clip_start <= int(row["total_step"]) <= clip_end]
    x = np.asarray([(int(row["total_step"]) - clip_start) / CONTROL_HZ for row in window])
    scores = scores_from_rows(window, candidates)
    fig = plt.figure(figsize=(14.0, 7.8))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.45], hspace=0.22, wspace=0.04)
    marker_colors = ["#2878B5", "#2A9D8F", "#D95319", "#7B2CBF"]
    for index, (step, label, color) in enumerate(zip(sample_steps, sample_labels, marker_colors)):
        ax_image = fig.add_subplot(grid[0, index])
        frame = cv2.cvtColor(frame_for_step(frames, step), cv2.COLOR_BGR2RGB)
        ax_image.imshow(frame)
        ax_image.set_title(f"{label}\nt={(step - clip_start) / CONTROL_HZ:.1f} s", color=color, fontweight="bold")
        ax_image.axis("off")
    ax = fig.add_subplot(grid[1, :])
    for candidate in candidates:
        ax.plot(x, scores[candidate], color=COLORS[candidate], linewidth=2.0, label=candidate)
    ax.axhline(-2.0, color="#D62828", linestyle="--", linewidth=1.5, label=r"$\theta=-2$")
    if symbolic is not None:
        symbolic_x = (symbolic - clip_start) / CONTROL_HZ
        ax.axvspan(symbolic_x, (clip_end - clip_start) / CONTROL_HZ, color="#2A9D8F", alpha=0.10)
        ax.axvline(symbolic_x, color="#2A9D8F", linestyle=":", linewidth=1.5, label="symbolic complete")
    if trigger is not None:
        ax.axvline((trigger - clip_start) / CONTROL_HZ, color="#7B2CBF", linestyle=":", linewidth=1.8, label="policy switch")
    if timeout_step is not None:
        ax.axvline((timeout_step - clip_start) / CONTROL_HZ, color="#D62828", linestyle=":", linewidth=1.8, label="timeout")
    for step, color in zip(sample_steps, marker_colors):
        ax.axvline((step - clip_start) / CONTROL_HZ, color=color, linewidth=1.2, alpha=0.8)
    ax.set_xlim(0, (clip_end - clip_start) / CONTROL_HZ)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Real control time within clip (s)")
    ax.set_ylabel("Transition log-score")
    ax.grid()
    ax.legend(frameon=False, ncol=4, loc="lower left")
    fig.suptitle(title, fontsize=16)
    save_figure(fig, output, stem)


def make_full_success_clip(root: Path, output: Path, episode: int) -> None:
    """Render a complete successful episode on the real 10 Hz control clock."""
    episode_row = next(
        row for row in read_csv(root / "episodes.csv") if int(row["episode"]) == episode
    )
    if episode_row["success"] != "True":
        raise ValueError(f"Episode {episode} is not a successful episode")
    total_steps = int(episode_row["total_steps"])
    rows = [
        row
        for row in read_csv(root / "transition_score_trace.csv")
        if int(row["episode"]) == episode
    ]
    rows.sort(key=lambda row: int(row["total_step"]))
    rows_by_step = {int(row["total_step"]): row for row in rows}
    by_phase = {
        phase: [row for row in rows if row["phase"] == phase]
        for phase in FROM_PHASES
    }
    tcc = [
        row
        for row in read_csv(root / "tcc_distance_trace.csv")
        if int(row["episode"]) == episode
    ]
    symbolic_steps = {
        phase: next(
            (int(row["total_step"]) for row in tcc if row["phase"] == phase and row["symbolic_complete"] == "True"),
            None,
        )
        for phase in FROM_PHASES
    }
    trigger_steps = {
        phase: next(
            (int(row["total_step"]) for row in by_phase[phase] if row["decision"] == "switch"),
            None,
        )
        for phase in FROM_PHASES
    }

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=100)
    for phase_index, phase in enumerate(PHASES):
        start = 1 if phase_index == 0 else (trigger_steps[PHASES[phase_index - 1]] or 0) + 1
        end = (trigger_steps.get(phase) or total_steps) if phase != "phase_4" else total_steps
        ax.axvspan(start / CONTROL_HZ, end / CONTROL_HZ, color="#E9ECEF", alpha=0.18 if phase_index % 2 == 0 else 0.06)
        ax.text(
            ((start + end) / 2) / CONTROL_HZ,
            0.96,
            phase.replace("phase_", "P"),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.5,
            color="#555555",
            fontweight="bold",
        )
    for candidate in PHASES:
        first = True
        for phase in FROM_PHASES:
            group = by_phase[phase]
            x_values: list[float] = []
            y_values: list[float] = []
            for row in group:
                scores = json.loads(row["transition_log_scores"])
                if candidate in scores:
                    x_values.append(int(row["total_step"]) / CONTROL_HZ)
                    y_values.append(max(-20.0, float(scores[candidate])))
            if x_values:
                ax.plot(
                    x_values,
                    y_values,
                    color=COLORS[candidate],
                    linewidth=1.35,
                    alpha=0.9,
                    label=candidate if first else None,
                )
                first = False
    ax.axhline(-2.0, color="#D62828", linestyle="--", linewidth=1.5, label=r"$\theta=-2$")
    for phase in FROM_PHASES:
        symbolic = symbolic_steps[phase]
        trigger = trigger_steps[phase]
        if symbolic is not None and trigger is not None:
            ax.axvspan(symbolic / CONTROL_HZ, trigger / CONTROL_HZ, color="#2A9D8F", alpha=0.18)
            ax.axvline(symbolic / CONTROL_HZ, color="#2A9D8F", linestyle=":", linewidth=1.0)
            ax.axvline(trigger / CONTROL_HZ, color="#7B2CBF", linestyle=":", linewidth=1.4)
    ax.set_xlim(0, total_steps / CONTROL_HZ)
    ax.set_ylim(-20.0, 0.5)
    ax.set_xlabel("Real control time (s)")
    ax.set_ylabel("Transition log-score")
    ax.set_title(f"Complete successful task — episode {episode:03d}")
    ax.grid()
    ax.legend(frameon=False, ncol=3, loc="lower left", fontsize=8.2)
    ax.text(0.99, 0.02, "scores clipped at -20", transform=ax.transAxes, ha="right", fontsize=8, color="#666666")
    fig.tight_layout()
    fig.canvas.draw()
    base_chart = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    bbox = ax.get_window_extent()
    chart_height, chart_width = base_chart.shape[:2]
    plot_left = int(round(bbox.x0))
    plot_right = int(round(bbox.x1))
    plot_top = int(round(chart_height - bbox.y1))
    plot_bottom = int(round(chart_height - bbox.y0))
    plt.close(fig)

    video_name = f"episode_{episode:03d}_success.mp4"
    source_frames = load_video_frames(root / video_name)
    raw_path = output / f"12_full_task_episode{episode:03d}_success_raw.mp4"
    final_path = output / f"12_full_task_episode{episode:03d}_success.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), OUTPUT_FPS, (1440, 480))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {raw_path}")

    phase_ranges: list[tuple[str, int, int]] = []
    for phase_index, phase in enumerate(PHASES):
        start = 1 if phase_index == 0 else (trigger_steps[PHASES[phase_index - 1]] or 0) + 1
        end = (trigger_steps.get(phase) or total_steps) if phase != "phase_4" else total_steps
        phase_ranges.append((phase, start, end))

    for total_step in range(1, total_steps + 1):
        current_phase = next(phase for phase, start, end in phase_ranges if start <= total_step <= end)
        scene = cv2.resize(frame_for_step(source_frames, total_step), (640, 480), interpolation=cv2.INTER_AREA)
        cv2.rectangle(scene, (0, 0), (640, 34), (0, 0, 0), -1)
        cv2.putText(
            scene,
            f"Episode {episode:03d} | {current_phase} | t={total_step / CONTROL_HZ:.1f} s",
            (12, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        chart = base_chart.copy()
        cursor_x = plot_left + int(round((total_step / total_steps) * (plot_right - plot_left)))
        cv2.line(chart, (cursor_x, plot_top), (cursor_x, plot_bottom), (20, 20, 20), 2, cv2.LINE_AA)
        if current_phase == "phase_4":
            status = "phase_4: terminal policy | no further routing decision"
        else:
            nearest = rows_by_step.get(total_step)
            if nearest is None:
                group = by_phase[current_phase]
                nearest = min(group, key=lambda row: abs(int(row["total_step"]) - total_step))
            symbolic_now = symbolic_steps[current_phase] is not None and total_step >= symbolic_steps[current_phase]
            status = (
                f"{current_phase} | best={nearest['selected_phase']}  "
                f"score={float(nearest['best_transition_log_score']):.2f}  "
                f"margin={float(nearest['margin']):.2f}  streak={nearest['confirmation_streak']}  "
                f"symbolic={symbolic_now}  decision={nearest['decision']}"
            )
        status_top = plot_top + 4
        cv2.rectangle(chart, (plot_left + 3, status_top), (plot_right - 3, status_top + 23), (255, 255, 255), -1)
        cv2.putText(chart, status, (plot_left + 8, status_top + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (25, 25, 25), 1, cv2.LINE_AA)
        writer.write(np.hstack([scene, chart]))
    writer.release()
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final_path),
        ],
        check=True,
    )
    raw_path.unlink()


def make_media(no_prior: Path, output: Path) -> None:
    rows_a, symbolic_a, trigger_a = trace_context(no_prior, 11, "phase_1")
    if symbolic_a is None or trigger_a is None:
        raise RuntimeError("Episode 11 does not contain the expected symbolic and switch events")
    clip_a_start = trigger_a - 80
    clip_a_end = trigger_a + 20
    make_synchronized_clip(
        no_prior,
        output,
        episode=11,
        phase="phase_1",
        video_name="episode_011_success.mp4",
        clip_start=clip_a_start,
        clip_end=clip_a_end,
        y_limits=(-8.0, 0.4),
        stem="08_clipA_episode11_phase1_to_phase2",
        title="Critical successful handoff: phase 1 → phase 2",
    )
    make_filmstrip(
        no_prior,
        output,
        episode=11,
        phase="phase_1",
        video_name="episode_011_success.mp4",
        clip_start=clip_a_start,
        clip_end=clip_a_end,
        sample_steps=[trigger_a - 60, symbolic_a, trigger_a, trigger_a + 10],
        sample_labels=["mid phase", "symbolic complete", "switch", "after switch"],
        y_limits=(-8.0, 0.4),
        stem="09_filmstripA_episode11_phase1_to_phase2",
        title="Filmstrip A — successful but low-margin phase 1 → phase 2 handoff",
    )

    episode_rows = [row for row in read_csv(no_prior / "episodes.csv") if int(row["episode"]) == 17][0]
    timeout_step = int(episode_rows["total_steps"])
    clip_b_start = timeout_step - 100
    clip_b_end = timeout_step
    make_synchronized_clip(
        no_prior,
        output,
        episode=17,
        phase="phase_1",
        video_name="episode_017_fail.mp4",
        clip_start=clip_b_start,
        clip_end=clip_b_end,
        y_limits=(-20.0, 0.5),
        stem="10_clipB_episode17_phase1_timeout",
        title="Policy deadlock: decoder correctly stays below threshold",
        timeout_step=timeout_step,
    )
    make_filmstrip(
        no_prior,
        output,
        episode=17,
        phase="phase_1",
        video_name="episode_017_fail.mp4",
        clip_start=clip_b_start,
        clip_end=clip_b_end,
        sample_steps=[timeout_step - 95, timeout_step - 65, timeout_step - 35, timeout_step - 5],
        sample_labels=["idle deadlock", "still idle", "no progress", "before timeout"],
        y_limits=(-20.0, 0.5),
        stem="11_filmstripB_episode17_phase1_timeout",
        title="Filmstrip B — phase 1 policy timeout with no false decoder transition",
        timeout_step=timeout_step,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-prior", type=Path, required=True)
    parser.add_argument("--no-prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-success-episode", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_style()
    if args.full_success_episode is not None:
        make_full_success_clip(args.no_prior, args.output, args.full_success_episode)
        print(f"Full successful episode written to {args.output}")
        return
    plot_decoder_delay(args.with_prior, args.no_prior, args.output)
    plot_end_start_space(args.no_prior, args.output)
    replay_margin(args.no_prior, args.output)
    make_media(args.no_prior, args.output)
    print(f"Presentation media written to {args.output}")


if __name__ == "__main__":
    main()
