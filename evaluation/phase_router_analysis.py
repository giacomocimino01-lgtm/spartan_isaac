#!/usr/bin/env python3
"""Generate a compact diagnostic report for a phase-router evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation_dir",
        default="artifacts/evaluations/phase_router_phase012_first_frame",
        help="Directory containing metrics.json and frame_predictions.csv.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Report directory. Defaults to artifacts/reports/<evaluation-dir-name>."
        ),
    )
    return parser.parse_args()


def project_artifacts_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "configs" / "defaults.yaml"
    try:
        import yaml

        if cfg_path.exists():
            with cfg_path.open(encoding="utf-8") as stream:
                cfg = yaml.safe_load(stream) or {}
            configured = cfg.get("paths", {}).get("artifacts_dir")
            if configured:
                return Path(configured)
    except Exception:
        pass
    return Path("artifacts")


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def short_phase(name: str) -> str:
    return name.replace("phase_", "Phase ").replace("_", " ").title()


def compact_number(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    if abs(value) >= 10_000:
        return f"{value:.2e}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()
    evaluation_dir = Path(args.evaluation_dir).resolve()
    metrics_path = evaluation_dir / "metrics.json"
    predictions_path = evaluation_dir / "frame_predictions.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing router metrics: {metrics_path}")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing router predictions: {predictions_path}")

    with metrics_path.open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    df = pd.read_csv(predictions_path)
    if df.empty:
        raise ValueError("Router prediction table is empty")
    df["correct"] = as_bool(df["correct"])

    phases = list(metrics.get("phase_names", []))
    if not phases:
        phases = sorted(df["true_phase"].unique().tolist())
    distance_columns = [f"distance_{phase}" for phase in phases]
    missing = [column for column in distance_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing distance columns: {', '.join(missing)}")

    labels = [short_phase(phase) for phase in phases]
    confusion = np.zeros((len(phases), len(phases)), dtype=int)
    for row_index, true_phase in enumerate(phases):
        for column_index, predicted_phase in enumerate(phases):
            confusion[row_index, column_index] = int(
                (
                    (df["true_phase"] == true_phase)
                    & (df["predicted_phase"] == predicted_phase)
                ).sum()
            )
    row_totals = confusion.sum(axis=1, keepdims=True)
    confusion_fraction = np.divide(
        confusion,
        row_totals,
        out=np.zeros_like(confusion, dtype=float),
        where=row_totals != 0,
    )
    per_phase_accuracy = np.diag(confusion_fraction)
    correct_count = int(df["correct"].sum())
    total_count = len(df)
    overall_accuracy = correct_count / total_count

    colors = plt.cm.Set2(np.linspace(0.05, 0.85, max(3, len(phases))))
    fig = plt.figure(figsize=(16, 17))
    grid = fig.add_gridspec(4, 2, hspace=0.62, wspace=0.3)

    # Per-phase accuracy.
    ax_accuracy = fig.add_subplot(grid[0, 0])
    bars = ax_accuracy.bar(labels, per_phase_accuracy * 100, color=colors[: len(phases)])
    ax_accuracy.axhline(
        overall_accuracy * 100,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall: {overall_accuracy:.2%}",
    )
    for bar, value in zip(bars, per_phase_accuracy, strict=True):
        ax_accuracy.text(
            bar.get_x() + bar.get_width() / 2,
            value * 100 + 1.5,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax_accuracy.set_ylim(0, 108)
    ax_accuracy.set_ylabel("Accuracy (%)")
    ax_accuracy.set_title("Per-phase Routing Accuracy", fontweight="bold")
    ax_accuracy.grid(True, axis="y", alpha=0.3)
    ax_accuracy.legend(loc="lower right")

    # Row-normalized confusion matrix with raw counts.
    ax_confusion = fig.add_subplot(grid[0, 1])
    image = ax_confusion.imshow(confusion_fraction, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(phases)):
        for column in range(len(phases)):
            fraction = confusion_fraction[row, column]
            color = "white" if fraction >= 0.55 else "black"
            ax_confusion.text(
                column,
                row,
                f"{confusion[row, column]}\n({fraction:.0%})",
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )
    ax_confusion.set_xticks(range(len(phases)), labels)
    ax_confusion.set_yticks(range(len(phases)), labels)
    ax_confusion.set_xlabel("Predicted phase")
    ax_confusion.set_ylabel("True phase")
    ax_confusion.set_title("Confusion Matrix", fontweight="bold")
    fig.colorbar(image, ax=ax_confusion, fraction=0.046, pad=0.04, label="Row fraction")

    # Median raw distance of every encoder for every true phase.
    median_distances = np.zeros_like(confusion_fraction)
    for row, true_phase in enumerate(phases):
        phase_rows = df[df["true_phase"] == true_phase]
        for column, candidate in enumerate(phases):
            median_distances[row, column] = float(
                phase_rows[f"distance_{candidate}"].median()
            )
    log_medians = np.log10(1.0 + median_distances)
    ax_distances = fig.add_subplot(grid[1, 0])
    distance_image = ax_distances.imshow(log_medians, cmap="viridis")
    midpoint = (float(log_medians.min()) + float(log_medians.max())) / 2
    for row in range(len(phases)):
        for column in range(len(phases)):
            color = "white" if log_medians[row, column] <= midpoint else "black"
            ax_distances.text(
                column,
                row,
                compact_number(median_distances[row, column]),
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )
    ax_distances.set_xticks(range(len(phases)), labels)
    ax_distances.set_yticks(range(len(phases)), labels)
    ax_distances.set_xlabel("Encoder used for distance")
    ax_distances.set_ylabel("True phase")
    ax_distances.set_title("Median Raw Squared Distance", fontweight="bold")
    fig.colorbar(
        distance_image,
        ax=ax_distances,
        fraction=0.046,
        pad=0.04,
        label="log10(1 + distance)",
    )

    # Selector margin, split by outcome.
    ax_margin = fig.add_subplot(grid[1, 1])
    groups: list[np.ndarray] = []
    group_labels: list[str] = []
    group_colors: list[str] = []
    for is_correct, label, color in (
        (True, "Correct", "#4daf4a"),
        (False, "Incorrect", "#e41a1c"),
    ):
        values = df.loc[df["correct"] == is_correct, "margin"].to_numpy(dtype=float)
        if len(values):
            groups.append(np.log10(1.0 + np.maximum(values, 0.0)))
            group_labels.append(f"{label}\n(n={len(values)})")
            group_colors.append(color)
    boxplot = ax_margin.boxplot(groups, tick_labels=group_labels, patch_artist=True)
    for box, color in zip(boxplot["boxes"], group_colors, strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.65)
    ax_margin.set_ylabel("log10(1 + winner margin)")
    ax_margin.set_title("Decision Margin by Outcome", fontweight="bold")
    ax_margin.grid(True, axis="y", alpha=0.3)

    # All candidate distances, grouped by the true phase of each query.
    ordered_parts = []
    for phase in phases:
        part = df[df["true_phase"] == phase].sort_values(["video", "frame"])
        ordered_parts.append(part)
    ordered = pd.concat(ordered_parts, ignore_index=True)
    x = np.arange(len(ordered))
    ax_samples = fig.add_subplot(grid[2, :])
    for index, phase in enumerate(phases):
        values = np.log10(1.0 + ordered[f"distance_{phase}"].to_numpy(dtype=float))
        ax_samples.plot(
            x,
            values,
            marker="o",
            markersize=3,
            linewidth=1.1,
            alpha=0.8,
            color=colors[index],
            label=f"{labels[index]} encoder",
        )
    cursor = 0
    for index, phase in enumerate(phases):
        count = int((ordered["true_phase"] == phase).sum())
        if count:
            ax_samples.axvspan(
                cursor - 0.5,
                cursor + count - 0.5,
                color=colors[index],
                alpha=0.08,
            )
            ax_samples.text(
                cursor + (count - 1) / 2,
                0.98,
                f"True {labels[index]}",
                transform=ax_samples.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5),
            )
            if cursor:
                ax_samples.axvline(cursor - 0.5, color="grey", linestyle=":", alpha=0.7)
        cursor += count
    incorrect_indices = np.flatnonzero(~ordered["correct"].to_numpy())
    for index in incorrect_indices:
        ax_samples.axvline(index, color="red", linewidth=1.2, alpha=0.65)
    ax_samples.set_xlim(-0.5, len(ordered) - 0.5)
    ax_samples.set_xlabel("Validation query (grouped by true phase); red line = routing error")
    ax_samples.set_ylabel("log10(1 + squared distance)")
    ax_samples.set_title(
        "Encoder Distances for Every First-frame Query",
        fontweight="bold",
        pad=36,
    )
    ax_samples.grid(True, axis="y", alpha=0.3)
    ax_samples.legend(
        ncol=len(phases),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )

    # Text summary and the most important failure modes.
    ax_summary = fig.add_subplot(grid[3, :])
    ax_summary.axis("off")
    errors = df[~df["correct"]]
    confusion_pairs = Counter(
        zip(errors["true_phase"], errors["predicted_phase"], strict=False)
    )
    pair_text = ", ".join(
        f"{short_phase(true)} -> {short_phase(predicted)}: {count}"
        for (true, predicted), count in confusion_pairs.most_common()
    ) or "none"
    correct_margin = df.loc[df["correct"], "margin"].median()
    error_margin = errors["margin"].median() if not errors.empty else float("nan")
    prototype_stats = metrics.get("prototype_stats", {})
    prototype_text = ", ".join(
        f"{short_phase(phase)}={prototype_stats.get(phase, {}).get('videos', 'n/a')} videos"
        for phase in phases
    )
    per_phase_text = ", ".join(
        f"{label}={accuracy:.1%}"
        for label, accuracy in zip(labels, per_phase_accuracy, strict=True)
    )
    worst_index = int(np.argmin(per_phase_accuracy))
    summary_text = (
        "ENCODER SELECTOR SUMMARY\n"
        f"{'─' * 56}\n"
        f"Accuracy:       {correct_count}/{total_count} ({overall_accuracy:.2%})\n"
        f"Per phase:      {per_phase_text}\n"
        f"Errors:         {total_count - correct_count} — {pair_text}\n"
        f"Weakest phase:  {labels[worst_index]} ({per_phase_accuracy[worst_index]:.1%})\n"
        f"Median margin:  correct={compact_number(float(correct_margin))}, "
        f"incorrect={compact_number(float(error_margin))}\n\n"
        f"Method:         {metrics.get('method', 'n/a')}\n"
        f"Frames:         prototype={metrics.get('prototype_frames', 'n/a')}, "
        f"query={metrics.get('query_frames', 'n/a')}\n"
        f"Prototypes:     {prototype_text}\n\n"
        "DIAGNOSTIC\n"
        "Raw distances come from independently trained encoders and are not calibrated.\n"
        "The distance and margin panels show whether errors are isolated ambiguities or scale bias."
    )
    ax_summary.text(
        0.01,
        0.98,
        summary_text,
        transform=ax_summary.transAxes,
        fontsize=10,
        va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(
        "Phase Encoder Selector Analysis — "
        f"First-frame Routing — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (project_artifacts_dir() / "reports" / evaluation_dir.name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "phase_router_analysis_report.png"
    fig.savefig(report_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[router-report] Loaded: {evaluation_dir}")
    print(
        f"[router-report] Accuracy: {correct_count}/{total_count} "
        f"({overall_accuracy:.2%})"
    )
    print(f"[router-report] Saved: {report_path}")


if __name__ == "__main__":
    main()
