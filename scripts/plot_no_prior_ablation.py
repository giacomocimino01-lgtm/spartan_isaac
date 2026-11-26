#!/usr/bin/env python3
"""Create presentation-ready plots for the transition-prior ablation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PHASES = [f"phase_{index}" for index in range(5)]
FROM_PHASES = PHASES[:-1]
LABELS = ["0→1", "1→2", "2→3", "3→4"]
COLORS = ["#2878B5", "#D95319", "#2A9D8F", "#7B2CBF", "#6C757D"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=260, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.alpha": 0.25,
        }
    )


def plot_outcomes(with_prior: Path, no_prior: Path, output: Path) -> None:
    datasets = [read_csv(with_prior / "episodes.csv"), read_csv(no_prior / "episodes.csv")]
    categories = ["Complete", "Timeout phase 0", "Timeout phase 1", "Timeout phase 3", "Routing error"]

    def counts(rows: list[dict[str, str]]) -> list[int]:
        reasons = Counter(
            "complete" if row["success"].lower() == "true" else row["failure_reason"]
            for row in rows
        )
        routing_errors = sum(value for key, value in reasons.items() if key.startswith("routing_error"))
        return [
            reasons["complete"],
            reasons["phase_timeout:phase_0"],
            reasons["phase_timeout:phase_1"],
            reasons["phase_timeout:phase_3"],
            routing_errors,
        ]

    values = np.asarray([counts(rows) for rows in datasets])
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(2)
    bottom = np.zeros(2)
    category_colors = ["#2A9D8F", "#B8C0CC", "#F4A261", "#8D99AE", "#D62828"]
    for index, (category, color) in enumerate(zip(categories, category_colors)):
        bars = ax.bar(x, values[:, index], bottom=bottom, label=category, color=color, width=0.58)
        for bar, value, base in zip(bars, values[:, index], bottom):
            if value:
                ax.text(bar.get_x() + bar.get_width() / 2, base + value / 2, str(value), ha="center", va="center", fontweight="bold")
        bottom += values[:, index]
    ax.set_xticks(x, ["Sparse transition prior", "No transition prior"])
    ax.set_ylabel("Episodes (count)")
    ax.set_ylim(0, 21)
    ax.set_yticks(range(0, 21, 5))
    ax.set_title("Closed-loop outcomes over 20 episodes")
    ax.grid(axis="y")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.text(0, 20.35, "17/20 complete", ha="center", fontweight="bold")
    ax.text(1, 20.35, "15/20 complete; 0 routing errors", ha="center", fontweight="bold")
    fig.subplots_adjust(bottom=0.24)
    save(fig, output, "01_closed_loop_outcomes")


def switch_rows(no_prior: Path) -> list[dict[str, str]]:
    return [row for row in read_csv(no_prior / "transition_score_trace.csv") if row["decision"] == "switch"]


def plot_margins(rows: list[dict[str, str]], output: Path) -> None:
    grouped = [[float(row["margin"]) for row in rows if row["phase"] == phase] for phase in FROM_PHASES]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    positions = np.arange(1, 5)
    bp = ax.boxplot(grouped, positions=positions, widths=0.5, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.23)
        patch.set_edgecolor(color)
    for median in bp["medians"]:
        median.set_color("#222222")
        median.set_linewidth(2)
    rng = np.random.default_rng(7)
    for position, values, color in zip(positions, grouped, COLORS):
        jitter = rng.uniform(-0.12, 0.12, len(values))
        ax.scatter(position + jitter, values, color=color, s=35, alpha=0.86, zorder=3)
        ax.text(position, max(values) + 0.25, f"mean {np.mean(values):.2f}\nmin {np.min(values):.2f}", ha="center", fontsize=10)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(positions, LABELS)
    ax.set_xlabel("Expected phase transition")
    ax.set_ylabel("Winner − runner-up transition log-score")
    ax.set_title("Decision margins without transition prior")
    ax.set_ylim(-0.15, 6.15)
    ax.grid(axis="y")
    ax.annotate(
        "Critical ambiguity: phase 0 is always the runner-up",
        xy=(2, min(grouped[1])),
        xytext=(2.45, 1.55),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
        fontsize=10,
    )
    save(fig, output, "02_no_prior_decision_margins")


def plot_heatmap(rows: list[dict[str, str]], output: Path) -> None:
    accum: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        scores = json.loads(row["transition_log_scores"])
        for candidate, score in scores.items():
            accum[(row["phase"], candidate)].append(float(score))
    matrix = np.full((4, 5), np.nan)
    for i, source in enumerate(FROM_PHASES):
        for j, candidate in enumerate(PHASES):
            values = accum.get((source, candidate), [])
            if values:
                matrix[i, j] = np.mean(values)
    finite = matrix[np.isfinite(matrix)]
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    image = ax.imshow(matrix, cmap="viridis", vmin=max(-8.0, finite.min()), vmax=0.0, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                text_color = "white" if value < -3.2 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontweight="bold" if j == i + 1 else "normal")
            else:
                ax.text(j, i, "current", ha="center", va="center", color="#444444")
    ax.set_xticks(range(5), [f"candidate {i}" for i in range(5)])
    ax.set_yticks(range(4), [f"current {i}" for i in range(4)])
    ax.set_xlabel("Candidate next phase")
    ax.set_ylabel("Current phase")
    ax.set_title("Mean transition log-score at policy switches (no prior)")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Mean transition log-score (higher is better)")
    save(fig, output, "03_no_prior_candidate_score_heatmap")


def plot_critical_trace(no_prior: Path, output: Path, episode: int = 11) -> None:
    rows = [
        row
        for row in read_csv(no_prior / "transition_score_trace.csv")
        if int(row["episode"]) == episode and row["phase"] == "phase_1"
    ]
    if not rows:
        raise RuntimeError(f"No phase_1 trace found for episode {episode}")
    time = np.asarray([int(row["phase_step"]) / 10.0 for row in rows])
    scores = {phase: [] for phase in ["phase_0", "phase_2", "phase_3", "phase_4"]}
    for row in rows:
        values = json.loads(row["transition_log_scores"])
        for phase in scores:
            scores[phase].append(float(values[phase]))
    switch_index = next(index for index, row in enumerate(rows) if row["decision"] == "switch")
    switch_time = time[switch_index]

    fig, (ax, streak_ax) = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [3.2, 1]})
    for phase, color in zip(scores, COLORS):
        emphasis = phase in {"phase_0", "phase_2"}
        ax.plot(time, scores[phase], color=color, linewidth=2.2 if emphasis else 1.2, alpha=1.0 if emphasis else 0.55, label=phase)
    ax.axhline(-2.0, color="#111111", linestyle="--", linewidth=1.4, label=r"threshold $\theta_1=-2$")
    ax.axvline(switch_time, color="#D62828", linestyle=":", linewidth=2, label="policy switch")
    ax.set_ylabel("Transition log-score")
    ax.set_title(f"Critical no-prior handoff: episode {episode}, phase 1 → phase 2")
    ax.set_ylim(-8.0, 0.5)
    ax.grid(axis="both")
    ax.legend(ncol=3, loc="lower left", frameon=False)
    ax.annotate(
        f"switch margin = {float(rows[switch_index]['margin']):.2f}",
        xy=(switch_time, float(rows[switch_index]["best_transition_log_score"])),
        xytext=(max(time.min(), switch_time - 7.0), -0.55),
        arrowprops={"arrowstyle": "->", "color": "#D62828"},
        color="#D62828",
        fontweight="bold",
    )

    streak = np.asarray([int(row["confirmation_streak"]) for row in rows])
    streak_ax.step(time, streak, where="post", color="#7B2CBF", linewidth=2)
    streak_ax.axhline(5, color="#111111", linestyle="--", linewidth=1.2, label="required streak = 5")
    streak_ax.axvline(switch_time, color="#D62828", linestyle=":", linewidth=2)
    streak_ax.set_ylabel("Streak")
    streak_ax.set_xlabel("Time within phase 1 (s at 10 Hz)")
    streak_ax.set_xlim(max(time.min(), switch_time - 8.0), switch_time + 0.8)
    streak_ax.set_ylim(-0.2, 5.8)
    streak_ax.set_yticks(range(0, 6))
    streak_ax.grid(axis="y")
    streak_ax.legend(loc="upper left", frameon=False)
    save(fig, output, "04_episode11_phase1_score_trace")


def write_metrics(rows: list[dict[str, str]], with_prior: Path, no_prior: Path, output: Path) -> None:
    with (output / "metrics_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["configuration", "episodes", "successes", "routing_queries", "routing_correct"])
        for label, path in [("sparse_prior", with_prior), ("no_prior", no_prior)]:
            summary = json.loads((path / "summary.json").read_text())
            writer.writerow([label, summary["episodes"], summary["successes"], summary["router"]["queries"], summary["router"]["correct"]])
        writer.writerow([])
        writer.writerow(["transition", "n", "runner_up", "min_margin", "mean_margin", "max_margin"])
        for source, label in zip(FROM_PHASES, LABELS):
            group = [row for row in rows if row["phase"] == source]
            margins = [float(row["margin"]) for row in group]
            runner_up = Counter(row["runner_up_phase"] for row in group).most_common(1)[0][0]
            writer.writerow([label, len(group), runner_up, min(margins), np.mean(margins), max(margins)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-prior", type=Path, required=True)
    parser.add_argument("--no-prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    rows = switch_rows(args.no_prior)
    plot_outcomes(args.with_prior, args.no_prior, args.output)
    plot_margins(rows, args.output)
    plot_heatmap(rows, args.output)
    plot_critical_trace(args.no_prior, args.output)
    write_metrics(rows, args.with_prior, args.no_prior, args.output)
    print(f"Wrote plots to {args.output}")


if __name__ == "__main__":
    main()
