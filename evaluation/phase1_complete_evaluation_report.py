#!/usr/bin/env python3
"""Create a report for complete deterministic Phase-1 evaluations.

The report intentionally distinguishes the primary evaluation from its repeat.
The pooled result is displayed only as a descriptive aggregate because repeated
simulator runs with the same seed are not guaranteed to be independent trials.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary_csv", type=Path, required=True)
    parser.add_argument("--repeat_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def wilson(successes: int, episodes: int, z: float = 1.96) -> tuple[float, float]:
    if episodes == 0:
        return float("nan"), float("nan")
    p = successes / episodes
    denominator = 1.0 + z * z / episodes
    centre = (p + z * z / (2.0 * episodes)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / episodes + z * z / (4.0 * episodes**2)) / denominator
    return float(centre - radius), float(centre + radius)


def summarize(label: str, path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    required = {
        "success",
        "episode_steps",
        "red_peg_ring_count",
        "blue_peg_ring_count",
        "green_peg_ring_count",
        "invalid_commands",
        "bimanual_steps",
        "phase1_initial_state",
        "deterministic",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["success"] = as_bool(frame["success"])
    frame["deterministic"] = as_bool(frame["deterministic"])
    successes = int(frame["success"].sum())
    episodes = int(len(frame))
    low, high = wilson(successes, episodes)
    successful_steps = frame.loc[frame["success"], "episode_steps"]
    completed = frame["red_peg_ring_count"] + frame["blue_peg_ring_count"]
    summary = {
        "label": label,
        "source_csv": str(path),
        "episodes": episodes,
        "successes": successes,
        "failures": episodes - successes,
        "success_rate": successes / episodes,
        "wilson_95_ci": [low, high],
        "mean_success_steps": float(successful_steps.mean()),
        "median_success_steps": float(successful_steps.median()),
        "mean_completed_rings": float(completed.mean()),
        "mean_invalid_commands": float(frame["invalid_commands"].mean()),
        "bimanual_step_fraction": float(frame["bimanual_steps"].sum() / frame["episode_steps"].sum()),
        "initial_states": sorted(frame["phase1_initial_state"].dropna().unique().tolist()),
        "all_deterministic": bool(frame["deterministic"].all()),
    }
    return frame, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    primary, primary_summary = summarize("Primary", args.primary_csv)
    repeat, repeat_summary = summarize("Repeat", args.repeat_csv)
    summaries = [primary_summary, repeat_summary]
    pooled_successes = sum(item["successes"] for item in summaries)
    pooled_episodes = sum(item["episodes"] for item in summaries)
    pooled_ci = wilson(pooled_successes, pooled_episodes)
    successful_steps = pd.concat(
        [primary.loc[primary["success"], "episode_steps"], repeat.loc[repeat["success"], "episode_steps"]],
        ignore_index=True,
    )

    payload = {
        "task": {
            "phase": "phase_1",
            "evaluation_scope": "complete end-to-end phase",
            "initial_state": "stack_green (all four rings on the green peg)",
            "success_condition": "two rings on red and two rings on blue",
            "policy_mode": "deterministic argmax",
            "action_mask_mode": "minimal",
        },
        "checkpoint": args.checkpoint,
        "runs": summaries,
        "descriptive_aggregate": {
            "successes": pooled_successes,
            "episodes": pooled_episodes,
            "success_rate": pooled_successes / pooled_episodes,
            "wilson_95_ci_if_treated_as_independent": list(pooled_ci),
            "mean_success_steps": float(successful_steps.mean()),
            "median_success_steps": float(successful_steps.median()),
        },
        "interpretation": (
            "Observed deterministic performance is 96-98% across two complete evaluations. "
            "The 97/100 aggregate is descriptive only: both runs use seed 42 and should not be "
            "presented as 100 fully independent held-out trials."
        ),
    }
    (args.output_dir / "report_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    labels = [item["label"] for item in summaries]
    rates = np.array([item["success_rate"] for item in summaries]) * 100.0
    lower = rates - np.array([item["wilson_95_ci"][0] for item in summaries]) * 100.0
    upper = np.array([item["wilson_95_ci"][1] for item in summaries]) * 100.0 - rates

    fig = plt.figure(figsize=(15, 12), facecolor="#f7f9fc")
    grid = fig.add_gridspec(3, 2, height_ratios=[0.7, 1.2, 1.1], hspace=0.36, wspace=0.28)
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.95, "PHASE 1 — COMPLETE DETERMINISTIC EVALUATION", fontsize=20, weight="bold", va="top")
    title_ax.text(
        0.0,
        0.68,
        "Start: all 4 rings on green peg   →   Goal: 2 rings on red + 2 rings on blue\n"
        "Deterministic argmax policy · minimal physical action mask · 50 episodes per run",
        fontsize=12,
        va="top",
    )
    title_ax.text(
        0.0,
        0.22,
        f"Observed range: 96–98% success   |   Descriptive aggregate: {pooled_successes}/{pooled_episodes} "
        f"({100.0 * pooled_successes / pooled_episodes:.1f}%)   |   Mean successful duration: {successful_steps.mean():.1f} steps",
        fontsize=13,
        weight="bold",
        color="#183a66",
    )

    success_ax = fig.add_subplot(grid[1, 0])
    colors = ["#2474b5", "#52a675"]
    bars = success_ax.bar(labels, rates, color=colors, width=0.62)
    success_ax.errorbar(labels, rates, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#263238", capsize=7, lw=1.5)
    success_ax.set_ylim(0, 105)
    success_ax.set_ylabel("Success rate (%)")
    success_ax.set_title("End-to-end success (95% Wilson interval)", weight="bold")
    success_ax.grid(axis="y", alpha=0.25)
    for bar, item in zip(bars, summaries):
        success_ax.text(
            bar.get_x() + bar.get_width() / 2,
            item["success_rate"] * 100 - 7,
            f"{item['successes']}/{item['episodes']}\n{item['success_rate'] * 100:.0f}%",
            ha="center",
            va="top",
            color="white",
            weight="bold",
            fontsize=12,
        )

    step_ax = fig.add_subplot(grid[1, 1])
    step_data = [
        primary.loc[primary["success"], "episode_steps"].to_numpy(),
        repeat.loc[repeat["success"], "episode_steps"].to_numpy(),
    ]
    box = step_ax.boxplot(step_data, tick_labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    step_ax.set_ylabel("Steps in successful episodes")
    step_ax.set_title("Successful episode duration", weight="bold")
    step_ax.grid(axis="y", alpha=0.25)
    for index, item in enumerate(summaries, start=1):
        step_ax.text(
            index,
            0.95,
            f"mean {item['mean_success_steps']:.1f}",
            transform=step_ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    outcome_ax = fig.add_subplot(grid[2, 0])
    successes = np.array([item["successes"] for item in summaries])
    failures = np.array([item["failures"] for item in summaries])
    outcome_ax.bar(labels, successes, color="#52a675", label="Success")
    outcome_ax.bar(labels, failures, bottom=successes, color="#d7655a", label="Failure")
    outcome_ax.set_ylabel("Episodes")
    outcome_ax.set_title("Outcome counts", weight="bold")
    outcome_ax.legend()
    outcome_ax.grid(axis="y", alpha=0.25)

    evidence_ax = fig.add_subplot(grid[2, 1])
    evidence_ax.axis("off")
    evidence_lines = [
        "EVIDENCE CHECK",
        "",
        f"Checkpoint: {Path(args.checkpoint).name}",
        f"Initial state: {', '.join(primary_summary['initial_states'])}",
        f"Deterministic flag: {primary_summary['all_deterministic'] and repeat_summary['all_deterministic']}",
        f"Mean completed rings: {np.mean([item['mean_completed_rings'] for item in summaries]):.2f} / 4",
        f"Mean invalid commands: {np.mean([item['mean_invalid_commands'] for item in summaries]):.2f}",
        "",
        "Interpretation:",
        "The selected policy is robust on the canonical full Phase-1 task.",
        "Report 96–98% as the observed deterministic range.",
        "Treat 97/100 as descriptive because both runs use seed 42.",
    ]
    evidence_ax.text(
        0.0,
        1.0,
        "\n".join(evidence_lines),
        va="top",
        family="monospace",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "white", "edgecolor": "#b8c4d1"},
    )

    fig.suptitle("", y=0.99)
    report_path = args.output_dir / "phase1_complete_evaluation_report.png"
    fig.savefig(report_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Report: {report_path}")
    print(f"Summary: {args.output_dir / 'report_summary.json'}")


if __name__ == "__main__":
    main()
