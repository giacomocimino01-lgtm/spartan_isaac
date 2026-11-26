#!/usr/bin/env python3
"""Build presentation-ready TCC and PPO/BC training figures from TensorBoard."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PHASE_COLORS = {
    "Phase 0": "#0072B2",
    "Phase 1": "#E69F00",
    "Phase 2": "#009E73",
    "Phase 3": "#CC79A7",
    "Phase 4": "#D55E00",
}

TCC_RUNS = {
    "Phase 0": "random_sim_phase0_tcc",
    "Phase 1": "random_sim_phase1_tcc",
    "Phase 2": "random_sim_phase2_tcc",
    "Phase 3": "random_sim_phase3_tcc",
    "Phase 4": "random_sim_phase4_tcc",
}

RL_RUNS = {
    "Phase 0": "random_sb3_log_sim_phase_0_adaptive075_minimalmask_finetune_3m",
    "Phase 1": "random_sb3_log_sim_phase_1_native_minimal_recovery_mixed_replay_probe_300k",
    "Phase 2": "random_sb3_log_sim_phase_2_phase1_teacherterminal_replay_kl_ppo_probe400k",
    "Phase 3": "random_sb3_log_sim_phase_3_bimanual_minimal_bc80_replay_probe_400k",
    "Phase 4": "random_sb3_log_sim_phase_4_bimanual_minimal_bc80_replay_probe_400k",
}

# Deterministic, isolated, matched-protocol rankings for BC and the selected PPO.
BC_PPO_EVAL = {
    "Phase 2": (0.98, 0.98, 50),
    "Phase 3": (0.96, 0.96, 50),
    "Phase 4 canonical": (0.72, 0.98, 50),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xirl-runs-root", type=Path, required=True)
    parser.add_argument("--rl-logs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def scalar_series(event_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    merged: dict[int, tuple[float, float]] = {}
    for event_file in sorted(event_dir.glob("events.out.tfevents*")):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for event in accumulator.Scalars(tag):
            merged[int(event.step)] = (float(event.wall_time), float(event.value))
    if not merged:
        raise ValueError(f"Missing TensorBoard scalar {tag!r} below {event_dir}")
    ordered = sorted((step, value) for step, (_, value) in merged.items())
    return (
        np.asarray([step for step, _ in ordered], dtype=np.float64),
        np.asarray([value for _, value in ordered], dtype=np.float64),
    )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 2 or window <= 1:
        return values.copy()
    result = np.empty_like(values, dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = float(np.mean(values[start : index + 1]))
    return result


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def build_tcc_plot(root: Path, output_dir: Path) -> None:
    rows: list[dict[str, float | str]] = []
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    loss_axis, tau_axis = axes

    for phase, run_name in TCC_RUNS.items():
        event_dir = root / run_name / "tb"
        loss_steps, losses = scalar_series(event_dir, "pretrain/valid/total_loss")
        tau_steps, taus = scalar_series(event_dir, "downstream/valid/kendalls_tau")
        loss_progress = 100.0 * loss_steps / max(float(loss_steps[-1]), 1.0)
        tau_progress = 100.0 * tau_steps / max(float(tau_steps[-1]), 1.0)
        color = PHASE_COLORS[phase]
        loss_axis.plot(loss_progress, losses, marker="o", ms=3.5, lw=2, color=color, label=phase)
        tau_axis.plot(tau_progress, taus, marker="o", ms=3.5, lw=2, color=color, label=phase)
        for step, loss in zip(loss_steps, losses):
            rows.append({"phase": phase, "step": int(step), "metric": "valid_loss", "value": float(loss)})
        for step, tau in zip(tau_steps, taus):
            rows.append({"phase": phase, "step": int(step), "metric": "valid_kendall_tau", "value": float(tau)})

    loss_axis.set_title("TCC validation loss")
    loss_axis.set_xlabel("Training progress (%)")
    loss_axis.set_ylabel("Regression loss (log scale)")
    loss_axis.set_yscale("log")
    tau_axis.set_title("Temporal alignment on validation videos")
    tau_axis.set_xlabel("Training progress (%)")
    tau_axis.set_ylabel("Kendall's τ")
    tau_axis.set_ylim(-0.05, 1.0)
    for axis in axes:
        style_axis(axis)
    handles, labels = tau_axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Phase-specific TCC encoders",
        y=1.035,
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output_dir / "tcc_training_loss_and_alignment.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "tcc_training_loss_and_alignment.pdf", bbox_inches="tight")
    plt.close(fig)

    with (output_dir / "tcc_training_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["phase", "step", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def build_rl_plot(root: Path, output_dir: Path) -> None:
    rows: list[dict[str, float | str]] = []
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), constrained_layout=True)
    flat_axes = axes.ravel()

    for axis, (phase, run_name) in zip(flat_axes, RL_RUNS.items()):
        steps, success = scalar_series(root / run_name, "rollout/success_rate")
        smooth = moving_average(success, max(3, min(15, len(success) // 12)))
        color = PHASE_COLORS[phase]
        axis.plot(steps / 1e6, success, color=color, alpha=0.20, lw=0.8)
        axis.plot(steps / 1e6, smooth, color=color, lw=2.2)
        axis.set_title(phase + (" (canonical PPO)" if phase == "Phase 4" else ""))
        axis.set_xlabel("Environment steps (millions)")
        axis.set_ylabel("Training success rate")
        axis.set_ylim(-0.03, 1.03)
        style_axis(axis)
        for step, raw, smoothed in zip(steps, success, smooth):
            rows.append(
                {
                    "phase": phase,
                    "step": int(step),
                    "success_rate": float(raw),
                    "success_rate_smoothed": float(smoothed),
                }
            )

    flat_axes[-1].axis("off")
    fig.suptitle("PPO learning curves after BC initialization", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.01,
        "Phase 4 shows the canonical PPO run; the policy deployed in the no-prior full task is the teacher-terminal BC.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output_dir / "rl_training_success_all_phases.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "rl_training_success_all_phases.pdf", bbox_inches="tight")
    plt.close(fig)

    with (output_dir / "rl_training_success.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["phase", "step", "success_rate", "success_rate_smoothed"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_bc_ppo_plot(output_dir: Path) -> None:
    labels = list(BC_PPO_EVAL)
    bc_values = [BC_PPO_EVAL[label][0] for label in labels]
    ppo_values = [BC_PPO_EVAL[label][1] for label in labels]
    episodes = [BC_PPO_EVAL[label][2] for label in labels]
    positions = np.arange(len(labels))
    width = 0.34

    fig, axis = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    bc_bars = axis.bar(positions - width / 2, bc_values, width, label="BC", color="#56B4E9")
    ppo_bars = axis.bar(positions + width / 2, ppo_values, width, label="BC + PPO", color="#009E73")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1.10)
    axis.set_ylabel("Deterministic isolated success rate")
    axis.set_title("Matched evaluation: does PPO improve the BC policy?", pad=30)
    axis.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.055),
    )
    style_axis(axis)
    for bars in (bc_bars, ppo_bars):
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.018,
                f"{100 * bar.get_height():.0f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    for position, count in zip(positions, episodes):
        axis.text(position, 0.035, f"n={count}", ha="center", va="bottom", fontsize=9, color="#333333")
    fig.savefig(output_dir / "bc_vs_ppo_deterministic_success.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "bc_vs_ppo_deterministic_success.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    build_tcc_plot(args.xirl_runs_root, args.output_dir)
    build_rl_plot(args.rl_logs_root, args.output_dir)
    build_bc_ppo_plot(args.output_dir)


if __name__ == "__main__":
    main()
