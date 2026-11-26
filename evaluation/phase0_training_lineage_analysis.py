#!/usr/bin/env python3
"""Generate a single report for the complete Phase-0 PPO training lineage."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Stage:
    name: str
    short_name: str
    log_dir: Path
    color: str
    initialization: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        default=REPO_ROOT / "artifacts",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--rolling_window", type=int, default=15)
    return parser.parse_args()


def load_stage(stage: Stage, offset_steps: float) -> tuple[pd.DataFrame, float]:
    path = stage.log_dir / "progress.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing training log: {path}")
    df = pd.read_csv(path)
    required = {
        "time/total_timesteps",
        "rollout/success_rate",
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

    df = df.dropna(subset=["time/total_timesteps"]).copy()
    df["local_steps"] = pd.to_numeric(
        df["time/total_timesteps"], errors="coerce"
    )
    df = df.dropna(subset=["local_steps"])
    df["global_steps"] = offset_steps + df["local_steps"]
    df["global_steps_m"] = df["global_steps"] / 1e6
    df["local_steps_m"] = df["local_steps"] / 1e6
    df["stage"] = stage.short_name
    df["stage_color"] = stage.color
    return df, offset_steps + float(df["local_steps"].max())


def rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=max(1, window), min_periods=1).mean()


def metric_series(df: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(df[column], errors="coerce")
    mask = values.notna()
    return df.loc[mask, "global_steps_m"], values.loc[mask]


def stage_backgrounds(ax, stages: list[Stage], frames: list[pd.DataFrame]) -> None:
    for idx, (stage, frame) in enumerate(zip(stages, frames, strict=True)):
        lo = float(frame["global_steps_m"].min())
        hi = float(frame["global_steps_m"].max())
        ax.axvspan(lo, hi, color=stage.color, alpha=0.045)
        ax.text(
            (lo + hi) / 2,
            0.965,
            stage.short_name,
            color=stage.color,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=2),
        )
        if idx:
            ax.axvline(lo, color="#555555", linestyle="--", linewidth=1.2)


def load_evaluation(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts_dir.resolve()
    logs = artifacts / "logs"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else artifacts / "reports" / "random_sb3_log_sim_phase_0_full_lineage"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        Stage(
            "Fixed-position PPO",
            "Fixed PPO",
            logs / "random_sb3_log_sim_phase_0_fixed_reachguard_success100_5m",
            "#2563eb",
            "BC phase0_bc_raw.zip",
        ),
        Stage(
            "Fixed-to-random curriculum PPO",
            "Curriculum PPO",
            logs / "random_sb3_log_sim_phase_0_curriculum_fixed_to_random_5m",
            "#d97706",
            "Final fixed PPO",
        ),
        Stage(
            "Adaptive random fine-tuning",
            "Adaptive fine-tune",
            logs / "random_sb3_log_sim_phase_0_adaptive_random_finetune_5m",
            "#059669",
            "Best curriculum PPO",
        ),
    ]

    frames: list[pd.DataFrame] = []
    offset = 0.0
    boundaries = [0.0]
    for stage in stages:
        frame, offset = load_stage(stage, offset)
        frames.append(frame)
        boundaries.append(offset)

    evaluations = [
        (
            "Fixed\nlayout",
            artifacts
            / "evaluations/phase0_fixed_reachguard_20ep/best_deterministic.json",
        ),
        (
            "Curriculum\nfull random",
            artifacts
            / "evaluations/phase0_curriculum_random_rank_20ep/best_deterministic.json",
        ),
        (
            "Final policy\nfull random",
            artifacts
            / "evaluations/phase0_adaptive_random_rank_50ep_retry/best_deterministic.json",
        ),
        (
            "Final policy\nscale 0.75",
            artifacts / "evaluations/phase0_scale075_50ep/best_deterministic.json",
        ),
        (
            "Final policy\nscale 0.80",
            artifacts / "evaluations/phase0_scale080_50ep/best_deterministic.json",
        ),
        (
            "Final policy\nscale 0.85",
            artifacts / "evaluations/phase0_scale085_50ep/best_deterministic.json",
        ),
    ]
    eval_rows = [(label, load_evaluation(path), path) for label, path in evaluations]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(18, 19), facecolor="#f8fafc")
    gs = fig.add_gridspec(4, 2, height_ratios=[1.18, 1.0, 1.0, 0.80], hspace=0.38, wspace=0.24)

    # Full-lineage training success with the randomization scale overlaid.
    ax_success = fig.add_subplot(gs[0, :])
    stage_backgrounds(ax_success, stages, frames)
    for stage, frame in zip(stages, frames, strict=True):
        x, y = metric_series(frame, "rollout/success_rate")
        ax_success.plot(x, y, color=stage.color, alpha=0.18, linewidth=0.8)
        ax_success.plot(
            x,
            rolling(y, args.rolling_window),
            color=stage.color,
            linewidth=2.4,
            label=f"{stage.short_name} success",
        )
    ax_success.set_ylim(-0.04, 1.04)
    ax_success.set_ylabel("Training success rate")
    ax_success.set_xlabel("Cumulative PPO timesteps (millions)")
    ax_success.set_title(
        "Phase 0 success across the complete PPO lineage",
        fontsize=14,
        fontweight="bold",
    )

    ax_scale = ax_success.twinx()
    for idx, (stage, frame) in enumerate(zip(stages, frames, strict=True)):
        if "curriculum/ring_randomization_scale" in frame.columns:
            scale = pd.to_numeric(
                frame["curriculum/ring_randomization_scale"], errors="coerce"
            )
        else:
            scale = pd.Series(np.zeros(len(frame)), index=frame.index)
        mask = scale.notna()
        ax_scale.plot(
            frame.loc[mask, "global_steps_m"],
            scale.loc[mask],
            color="#111827",
            linestyle=":" if idx else "--",
            linewidth=1.4,
            alpha=0.8,
            label="Randomization scale" if idx == 1 else None,
        )
    ax_scale.set_ylim(-0.04, 1.04)
    ax_scale.set_ylabel("Ring randomization scale", color="#111827")
    handles, labels = ax_success.get_legend_handles_labels()
    scale_handles, scale_labels = ax_scale.get_legend_handles_labels()
    ax_success.legend(handles + scale_handles, labels + scale_labels, loc="lower left", ncol=2)

    # Reward curve.
    ax_reward = fig.add_subplot(gs[1, 0])
    for stage, frame in zip(stages, frames, strict=True):
        x, y = metric_series(frame, "rollout/ep_rew_mean")
        ax_reward.plot(x, rolling(y, args.rolling_window), color=stage.color, linewidth=2)
    for boundary in np.asarray(boundaries[1:-1]) / 1e6:
        ax_reward.axvline(boundary, color="#6b7280", linestyle="--", linewidth=1)
    ax_reward.set_title("Episode reward mean (rolling)", fontweight="bold")
    ax_reward.set_xlabel("Cumulative PPO timesteps (millions)")
    ax_reward.set_ylabel("Mean episode reward")

    # Episode length curve.
    ax_length = fig.add_subplot(gs[1, 1])
    for stage, frame in zip(stages, frames, strict=True):
        x, y = metric_series(frame, "rollout/ep_len_mean")
        ax_length.plot(x, rolling(y, args.rolling_window), color=stage.color, linewidth=2)
    for boundary in np.asarray(boundaries[1:-1]) / 1e6:
        ax_length.axvline(boundary, color="#6b7280", linestyle="--", linewidth=1)
    ax_length.set_title("Episode length (rolling)", fontweight="bold")
    ax_length.set_xlabel("Cumulative PPO timesteps (millions)")
    ax_length.set_ylabel("Mean episode length")

    # Learning-rate and entropy schedules.
    ax_schedule = fig.add_subplot(gs[2, 0])
    for stage, frame in zip(stages, frames, strict=True):
        x, lr = metric_series(frame, "train/learning_rate")
        ax_schedule.plot(x, lr, color=stage.color, linewidth=2, label=stage.short_name)
    ax_schedule.set_yscale("log")
    ax_schedule.set_title("Learning-rate schedule", fontweight="bold")
    ax_schedule.set_xlabel("Cumulative PPO timesteps (millions)")
    ax_schedule.set_ylabel("Learning rate (log scale)")
    ax_schedule.legend(fontsize=9)

    # Zoom on the adaptive fine-tune, including its advancement criterion.
    ax_zoom = fig.add_subplot(gs[2, 1])
    fine = frames[-1]
    success = pd.to_numeric(fine["rollout/success_rate"], errors="coerce")
    mask = success.notna()
    ax_zoom.plot(
        fine.loc[mask, "local_steps_m"],
        rolling(success.loc[mask], args.rolling_window),
        color=stages[-1].color,
        linewidth=2.3,
        label="Training success",
    )
    if "curriculum/recent_success_rate" in fine.columns:
        recent = pd.to_numeric(fine["curriculum/recent_success_rate"], errors="coerce")
        recent_mask = recent.notna()
        ax_zoom.plot(
            fine.loc[recent_mask, "local_steps_m"],
            recent.loc[recent_mask],
            color="#7c3aed",
            linewidth=1.4,
            alpha=0.8,
            label="Curriculum window success",
        )
    ax_zoom.axhline(0.85, color="#dc2626", linestyle="--", linewidth=1.2, label="Advance threshold")
    ax_zoom.set_ylim(-0.04, 1.04)
    ax_zoom.set_title("Adaptive fine-tuning detail", fontweight="bold")
    ax_zoom.set_xlabel("Fine-tuning timesteps (millions)")
    ax_zoom.set_ylabel("Success rate")
    zoom_scale = ax_zoom.twinx()
    scale = pd.to_numeric(fine["curriculum/ring_randomization_scale"], errors="coerce")
    scale_mask = scale.notna()
    zoom_scale.plot(
        fine.loc[scale_mask, "local_steps_m"],
        scale.loc[scale_mask],
        color="#111827",
        linestyle=":",
        linewidth=1.8,
        label="Randomization scale",
    )
    zoom_scale.set_ylim(-0.04, 1.04)
    zoom_scale.set_ylabel("Randomization scale")
    handles, labels = ax_zoom.get_legend_handles_labels()
    scale_handles, scale_labels = zoom_scale.get_legend_handles_labels()
    ax_zoom.legend(handles + scale_handles, labels + scale_labels, loc="lower left", fontsize=8)

    # Deterministic evaluation results for the selected checkpoints.
    ax_eval = fig.add_subplot(gs[3, 0])
    valid_eval_rows = [(label, data) for label, data, _ in eval_rows if data]
    labels = [label for label, _ in valid_eval_rows]
    rates = [float(data["success_rate"]) for _, data in valid_eval_rows]
    episodes = [int(data["episodes"]) for _, data in valid_eval_rows]
    bars = ax_eval.bar(
        np.arange(len(rates)),
        rates,
        color=["#2563eb", "#d97706", "#059669", "#10b981", "#34d399", "#6ee7b7"][: len(rates)],
        edgecolor="#1f2937",
        linewidth=0.7,
    )
    ax_eval.set_ylim(0, 1.12)
    ax_eval.set_xticks(np.arange(len(labels)), labels, fontsize=8)
    ax_eval.set_ylabel("Deterministic success rate")
    ax_eval.set_title("Held-out deterministic evaluations", fontweight="bold")
    for bar, rate, count in zip(bars, rates, episodes, strict=True):
        ax_eval.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.025,
            f"{rate:.0%}\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Concise provenance and interpretation block.
    ax_summary = fig.add_subplot(gs[3, 1])
    ax_summary.axis("off")
    stage_lines = []
    for stage, frame in zip(stages, frames, strict=True):
        success_values = pd.to_numeric(frame["rollout/success_rate"], errors="coerce").dropna()
        scale_values = (
            pd.to_numeric(frame["curriculum/ring_randomization_scale"], errors="coerce").dropna()
            if "curriculum/ring_randomization_scale" in frame.columns
            else pd.Series([0.0])
        )
        stage_lines.append(
            f"{stage.short_name:<19} {frame['local_steps'].max()/1e6:>4.2f}M  "
            f"final train success={success_values.iloc[-1]:>5.1%}  "
            f"scale={scale_values.iloc[-1]:.2f}"
        )
    summary = (
        "CHECKPOINT LINEAGE\n"
        "BC phase0_bc_raw.zip\n"
        "  -> Fixed PPO (5M)\n"
        "  -> Fixed-to-random curriculum PPO (5M)\n"
        "  -> Adaptive random fine-tuning (5M)\n\n"
        + "\n".join(stage_lines)
        + "\n\nSELECTED INFERENCE CHECKPOINT\n"
        "phase0_scale075_best.zip\n"
        "98% deterministic success at randomization scale 0.75 (50 episodes)\n\n"
        "Important: success curves are measured under changing reset distributions.\n"
        "The held-out bars, not the final training window alone, determine deployment."
    )
    ax_summary.text(
        0.02,
        0.98,
        summary,
        ha="left",
        va="top",
        fontsize=9.5,
        family="monospace",
        linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#fef3c7", edgecolor="#d97706", alpha=0.75),
    )

    fig.suptitle(
        "Phase 0 — Complete PPO Training Analysis (15M steps after BC)",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.975,
        "Fixed policy → fixed-to-random curriculum → adaptive fine-tuning",
        ha="center",
        fontsize=11,
        color="#475569",
    )

    report_path = output_dir / "training_analysis_report.png"
    fig.savefig(report_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_ppo_timesteps": int(boundaries[-1]),
        "rolling_window": args.rolling_window,
        "stages": [
            {
                "name": stage.name,
                "initialization": stage.initialization,
                "log_dir": str(stage.log_dir),
                "timesteps": int(frame["local_steps"].max()),
                "final_training_success_rate": float(
                    pd.to_numeric(frame["rollout/success_rate"], errors="coerce").dropna().iloc[-1]
                ),
            }
            for stage, frame in zip(stages, frames, strict=True)
        ],
        "evaluations": {
            label.replace("\n", " "): data for label, data, _ in eval_rows if data
        },
        "selected_inference_checkpoint": str(
            artifacts / "evaluations/phase0_scale075_50ep/phase0_scale075_best.zip"
        ),
    }
    with (output_dir / "report_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary_payload, stream, indent=2)

    print(f"Report saved to: {report_path}")
    print(f"Summary saved to: {output_dir / 'report_summary.json'}")


if __name__ == "__main__":
    main()
