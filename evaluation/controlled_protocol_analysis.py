#!/usr/bin/env python3
"""Aggregate the three-phase, three-seed controlled PPO protocol.

The script consumes deterministic evaluation CSVs plus the per-episode CSVs
written during training. It produces machine-readable summaries, a concise
Markdown report and (when matplotlib is available) learning-curve figures.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


PHASES = ("phase_0", "phase_1", "phase_2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation_root", required=True, type=Path)
    parser.add_argument("--log_root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--suffix_prefix", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--curve_bins", type=int, default=20)
    return parser.parse_args()


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def summarize_eval(phase: str, seed: int, rows: list[dict[str, str]]) -> dict[str, object]:
    successes = [row for row in rows if truth(row.get("success", False))]
    steps = [number(row, "episode_steps") for row in rows]
    total_steps = sum(steps)
    bimanual = [number(row, "bimanual_steps") for row in rows]
    left_active = [number(row, "left_active_steps") for row in rows]
    right_active = [number(row, "right_active_steps") for row in rows]
    low, high = wilson(len(successes), len(rows))
    return {
        "phase": phase,
        "seed": seed,
        "episodes": len(rows),
        "successes": len(successes),
        "success_rate": len(successes) / len(rows) if rows else float("nan"),
        "success_ci95_low": low,
        "success_ci95_high": high,
        "mean_episode_steps": mean(steps) if steps else float("nan"),
        "mean_success_steps": (
            mean(number(row, "episode_steps") for row in successes)
            if successes
            else float("nan")
        ),
        "bimanual_step_fraction": sum(bimanual) / total_steps if total_steps else float("nan"),
        "episodes_with_bimanuality": sum(value > 0 for value in bimanual),
        "left_active_step_fraction": sum(left_active) / total_steps if total_steps else float("nan"),
        "right_active_step_fraction": sum(right_active) / total_steps if total_steps else float("nan"),
        "mean_joint_action_changes": mean(
            number(row, "joint_action_changes") for row in rows
        ) if rows else float("nan"),
        "mean_free_arm_step_slots": mean(
            number(row, "free_arm_step_slots") for row in rows
        ) if rows else float("nan"),
        "mean_invalid_commands": mean(
            number(row, "invalid_commands") for row in rows
        ) if rows else float("nan"),
        "mean_terminal_distance": mean(
            number(row, "terminal_distance", float("nan")) for row in rows
        ) if rows else float("nan"),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def training_curve(rows: list[dict[str, str]], total: int, bins: int) -> list[dict[str, float]]:
    width = total / bins
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        step = min(total - 1, max(0, int(number(row, "total_timesteps"))))
        grouped[min(bins - 1, int(step / width))].append(row)
    curve = []
    for index in range(bins):
        batch = grouped.get(index, [])
        lengths = [number(row, "episode_length") for row in batch]
        denominator = sum(lengths)
        curve.append(
            {
                "timestep": (index + 1) * width,
                "success_rate": (
                    mean(1.0 if truth(row.get("is_success", False)) else 0.0 for row in batch)
                    if batch else float("nan")
                ),
                "bimanual_fraction": (
                    sum(number(row, "bimanual_steps") for row in batch) / denominator
                    if denominator else float("nan")
                ),
            }
        )
    return curve


def aggregate(values: list[float]) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan"), float("nan")
    return mean(finite), pstdev(finite) if len(finite) > 1 else 0.0


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_summaries: list[dict[str, object]] = []
    curves: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(dict)

    for phase in PHASES:
        for seed in args.seeds:
            eval_csv = args.evaluation_root / phase / f"seed_{seed}" / "episodes.csv"
            suffix = f"{args.suffix_prefix}_seed{seed}"
            train_csv = (
                args.log_root
                / f"random_sb3_log_sim_{phase}_{suffix}"
                / "episode_progress.csv"
            )
            if not eval_csv.exists():
                raise SystemExit(f"Missing evaluation CSV: {eval_csv}")
            if not train_csv.exists():
                raise SystemExit(f"Missing training CSV: {train_csv}")
            run_summaries.append(summarize_eval(phase, seed, read_rows(eval_csv)))
            curves[phase][seed] = training_curve(
                read_rows(train_csv), args.total_timesteps, args.curve_bins
            )

    write_csv(args.output_dir / "summary_by_run.csv", run_summaries)

    phase_summaries: list[dict[str, object]] = []
    aggregate_keys = (
        "success_rate",
        "mean_success_steps",
        "bimanual_step_fraction",
        "episodes_with_bimanuality",
        "left_active_step_fraction",
        "right_active_step_fraction",
        "mean_joint_action_changes",
        "mean_invalid_commands",
        "mean_terminal_distance",
    )
    for phase in PHASES:
        phase_runs = [row for row in run_summaries if row["phase"] == phase]
        summary: dict[str, object] = {"phase": phase, "training_seeds": len(phase_runs)}
        for key in aggregate_keys:
            avg, sd = aggregate([float(row[key]) for row in phase_runs])
            summary[f"{key}_mean"] = avg
            summary[f"{key}_sd"] = sd
        phase_summaries.append(summary)
    write_csv(args.output_dir / "summary_by_phase.csv", phase_summaries)

    report = [
        "# Controlled serial-minimal PPO protocol",
        "",
        f"Budget: {args.total_timesteps:,} aggregate environment transitions per phase/seed.",
        f"Training seeds: {', '.join(map(str, args.seeds))}.",
        "Deterministic evaluation: 100 episodes per phase/seed.",
        "",
        "| Phase | Success (mean ± SD) | Successful steps | Bimanual step fraction | Episodes with bimanuality |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in phase_summaries:
        report.append(
            f"| {row['phase']} | "
            f"{100 * float(row['success_rate_mean']):.1f}% ± {100 * float(row['success_rate_sd']):.1f}% | "
            f"{float(row['mean_success_steps_mean']):.1f} ± {float(row['mean_success_steps_sd']):.1f} | "
            f"{100 * float(row['bimanual_step_fraction_mean']):.2f}% ± {100 * float(row['bimanual_step_fraction_sd']):.2f}% | "
            f"{float(row['episodes_with_bimanuality_mean']):.1f} ± {float(row['episodes_with_bimanuality_sd']):.1f} |"
        )
    report.extend(
        [
            "",
            "Success uncertainty inside each 100-episode run is reported as a Wilson 95% interval in `summary_by_run.csv`.",
            "Across-seed values above are mean ± population SD over the three independent PPO seeds.",
        ]
    )
    (args.output_dir / "controlled_protocol_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; CSV and Markdown summaries were generated")
        return

    for metric, ylabel, filename in (
        ("success_rate", "Training episode success rate", "learning_curves_success.png"),
        ("bimanual_fraction", "Bimanual active-step fraction", "learning_curves_bimanuality.png"),
    ):
        figure, axis = plt.subplots(figsize=(9, 5.5))
        for phase in PHASES:
            xs = [point["timestep"] for point in next(iter(curves[phase].values()))]
            means, sds = [], []
            for index in range(args.curve_bins):
                avg, sd = aggregate(
                    [curves[phase][seed][index][metric] for seed in args.seeds]
                )
                means.append(avg)
                sds.append(sd)
            axis.plot(xs, means, label=phase)
            lower = [max(0.0, avg - sd) for avg, sd in zip(means, sds)]
            upper = [min(1.0, avg + sd) for avg, sd in zip(means, sds)]
            axis.fill_between(xs, lower, upper, alpha=0.18)
        axis.set_xlabel("Aggregate environment transitions")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.output_dir / filename, dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    main()
