#!/usr/bin/env python3
"""Evaluate PPO checkpoints deterministically on the same randomized reset stream."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import signal
import shutil
import subprocess
import time
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths or glob patterns.")
    parser.add_argument("--isaaclab_path", default=os.environ.get("IsaacLab_PATH"), help="IsaacLab root directory.")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Common reset-stream seed used for every checkpoint.")
    parser.add_argument(
        "--task_phase",
        choices=["phase_0", "phase_1", "phase_2", "phase_3", "phase_4"],
        default="phase_0",
    )
    parser.add_argument(
        "--phase1_initial_state",
        choices=(
            "stack_green",
            "red3_blue1",
            "red1_blue3",
            "red2_blue1_green1",
            "red1_blue2_green1",
            "recovery_mixed",
            "mixed_curriculum",
        ),
        default="stack_green",
    )
    parser.add_argument(
        "--phase2_initial_state",
        choices=("red2_blue2", "center2", "center3", "recovery_mixed"),
        default="red2_blue2",
    )
    parser.add_argument("--tcc_checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", default="artifacts/evaluations/deterministic")
    parser.add_argument("--best_name", default=None, help="Filename stem for the copied winning checkpoint.")
    parser.add_argument("--timeout_seconds", type=int, default=3600)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fixed_rings", action="store_true", help="Use the fixed layout instead of randomized resets.")
    parser.add_argument(
        "--ring_randomization_scale",
        type=float,
        default=1.0,
        help="Randomized reset scale passed to eval_run.py (default: full range).",
    )
    parser.add_argument("--action_masking", action="store_true")
    parser.add_argument(
        "--phase_entry_park_reset",
        action="store_true",
        help="Evaluate from the full-task parked phase-entry joint pose.",
    )
    parser.add_argument(
        "--phase_entry_park_neighborhood_probability",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--phase_entry_park_neighborhood_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument("--phase_entry_empirical_bank", default=None)
    parser.add_argument(
        "--phase_entry_empirical_probability", type=float, default=1.0
    )
    parser.add_argument(
        "--phase_entry_empirical_sample_mode",
        choices=("random", "sequential"),
        default="sequential",
    )
    parser.add_argument(
        "--action_mask_mode",
        choices=("guided", "goal_agnostic", "minimal"),
        default="guided",
    )
    return parser.parse_args()


def resolve_checkpoints(patterns: list[str]) -> list[Path]:
    resolved: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        resolved.update(Path(match).resolve() for match in matches if match.endswith(".zip"))
    return sorted(resolved, key=lambda path: path.name)


def mean(values: list[float], default: float = math.nan) -> float:
    return sum(values) / len(values) if values else default


def summarize(
    rows: list[dict[str, str]], checkpoint: Path, task_phase: str
) -> dict[str, object]:
    successes = [row["success"].lower() == "true" for row in rows]
    steps = [int(row["episode_steps"]) for row in rows]
    success_steps = [step for step, success in zip(steps, successes, strict=True) if success]
    count_columns = {
        "phase_0": ("green_peg_ring_count",),
        "phase_1": ("red_peg_ring_count", "blue_peg_ring_count"),
        "phase_2": ("board_center_ring_count",),
        "phase_3": (
            "red_peg_ring_count",
            "yellow_peg_ring_count",
            "green_peg_ring_count",
            "blue_peg_ring_count",
        ),
        "phase_4": (
            "outside_right_ring_count",
            "outside_left_ring_count",
        ),
    }
    columns = count_columns[task_phase]
    if rows and all(column in rows[0] for column in columns):
        completed_rings = [sum(int(row[column]) for column in columns) for row in rows]
    else:
        completed_rings = []
    bimanual_steps = [int(row["bimanual_steps"]) for row in rows]
    return {
        "checkpoint": str(checkpoint),
        "episodes": len(rows),
        "success_rate": mean([float(value) for value in successes], 0.0),
        "mean_success_steps": mean(success_steps),
        "mean_completed_rings": mean(completed_rings, 0.0),
        "mean_terminal_distance": mean([float(row["terminal_distance"]) for row in rows]),
        "mean_invalid_commands": mean([int(row["invalid_commands"]) for row in rows], 0.0),
        "bimanual_step_fraction": sum(bimanual_steps) / max(1, sum(steps)),
        "status": "ok",
    }


def ranking_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    success_steps = float(row["mean_success_steps"])
    if not math.isfinite(success_steps):
        success_steps = float("inf")
    return (
        -float(row["success_rate"]),
        -float(row["mean_completed_rings"]),
        success_steps,
        float(row["mean_invalid_commands"]),
    )


def find_vecnormalize(checkpoint: Path) -> Path | None:
    """Find SB3's checkpoint-specific VecNormalize companion."""
    candidates = [checkpoint.with_name(f"{checkpoint.stem}_vecnormalize.pkl")]
    step_match = re.fullmatch(r"(.+)_(\d+)_steps", checkpoint.stem)
    if step_match:
        prefix, steps = step_match.groups()
        candidates.insert(
            0,
            checkpoint.with_name(f"{prefix}_vecnormalize_{steps}_steps.pkl"),
        )
    return next((path for path in candidates if path.is_file()), None)


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(process: subprocess.Popen[bytes], grace_seconds: float = 10.0) -> None:
    """Terminate every launcher descendant, even if the launcher exits first."""
    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass

    deadline = time.monotonic() + grace_seconds
    while _process_group_exists(process_group_id) and time.monotonic() < deadline:
        time.sleep(0.25)

    if _process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_isolated(command: list[str], cwd: Path, log_path: Path, timeout_seconds: int) -> int:
    """Run one Isaac evaluation in its own process group and always reap it."""
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            raise


def main() -> None:
    args = parse_args()
    if not args.isaaclab_path:
        raise ValueError("Set IsaacLab_PATH or pass --isaaclab_path")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if not 0.0 <= args.ring_randomization_scale <= 1.0:
        raise ValueError("--ring_randomization_scale must be in [0, 1]")
    if not 0.0 <= args.phase_entry_empirical_probability <= 1.0:
        raise ValueError("--phase_entry_empirical_probability must be in [0, 1]")
    if args.phase_entry_empirical_bank and args.phase_entry_park_reset:
        raise ValueError("empirical-bank and parked resets are mutually exclusive")

    checkpoints = resolve_checkpoints(args.checkpoints)
    if not checkpoints:
        raise FileNotFoundError("No .zip checkpoints matched --checkpoints")

    repo_root = Path(__file__).resolve().parents[1]
    eval_script = repo_root / "scripts" / "eval_run.py"
    launcher = Path(args.isaaclab_path) / "isaaclab.sh"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        run_stem = f"{index:03d}_{checkpoint.stem}"
        episode_csv = output_dir / f"{run_stem}_episodes.csv"
        log_path = output_dir / f"{run_stem}.log"
        command = [
            str(launcher),
            "-p",
            str(eval_script),
            "--checkpoint",
            str(checkpoint),
            "--tcc_checkpoint",
            args.tcc_checkpoint,
            "--dataset_path",
            args.dataset_path,
            "--task_phase",
            args.task_phase,
            "--phase1_initial_state",
            args.phase1_initial_state,
            "--phase2_initial_state",
            args.phase2_initial_state,
            "--episodes",
            str(args.episodes),
            "--seed",
            str(args.seed),
            "--metrics_output",
            str(episode_csv),
            "--device",
            args.device,
            "--disable_video",
            "--quiet_steps",
            "--headless",
        ]
        if not args.fixed_rings:
            command.append("--randomize_rings")
            command.extend(
                ["--ring_randomization_scale", str(args.ring_randomization_scale)]
            )
        if args.action_masking:
            command.append("--action_masking")
            command.extend(["--action_mask_mode", args.action_mask_mode])
        if args.phase_entry_park_reset:
            command.append("--phase_entry_park_reset")
            command.extend(
                [
                    "--phase_entry_park_neighborhood_probability",
                    str(args.phase_entry_park_neighborhood_probability),
                    "--phase_entry_park_neighborhood_scale",
                    str(args.phase_entry_park_neighborhood_scale),
                ]
            )
        if args.phase_entry_empirical_bank:
            command.extend(
                [
                    "--phase_entry_empirical_bank",
                    args.phase_entry_empirical_bank,
                    "--phase_entry_empirical_probability",
                    str(args.phase_entry_empirical_probability),
                    "--phase_entry_empirical_sample_mode",
                    args.phase_entry_empirical_sample_mode,
                ]
            )

        print(f"[rank] ({index}/{len(checkpoints)}) {checkpoint.name}", flush=True)
        try:
            returncode = run_isolated(command, repo_root, log_path, args.timeout_seconds)
            if returncode != 0:
                summaries.append(
                    {"checkpoint": str(checkpoint), "episodes": 0, "status": f"exit_{returncode}"}
                )
                continue
            with episode_csv.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            summaries.append(summarize(rows, checkpoint, args.task_phase))
        except subprocess.TimeoutExpired:
            summaries.append({"checkpoint": str(checkpoint), "episodes": 0, "status": "timeout"})

    successful_runs = [row for row in summaries if row.get("status") == "ok"]
    successful_runs.sort(key=ranking_key)
    failed_runs = [row for row in summaries if row.get("status") != "ok"]
    ordered = successful_runs + failed_runs

    summary_csv = output_dir / "checkpoint_ranking.csv"
    fieldnames = sorted({key for row in ordered for key in row})
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)

    if not successful_runs:
        raise RuntimeError(f"No checkpoint completed evaluation; inspect logs in {output_dir}")

    best = successful_runs[0]
    best_checkpoint = Path(str(best["checkpoint"]))
    best_name = args.best_name or f"dvrk_ppo_best_deterministic_{args.task_phase}"
    best_copy = output_dir / f"{best_name}.zip"
    shutil.copy2(best_checkpoint, best_copy)
    vecnormalize = find_vecnormalize(best_checkpoint)
    if vecnormalize is not None:
        shutil.copy2(vecnormalize, output_dir / f"{best_name}_vecnormalize.pkl")

    best_manifest = output_dir / "best_deterministic.json"
    with best_manifest.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                **best,
                "copied_checkpoint": str(best_copy),
                "action_masking": args.action_masking,
                "action_mask_mode": args.action_mask_mode,
                "seed": args.seed,
                "randomized_rings": not args.fixed_rings,
                "ring_randomization_scale": (
                    args.ring_randomization_scale if not args.fixed_rings else 0.0
                ),
                "phase1_initial_state": args.phase1_initial_state,
                "phase2_initial_state": args.phase2_initial_state,
                "phase_entry_park_reset": args.phase_entry_park_reset,
                "phase_entry_park_neighborhood_probability": (
                    args.phase_entry_park_neighborhood_probability
                ),
                "phase_entry_park_neighborhood_scale": (
                    args.phase_entry_park_neighborhood_scale
                ),
                "phase_entry_empirical_bank": args.phase_entry_empirical_bank,
                "phase_entry_empirical_probability": (
                    args.phase_entry_empirical_probability
                ),
                "phase_entry_empirical_sample_mode": (
                    args.phase_entry_empirical_sample_mode
                ),
            },
            stream,
            indent=2,
        )

    print(f"[rank] Best deterministic checkpoint: {best_checkpoint}")
    print(f"[rank] Success rate: {float(best['success_rate']):.2%}")
    print(f"[rank] Ranking: {summary_csv}")
    print(f"[rank] Best copy: {best_copy}")


if __name__ == "__main__":
    main()
