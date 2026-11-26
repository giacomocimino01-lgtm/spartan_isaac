#!/usr/bin/env python3
"""Compare true Phase-4 entry observations with the full-state training bank."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_demos", required=True)
    parser.add_argument("--evaluation_entries", required=True)
    parser.add_argument("--evaluation_episodes", required=True)
    parser.add_argument("--bank_evaluation", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def first_episode_observations(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        obs = np.asarray(data["obs"], dtype=np.float64)
        episode_ids = np.asarray(data["episode_ids"], dtype=np.int64)
    first = np.unique(episode_ids, return_index=True)[1]
    first.sort()
    return obs[first], episode_ids[first]


def read_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_obs, train_episode_ids = first_episode_observations(
        Path(args.training_demos)
    )
    with np.load(args.evaluation_entries, allow_pickle=False) as data:
        eval_obs_all = np.asarray(data["obs"], dtype=np.float64)
        eval_episode_all = np.asarray(data["episode"], dtype=np.int64)
        eval_phase_all = np.asarray(data["phase"], dtype=str)
    phase4 = eval_phase_all == "phase_4"
    eval_obs = eval_obs_all[phase4]
    eval_episode = eval_episode_all[phase4]

    with Path(args.evaluation_episodes).open(newline="", encoding="utf-8") as stream:
        episode_rows = list(csv.DictReader(stream))
    eval_success_by_episode = {
        int(row["episode"]): read_bool(row["success"]) for row in episode_rows
    }

    with Path(args.bank_evaluation).open(newline="", encoding="utf-8") as stream:
        bank_rows = list(csv.DictReader(stream))
    # The sequential evaluator starts episode N from bank row N.  On a
    # time-limit truncation Isaac resets before terminal metrics are emitted,
    # so the live index in ``info`` can already point at row N+1.  Episode
    # order is therefore the authoritative row identifier for this protocol.
    bank_success_by_index = {
        int(row["episode"]): read_bool(row["success"]) for row in bank_rows
    }
    reported_index_mismatches = [
        {
            "episode": int(row["episode"]),
            "reported_index": int(row["phase4_empirical_entry_index"]),
        }
        for row in bank_rows
        if row["phase4_empirical_entry_index"] != ""
        and int(row["phase4_empirical_entry_index"]) != int(row["episode"])
    ]

    if train_obs.ndim != 2 or eval_obs.ndim != 2:
        raise ValueError("Expected two-dimensional observation arrays")
    if train_obs.shape[1] != eval_obs.shape[1]:
        raise ValueError(
            f"Observation dimension mismatch: {train_obs.shape[1]} vs "
            f"{eval_obs.shape[1]}"
        )

    scale = train_obs.std(axis=0)
    active = scale > 1e-6
    if not bool(active.any()):
        raise ValueError("No varying observation dimensions in training entries")
    center = train_obs.mean(axis=0)
    train_z = (train_obs[:, active] - center[active]) / scale[active]
    eval_z = (eval_obs[:, active] - center[active]) / scale[active]

    train_pairwise = np.linalg.norm(
        train_z[:, None, :] - train_z[None, :, :], axis=2
    )
    np.fill_diagonal(train_pairwise, np.inf)
    train_loo = train_pairwise.min(axis=1)

    eval_pairwise = np.linalg.norm(
        eval_z[:, None, :] - train_z[None, :, :], axis=2
    )
    nearest_pos = eval_pairwise.argmin(axis=1)
    nearest_distance = eval_pairwise[
        np.arange(len(eval_pairwise)), nearest_pos
    ]
    nearest_episode_id = train_episode_ids[nearest_pos]

    detail_rows: list[dict[str, object]] = []
    for position, episode in enumerate(eval_episode.tolist()):
        nearest_id = int(nearest_episode_id[position])
        detail_rows.append(
            {
                "episode": int(episode),
                "success": eval_success_by_episode.get(int(episode), False),
                "nearest_training_entry": nearest_id,
                "standardized_distance": float(nearest_distance[position]),
                "nearest_entry_deterministic_success": bank_success_by_index.get(
                    nearest_id, ""
                ),
            }
        )

    detail_path = output_dir / "entry_nearest_neighbors.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    train_p95 = percentile(train_loo, 95)
    eval_success_mask = np.asarray(
        [eval_success_by_episode.get(int(ep), False) for ep in eval_episode],
        dtype=bool,
    )
    nearest_known = np.asarray(
        [int(idx) in bank_success_by_index for idx in nearest_episode_id],
        dtype=bool,
    )
    nearest_success = np.asarray(
        [bank_success_by_index.get(int(idx), False) for idx in nearest_episode_id],
        dtype=bool,
    )
    summary = {
        "observation_dimension": int(train_obs.shape[1]),
        "varying_dimensions": int(active.sum()),
        "training_entries": int(len(train_obs)),
        "bank_deterministic_evaluation": {
            "episodes": int(len(bank_rows)),
            "successes": int(sum(bank_success_by_index.values())),
            "success_rate": float(np.mean(list(bank_success_by_index.values()))),
            "terminal_live_index_mismatches": len(reported_index_mismatches),
        },
        "evaluation_entries": int(len(eval_obs)),
        "evaluation_successes": int(eval_success_mask.sum()),
        "training_leave_one_out_distance": {
            "median": percentile(train_loo, 50),
            "p90": percentile(train_loo, 90),
            "p95": train_p95,
            "max": float(train_loo.max()),
        },
        "evaluation_to_training_distance": {
            "median": percentile(nearest_distance, 50),
            "p90": percentile(nearest_distance, 90),
            "p95": percentile(nearest_distance, 95),
            "max": float(nearest_distance.max()),
            "fraction_above_training_p95": float(
                np.mean(nearest_distance > train_p95)
            ),
        },
        "distance_by_evaluation_outcome": {
            "success_median": percentile(
                nearest_distance[eval_success_mask], 50
            ),
            "failure_median": percentile(
                nearest_distance[~eval_success_mask], 50
            ),
        },
        "nearest_training_entry": {
            "known_outcome_count": int(nearest_known.sum()),
            "deterministic_success_fraction": (
                float(nearest_success[nearest_known].mean())
                if nearest_known.any()
                else float("nan")
            ),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
