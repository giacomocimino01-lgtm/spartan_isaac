#!/usr/bin/env python3
"""Split a decision-only BC dataset by complete, possibly variable episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--valid_output", required=True)
    parser.add_argument("--validation_episodes", type=int, required=True)
    parser.add_argument("--split_seed", type=int, default=20260923)
    return parser.parse_args()


def save(
    path: str,
    source_path: str,
    source,
    rows: np.ndarray,
    metadata: dict,
    split: str,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    episode_ids = source["episode_ids"][rows]
    split_metadata = dict(metadata)
    split_metadata.update(
        {
            "split": split,
            "source_dataset": str(Path(source_path).resolve()),
            "episodes": int(len(np.unique(episode_ids))),
            "transitions": int(rows.sum()),
        }
    )
    np.savez(
        output,
        obs=source["obs"][rows].astype(np.float32, copy=False),
        actions=source["actions"][rows].astype(np.int64, copy=False),
        episode_ids=episode_ids.astype(np.int64, copy=False),
        decision_indices=source["decision_indices"][rows].astype(np.int64, copy=False),
        metadata=split_metadata,
    )
    print(
        f"[{split}] episodes={split_metadata['episodes']} "
        f"transitions={split_metadata['transitions']} output={output}"
    )


def main() -> None:
    args = parse_args()
    source = np.load(args.source, allow_pickle=True)
    required = {"obs", "actions", "episode_ids", "decision_indices"}
    if not required.issubset(source.files):
        raise SystemExit(f"Source is missing: {sorted(required - set(source.files))}")
    episode_ids = np.asarray(source["episode_ids"], dtype=np.int64)
    unique = np.unique(episode_ids)
    if args.validation_episodes <= 0 or args.validation_episodes >= len(unique):
        raise SystemExit("Validation count must leave at least one training episode")
    rng = np.random.default_rng(args.split_seed)
    shuffled = rng.permutation(unique)
    valid_ids = shuffled[: args.validation_episodes]
    train_ids = shuffled[args.validation_episodes :]
    metadata = dict(source["metadata"].item()) if "metadata" in source.files else {}
    save(
        args.train_output,
        args.source,
        source,
        np.isin(episode_ids, train_ids),
        metadata,
        "train",
    )
    save(
        args.valid_output,
        args.source,
        source,
        np.isin(episode_ids, valid_ids),
        metadata,
        "valid",
    )


if __name__ == "__main__":
    main()
