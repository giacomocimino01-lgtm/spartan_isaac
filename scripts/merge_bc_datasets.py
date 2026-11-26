#!/usr/bin/env python3
"""Merge compatible BC datasets, optionally repeating selected sources."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("PATH", "REPEAT"),
        required=True,
        help="Input .npz and positive integer repetition factor.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    sources: list[dict[str, object]] = []
    reference_metadata: dict[str, object] | None = None

    for raw_path, raw_repeat in args.dataset:
        path = Path(raw_path).expanduser().resolve()
        repeat = int(raw_repeat)
        if repeat <= 0:
            raise ValueError(f"Repeat must be positive for {path}")
        with np.load(path, allow_pickle=True) as data:
            obs = np.asarray(data["obs"], dtype=np.float32)
            acts = np.asarray(data["actions"], dtype=np.int64)
            metadata = dict(data["metadata"].item())

        if obs.ndim != 2 or acts.ndim != 2 or len(obs) != len(acts):
            raise ValueError(f"Invalid observation/action shapes in {path}")
        if acts.shape[1] != 4:
            raise ValueError(f"Expected four action columns in {path}")

        if reference_metadata is None:
            reference_metadata = metadata
        else:
            for key in ("obs_dim", "task_phase", "action_mask_mode"):
                if metadata.get(key) != reference_metadata.get(key):
                    raise ValueError(
                        f"Incompatible {key} in {path}: "
                        f"{metadata.get(key)!r} != {reference_metadata.get(key)!r}"
                    )

        observations.extend([obs] * repeat)
        actions.extend([acts] * repeat)
        sources.append(
            {
                "path": str(path),
                "repeat": repeat,
                "transitions": len(obs),
            }
        )

    merged_obs = np.concatenate(observations, axis=0)
    merged_actions = np.concatenate(actions, axis=0)
    metadata = dict(reference_metadata or {})
    metadata.update(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "merged_sources": sources,
            "merged_transition_count": int(len(merged_obs)),
        }
    )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, obs=merged_obs, actions=merged_actions, metadata=metadata)
    print(
        json.dumps(
            {
                "output": str(output),
                "observations": list(merged_obs.shape),
                "actions": list(merged_actions.shape),
                "sources": sources,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
