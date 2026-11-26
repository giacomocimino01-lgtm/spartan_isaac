#!/usr/bin/env python3
"""Write a BC dataset copy after proving every label is legal in a new mask mode."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from m_dVrk.hrl.masked_policy import build_joint_action_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task_phase",
        choices=("phase_0", "phase_1", "phase_2", "phase_3", "phase_4"),
        required=True,
    )
    parser.add_argument("--action_mask_mode", choices=("guided", "goal_agnostic", "minimal"), required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = np.load(args.source, allow_pickle=True)
    if not {"obs", "actions"}.issubset(source.files):
        raise ValueError("Dataset must contain obs and actions")
    observations = np.asarray(source["obs"], dtype=np.float32)
    actions = np.asarray(source["actions"], dtype=np.int64)
    metadata = source["metadata"].item().copy() if "metadata" in source.files else {}
    arrays = {name: source[name] for name in source.files if name != "metadata"}

    action_table = torch.cartesian_prod(
        torch.arange(4), torch.arange(10), torch.arange(4), torch.arange(10)
    )
    flat_actions = ((actions[:, 0] * 10 + actions[:, 1]) * 4 + actions[:, 2]) * 10 + actions[:, 3]
    valid_count = 0
    for start in range(0, len(observations), args.batch_size):
        stop = min(start + args.batch_size, len(observations))
        mask = build_joint_action_mask(
            torch.from_numpy(observations[start:stop]),
            action_table,
            args.task_phase,
            args.action_mask_mode,
        )
        indices = torch.from_numpy(flat_actions[start:stop]).view(-1, 1)
        valid_count += int(mask.gather(1, indices).sum().item())
    if valid_count != len(actions):
        raise ValueError(
            f"Refusing to relabel: only {valid_count}/{len(actions)} labels are legal "
            f"under {args.action_mask_mode}"
        )

    old_mode = metadata.get("action_mask_mode")
    metadata["source_action_mask_mode"] = old_mode
    metadata["action_mask_mode"] = args.action_mask_mode
    metadata["mask_compatibility_verified"] = True
    metadata["mask_compatibility_rows"] = len(actions)
    metadata["source_dataset_before_mask_relabel"] = str(args.source.resolve())
    arrays["metadata"] = np.array(metadata, dtype=object)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(
        f"[mask-relabel] wrote {args.output} with {valid_count}/{len(actions)} "
        f"labels legal under {args.action_mask_mode} (source mode: {old_mode})"
    )


if __name__ == "__main__":
    main()
