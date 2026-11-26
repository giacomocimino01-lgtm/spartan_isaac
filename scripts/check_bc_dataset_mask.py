#!/usr/bin/env python3
"""Fail when a BC label is illegal under its observation-derived action mask."""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch

from m_dVrk.hrl.masked_policy import build_joint_action_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
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
    data = np.load(args.dataset, allow_pickle=True)
    observations = np.asarray(data["obs"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.int64)
    action_table = torch.cartesian_prod(
        torch.arange(4), torch.arange(10), torch.arange(4), torch.arange(10)
    )
    flat = ((actions[:, 0] * 10 + actions[:, 1]) * 4 + actions[:, 2]) * 10 + actions[:, 3]
    valid_labels: list[bool] = []
    for start in range(0, len(observations), args.batch_size):
        stop = start + args.batch_size
        mask = build_joint_action_mask(
            torch.from_numpy(observations[start:stop]),
            action_table,
            args.task_phase,
            args.action_mask_mode,
        )
        indices = torch.from_numpy(flat[start:stop]).view(-1, 1)
        valid_labels.extend(mask.gather(1, indices).squeeze(1).tolist())

    invalid = np.flatnonzero(~np.asarray(valid_labels, dtype=bool))
    print(
        f"[mask-check] rows={len(actions)} valid={len(actions)-len(invalid)} "
        f"invalid={len(invalid)}"
    )
    if len(invalid):
        counts = Counter(map(tuple, actions[invalid]))
        for action, count in counts.most_common(20):
            print(f"  invalid_action={action} count={count}")
        raise SystemExit("BC dataset contains actions rejected by its mask")


if __name__ == "__main__":
    main()
