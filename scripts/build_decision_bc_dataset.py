"""Extract semantic command transitions from a low-level BC dataset.

The legacy collector stored the same high-level action at every physics step.
For a fixed scripted trajectory, a semantic decision is the first row of an
episode or a row whose joint action differs from the preceding row.  This
utility validates the expected number of decisions per episode and creates a
train/validation split by complete episode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Low-level NPZ dataset")
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--valid_output", required=True)
    parser.add_argument(
        "--decisions_per_episode",
        type=int,
        default=16,
        help=(
            "Expected inferred decisions per episode. Use 0 to select the "
            "first row of each recorded (episode_id, decision_index) group."
        ),
    )
    parser.add_argument("--validation_episodes", type=int, default=32)
    parser.add_argument("--split_seed", type=int, default=20260921)
    parser.add_argument(
        "--ring_reach_repeat",
        type=int,
        default=1,
        help=(
            "Repeat Phase-2 reach-to-ring training rows this many times. "
            "Validation rows are never repeated."
        ),
    )
    return parser.parse_args()


def save_split(
    path: str,
    observations: np.ndarray,
    actions: np.ndarray,
    source_episode_ids: np.ndarray,
    decision_indices: np.ndarray,
    metadata: dict,
    split: str,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    split_metadata = dict(metadata)
    split_metadata.update(
        {
            "record_mode": "decision_only_derived",
            "split": split,
            "source_dataset": str(metadata.get("source_dataset", "")),
            "episodes": int(len(np.unique(source_episode_ids))),
            "transitions": int(len(observations)),
        }
    )
    np.savez(
        output,
        obs=observations.astype(np.float32, copy=False),
        actions=actions.astype(np.int64, copy=False),
        episode_ids=source_episode_ids.astype(np.int64, copy=False),
        decision_indices=decision_indices.astype(np.int64, copy=False),
        metadata=split_metadata,
    )
    print(
        f"[{split}] episodes={split_metadata['episodes']} "
        f"transitions={len(observations)} output={output}"
    )


def main() -> None:
    args = parse_args()
    if args.decisions_per_episode < 0:
        raise SystemExit("--decisions_per_episode must be non-negative")
    if args.validation_episodes <= 0:
        raise SystemExit("--validation_episodes must be positive")
    if args.ring_reach_repeat <= 0:
        raise SystemExit("--ring_reach_repeat must be positive")

    source = np.load(args.source, allow_pickle=True)
    observations = source["obs"]
    actions = source["actions"]
    if observations.ndim != 2 or actions.ndim != 2:
        raise SystemExit("Source observations/actions must both be rank-2 arrays")
    if len(observations) != len(actions) or len(actions) == 0:
        raise SystemExit("Source observations/actions must have equal non-zero length")

    if args.decisions_per_episode == 0:
        required = {"episode_ids", "decision_indices"}
        if not required.issubset(source.files):
            raise SystemExit(
                "Recorded decision extraction requires episode_ids and "
                "decision_indices"
            )
        recorded_episode_ids = np.asarray(source["episode_ids"], dtype=np.int64)
        recorded_decision_indices = np.asarray(
            source["decision_indices"], dtype=np.int64
        )
        first_in_group = np.ones(len(actions), dtype=bool)
        first_in_group[1:] = (
            (recorded_episode_ids[1:] != recorded_episode_ids[:-1])
            | (recorded_decision_indices[1:] != recorded_decision_indices[:-1])
        )
        transition_rows = np.flatnonzero(first_in_group)
        episode_ids = recorded_episode_ids[transition_rows]
        decision_indices = recorded_decision_indices[transition_rows]
        unique_episode_ids = np.unique(episode_ids)
        episode_count = len(unique_episode_ids)
    else:
        change_mask = np.ones(len(actions), dtype=bool)
        change_mask[1:] = np.any(actions[1:] != actions[:-1], axis=1)
        transition_rows = np.flatnonzero(change_mask)
        if len(transition_rows) % args.decisions_per_episode:
            raise SystemExit(
                "Semantic transition count is not divisible by decisions_per_episode: "
                f"{len(transition_rows)} % {args.decisions_per_episode}"
            )
        episode_count = len(transition_rows) // args.decisions_per_episode
        episode_ids = np.repeat(
            np.arange(episode_count), args.decisions_per_episode
        )
        decision_indices = np.tile(
            np.arange(args.decisions_per_episode), episode_count
        )
        unique_episode_ids = np.arange(episode_count)
    if args.validation_episodes >= episode_count:
        raise SystemExit("Validation split must leave at least one training episode")

    # A Phase-2 demonstration must contain reach, grasp, reach, release for
    # each ring. Validate the verb pattern independently for both arms so a
    # corrupted or unexpectedly interrupted trace cannot silently enter BC.
    selected_actions = actions[transition_rows]
    expected_verbs = np.tile(np.array([0, 1, 0, 2], dtype=np.int64), 4)
    if args.decisions_per_episode == len(expected_verbs):
        per_episode_actions = selected_actions.reshape(
            episode_count, args.decisions_per_episode, actions.shape[1]
        )
        active_verbs = np.where(
            per_episode_actions[:, :, 0] != 3,
            per_episode_actions[:, :, 0],
            per_episode_actions[:, :, 2],
        )
        bad_episodes = np.flatnonzero(np.any(active_verbs != expected_verbs, axis=1))
        if len(bad_episodes):
            raise SystemExit(
                "Unexpected Phase-2 command pattern in source episodes: "
                f"{bad_episodes[:10].tolist()}"
            )

    rng = np.random.default_rng(args.split_seed)
    shuffled = rng.permutation(unique_episode_ids)
    valid_episodes = np.sort(shuffled[: args.validation_episodes])
    train_episodes = np.sort(shuffled[args.validation_episodes :])
    train_mask = np.isin(episode_ids, train_episodes)
    valid_mask = np.isin(episode_ids, valid_episodes)

    metadata = dict(source["metadata"].item()) if "metadata" in source.files else {}
    metadata.update(
        {
            "source_dataset": str(Path(args.source).resolve()),
            "source_transitions": int(len(actions)),
            "semantic_transitions": int(len(transition_rows)),
            "decisions_per_episode": (
                "recorded" if args.decisions_per_episode == 0
                else int(args.decisions_per_episode)
            ),
            "split_seed": int(args.split_seed),
            "ring_reach_repeat": int(args.ring_reach_repeat),
        }
    )
    selected_observations = observations[transition_rows]

    train_observations = selected_observations[train_mask]
    train_actions = selected_actions[train_mask]
    train_episode_ids = episode_ids[train_mask]
    train_decision_indices = decision_indices[train_mask]
    if args.ring_reach_repeat > 1:
        # In the canonical Phase-2 trace, indices 0/4/8/12 are the only
        # decisions that choose which ring to approach.  These are precisely
        # the labels on which the unbalanced decision-only BC was inaccurate.
        ring_reach_mask = (train_decision_indices % 4) == 0
        repeat_indices = np.flatnonzero(ring_reach_mask)
        extra_indices = np.tile(repeat_indices, args.ring_reach_repeat - 1)
        train_observations = np.concatenate(
            [train_observations, train_observations[extra_indices]], axis=0
        )
        train_actions = np.concatenate(
            [train_actions, train_actions[extra_indices]], axis=0
        )
        train_episode_ids = np.concatenate(
            [train_episode_ids, train_episode_ids[extra_indices]], axis=0
        )
        train_decision_indices = np.concatenate(
            [train_decision_indices, train_decision_indices[extra_indices]], axis=0
        )
        metadata["base_train_transitions"] = int(train_mask.sum())
        metadata["ring_reach_train_transitions"] = int(
            ring_reach_mask.sum() * args.ring_reach_repeat
        )

    save_split(
        args.train_output,
        train_observations,
        train_actions,
        train_episode_ids,
        train_decision_indices,
        metadata,
        "train",
    )
    save_split(
        args.valid_output,
        selected_observations[valid_mask],
        selected_actions[valid_mask],
        episode_ids[valid_mask],
        decision_indices[valid_mask],
        metadata,
        "valid",
    )
    print(
        f"[source] low_level={len(actions)} semantic={len(transition_rows)} "
        f"episodes={episode_count} train={len(train_episodes)} "
        f"valid={len(valid_episodes)} ring_reach_repeat={args.ring_reach_repeat}"
    )


if __name__ == "__main__":
    main()
