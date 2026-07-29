"""Offline diagnostics for a behavior-cloned SB3 PPO policy.

Run it with the same Python environment used for Stable-Baselines3, for example:

    ${IsaacLab_PATH}/isaaclab.sh -p scripts/check_bc_policy.py \
        --checkpoint modelli_salvati_sim/randomized_dvrk_ppo_pretrained_phase1.zip \
        --dataset_path dataset_supervisionato_randomized_phase1.npz

The script does not launch Isaac Sim.  It checks whether a BC checkpoint still
matches the dataset it was trained on, and how much observation normalization
changes its predictions.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

try:
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecEnv, VecNormalize
except ImportError as exc:
    raise SystemExit(
        "Missing Stable-Baselines3/Gymnasium imports. Run this with IsaacLab's "
        "Python environment, e.g. `${IsaacLab_PATH}/isaaclab.sh -p "
        "scripts/check_bc_policy.py ...`."
    ) from exc


VERB_MAP = {0: "reach", 1: "grasp", 2: "release", 3: "idle"}
TARGET_MAP = {
    0: "ring_red",
    1: "ring_yellow",
    2: "ring_green",
    3: "ring_blue",
    4: "peg_red",
    5: "peg_yellow",
    6: "peg_green",
    7: "peg_blue",
    8: "peg_gray",
    9: "None",
}

ACTION_HEADS = ("verb_l", "target_l", "verb_r", "target_r")
IDLE_VERB = 3
NONE_TARGET = 9
RING_TARGET_IDS = {0, 1, 2, 3}
PEG_TARGET_IDS = {4, 5, 6, 7, 8}


class SpaceOnlyVecEnv(VecEnv):
    """Minimal VecEnv used only to attach spaces to VecNormalize."""

    def __init__(self, obs_dim: int, nvec: np.ndarray):
        self.num_envs = 1
        self.render_mode = None
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        act_space = spaces.MultiDiscrete(nvec)
        super().__init__(self.num_envs, obs_space, act_space)

    def reset(self):
        return np.zeros((self.num_envs, self.observation_space.shape[0]), dtype=np.float32)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs = np.zeros((self.num_envs, self.observation_space.shape[0]), dtype=np.float32)
        return obs, np.zeros(self.num_envs), np.zeros(self.num_envs, dtype=bool), [{}]

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check behavior cloning policy quality offline.")
    parser.add_argument(
        "--checkpoint",
        default="modelli_salvati_sim/randomized_dvrk_ppo_pretrained_phase1.zip",
        help="Path to the SB3 PPO checkpoint to evaluate.",
    )
    parser.add_argument(
        "--dataset_path",
        default="dataset_supervisionato_randomized_phase1.npz",
        help="Path to the NPZ dataset containing `obs` and `actions` arrays.",
    )
    parser.add_argument(
        "--normalization",
        choices=("raw", "dataset", "vecnormalize"),
        default="raw",
        help="Observation preprocessing before passing samples to the policy.",
    )
    parser.add_argument(
        "--vecnormalize_path",
        default=None,
        help="VecNormalize pickle to use when --normalization=vecnormalize.",
    )
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0, help="Limit samples for a quick check. 0 means all.")
    parser.add_argument("--device", default="cuda", help="Torch device for policy inference.")
    return parser.parse_args()


def format_action(row: np.ndarray) -> str:
    return (
        f"L={VERB_MAP[int(row[0])]}->{TARGET_MAP[int(row[1])]} "
        f"R={VERB_MAP[int(row[2])]}->{TARGET_MAP[int(row[3])]}"
    )


def semantic_invalid_mask(actions: np.ndarray) -> np.ndarray:
    invalid = np.zeros(actions.shape[0], dtype=bool)
    for verb_col, target_col in ((0, 1), (2, 3)):
        verb = actions[:, verb_col]
        target = actions[:, target_col]
        invalid |= (verb == IDLE_VERB) & (target != NONE_TARGET)
        invalid |= (verb != IDLE_VERB) & (target == NONE_TARGET)
        invalid |= (verb == 1) & ~np.isin(target, list(RING_TARGET_IDS))
        invalid |= (verb == 2) & ~np.isin(target, list(PEG_TARGET_IDS))
    return invalid


def print_action_distribution(title: str, actions: np.ndarray, max_rows: int = 10) -> None:
    print(f"\n[{title}] top complete actions")
    counts = Counter(map(tuple, actions.tolist()))
    for row, count in counts.most_common(max_rows):
        arr = np.asarray(row, dtype=np.int64)
        pct = 100.0 * count / len(actions)
        print(f"  {arr.tolist()}  {count:7d}  {pct:6.2f}%  {format_action(arr)}")


def describe_dataset(obs: np.ndarray, actions: np.ndarray) -> None:
    print(f"[dataset] obs shape={obs.shape} dtype={obs.dtype}")
    print(f"[dataset] actions shape={actions.shape} dtype={actions.dtype}")
    print(f"[dataset] obs finite={np.isfinite(obs).all()} actions finite={np.isfinite(actions).all()}")
    print(
        "[dataset] obs mean/std/min/max="
        f"{obs.mean():.6f}/{obs.std():.6f}/{obs.min():.6f}/{obs.max():.6f}"
    )
    for name, start, end in (("embedding", 0, 96), ("aux", 96, 158), ("geom", 158, obs.shape[1])):
        if start < obs.shape[1]:
            chunk = obs[:, start:end]
            print(
                f"[dataset] {name:9s} mean/std/min/max="
                f"{chunk.mean():.6f}/{chunk.std():.6f}/{chunk.min():.6f}/{chunk.max():.6f}"
            )
    print(f"[dataset] label semantic-invalid rate={semantic_invalid_mask(actions).mean() * 100:.3f}%")


def normalize_observations(
    obs: np.ndarray,
    mode: str,
    vecnormalize_path: str | None,
    obs_dim: int,
    nvec: np.ndarray,
) -> np.ndarray:
    if mode == "raw":
        return obs.astype(np.float32, copy=True)

    if mode == "dataset":
        mean = obs.mean(axis=0, keepdims=True)
        std = obs.std(axis=0, keepdims=True)
        return ((obs - mean) / np.maximum(std, 1e-8)).astype(np.float32)

    if vecnormalize_path is None:
        raise ValueError("--vecnormalize_path is required with --normalization=vecnormalize")

    dummy_env = SpaceOnlyVecEnv(obs_dim=obs_dim, nvec=nvec)
    vec_env = VecNormalize.load(vecnormalize_path, dummy_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env.normalize_obs(obs.astype(np.float32, copy=True)).astype(np.float32)


def evaluate_policy(model: PPO, obs: np.ndarray, labels: np.ndarray, batch_size: int) -> tuple[np.ndarray, float]:
    policy = model.policy
    policy.eval()
    device = policy.device

    nvec = [int(x) for x in model.action_space.nvec]
    predictions = []
    total_loss = 0.0
    total_seen = 0

    with torch.no_grad():
        for start in range(0, len(obs), batch_size):
            end = min(start + batch_size, len(obs))
            obs_tensor = torch.as_tensor(obs[start:end], dtype=torch.float32, device=device)
            label_tensor = torch.as_tensor(labels[start:end], dtype=torch.long, device=device)

            features = policy.extract_features(obs_tensor)
            if isinstance(features, tuple):
                features = features[0]
            latent_pi, _ = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi)
            split_logits = torch.split(logits, nvec, dim=1)

            batch_pred = torch.stack([head.argmax(dim=1) for head in split_logits], dim=1)
            predictions.append(batch_pred.cpu().numpy())

            batch_loss = sum(F.cross_entropy(head, label_tensor[:, idx]) for idx, head in enumerate(split_logits))
            total_loss += float(batch_loss.item()) * (end - start)
            total_seen += end - start

    return np.concatenate(predictions, axis=0), total_loss / max(total_seen, 1)


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.dataset_path):
        raise SystemExit(f"Dataset not found: {args.dataset_path}")
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    data = np.load(args.dataset_path)
    obs = data["obs"].astype(np.float32)
    labels = data["actions"].astype(np.int64)
    if args.limit and args.limit > 0:
        obs = obs[: args.limit]
        labels = labels[: args.limit]

    describe_dataset(obs, labels)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"[model] loading checkpoint={args.checkpoint} device={device}")
    model = PPO.load(args.checkpoint, device=device)

    model_obs_shape = tuple(model.observation_space.shape)
    model_nvec = np.asarray(model.action_space.nvec, dtype=np.int64)
    print(f"[model] observation_space={model_obs_shape} action_nvec={model_nvec.tolist()}")

    if model_obs_shape != tuple(obs.shape[1:]):
        raise SystemExit(
            "Observation shape mismatch: "
            f"dataset has {tuple(obs.shape[1:])}, checkpoint expects {model_obs_shape}. "
            "Use the matching phase/dataset or regenerate the BC checkpoint with the current wrapper."
        )

    if tuple(model_nvec.tolist()) != (4, 10, 4, 10):
        raise SystemExit(f"Unexpected action space: {model_nvec.tolist()}")

    obs_for_policy = normalize_observations(
        obs=obs,
        mode=args.normalization,
        vecnormalize_path=args.vecnormalize_path,
        obs_dim=obs.shape[1],
        nvec=model_nvec,
    )
    print(f"[eval] normalization={args.normalization} samples={len(obs_for_policy)}")

    pred, mean_loss = evaluate_policy(model, obs_for_policy, labels, args.batch_size)
    exact = (pred == labels).all(axis=1).mean()
    per_head = (pred == labels).mean(axis=0)

    print(f"[eval] mean CE loss={mean_loss:.6f}")
    print(f"[eval] exact action accuracy={exact * 100:.2f}%")
    for name, acc in zip(ACTION_HEADS, per_head):
        print(f"[eval] {name:8s} accuracy={acc * 100:.2f}%")

    invalid_pred = semantic_invalid_mask(pred).mean() * 100.0
    changed = (pred != labels).any(axis=1).mean() * 100.0
    print(f"[eval] prediction semantic-invalid rate={invalid_pred:.3f}%")
    print(f"[eval] any-head mismatch rate={changed:.2f}%")

    print_action_distribution("labels", labels)
    print_action_distribution("predictions", pred)


if __name__ == "__main__":
    main()
