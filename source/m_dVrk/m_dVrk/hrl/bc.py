"""Behavior Cloning (BC) utilities for the m_dVrk HRL policy.

This module contains:
  - DummyHRLWrapper: a lightweight VecEnv stub used by the BC pretraining
    script to define observation / action spaces without running Isaac Sim.
  - train_bc: the full BC training loop (cross-entropy on MultiDiscrete heads).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from m_dVrk.hrl.constants import VERB_MAP, TARGET_MAP


# ---------------------------------------------------------------------------
# Dummy wrapper (spaces only, no Isaac Sim)
# ---------------------------------------------------------------------------

class DummyHRLWrapper(VecEnv):
    """Lightweight VecEnv that only defines observation and action spaces.

    Used by the BC pretraining script to instantiate an SB3 PPO model
    without needing a running Isaac Sim environment. The ``obs_dim`` is
    read from the dataset so it always matches the collected data.

    Args:
        isaac_env: IsaacLab environment (used only for ``num_envs``).
        obs_dim: Observation dimension, inferred from the dataset.
    """

    def __init__(self, isaac_env, obs_dim: int) -> None:
        self.env = isaac_env
        self.num_envs = isaac_env.num_envs
        self.obs_dim = obs_dim

        act_space = spaces.MultiDiscrete(
            [len(VERB_MAP), len(TARGET_MAP), len(VERB_MAP), len(TARGET_MAP)]
        )
        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.render_mode = None
        super().__init__(self.num_envs, obs_space, act_space)

    def reset(self):
        return np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)

    def step_async(self, actions): pass

    def step_wait(self):
        return (
            np.zeros((self.num_envs, self.obs_dim), dtype=np.float32),
            np.zeros(self.num_envs),
            np.zeros(self.num_envs),
            [],
        )

    def close(self): pass

    def get_attr(self, attr_name, indices=None):
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None): pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


# ---------------------------------------------------------------------------
# BC training loop
# ---------------------------------------------------------------------------

def train_bc(
    model,
    obs_array: np.ndarray,
    act_array: np.ndarray,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
) -> None:
    """Train an SB3 PPO policy via Behavior Cloning (cross-entropy).

    Each of the 4 MultiDiscrete action heads is trained independently with
    cross-entropy loss. The function trains in-place (modifies ``model.policy``).

    Args:
        model: SB3 ``PPO`` model whose policy will be trained.
        obs_array: ``(N, obs_dim)`` float32 array of observations.
        act_array: ``(N, 4)`` int64 array of expert actions
            ``[verb_l, tgt_l, verb_r, tgt_r]``.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        device: PyTorch device string.

    Head sizes (must match the MultiDiscrete action space):
        verb_l  4  |  tgt_l  10  |  verb_r  4  |  tgt_r  10
    """
    obs_tensor = torch.tensor(obs_array, dtype=torch.float32)
    act_tensor = torch.tensor(act_array, dtype=torch.long)
    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy = model.policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Head sizes for MultiDiscrete([4, 10, 4, 10])
    head_sizes = [len(VERB_MAP), len(TARGET_MAP), len(VERB_MAP), len(TARGET_MAP)]

    print("[BC] Starting Behavior Cloning training...")
    policy.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = [0, 0, 0, 0]
        total_samples = 0

        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)

            features = policy.extract_features(batch_obs)
            latent_pi, _ = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi)

            logits_split = torch.split(logits, head_sizes, dim=-1)

            loss = sum(loss_fn(logits_split[i], batch_act[:, i]) for i in range(4))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = len(batch_obs)
            epoch_loss += loss.item() * n
            total_samples += n
            for i in range(4):
                correct[i] += (logits_split[i].argmax(dim=-1) == batch_act[:, i]).sum().item()

        avg_loss = epoch_loss / total_samples
        accs = [c / total_samples * 100 for c in correct]
        print(
            f"Epoch {epoch + 1:02d}/{epochs:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Acc L-Arm: [Verb: {accs[0]:.1f}%, Tgt: {accs[1]:.1f}%] | "
            f"Acc R-Arm: [Verb: {accs[2]:.1f}%, Tgt: {accs[3]:.1f}%]"
        )
