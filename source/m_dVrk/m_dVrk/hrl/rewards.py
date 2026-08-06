"""Reward computation for the m_dVrk HRL policy.

All reward logic lives here so that training, collection, and evaluation
share identical reward signals.
"""

from __future__ import annotations

import torch

from m_dVrk.hrl.constants import INVALID_COMMAND_PENALTY


def compute_tcc_reward(
    new_embs: torch.Tensor,
    goal_embedding: torch.Tensor,
    invalid_command_counts: torch.Tensor,
    penalty: float = INVALID_COMMAND_PENALTY,
) -> torch.Tensor:
    """Compute the per-step TCC reward.

    Current formulation::

        reward = -||emb - goal||^2 * 1e-3
                 - invalid_commands * penalty

    Args:
        new_embs: ``(num_envs, emb_dim)`` current frame embeddings.
        goal_embedding: ``(emb_dim,)`` or ``(1, emb_dim)`` mean goal embedding.
        invalid_command_counts: ``(num_envs,)`` float tensor counting invalid
            commands issued this step.
        penalty: Scalar penalty per invalid command.

    Returns:
        Float tensor of shape ``(num_envs,)`` containing per-env rewards.
    """
    rew_distance_raw = torch.norm(new_embs - goal_embedding, dim=1) ** 2
    rewards = -rew_distance_raw * 1e-3
    rewards = rewards - invalid_command_counts * penalty
    return rewards


def tcc_distance(
    new_embs: torch.Tensor,
    goal_embedding: torch.Tensor,
) -> torch.Tensor:
    """Return the raw squared TCC distance (for logging / info dicts).

    Args:
        new_embs: ``(num_envs, emb_dim)`` current frame embeddings.
        goal_embedding: ``(emb_dim,)`` or ``(1, emb_dim)`` mean goal embedding.

    Returns:
        Float tensor of shape ``(num_envs,)``.
    """
    return torch.norm(new_embs - goal_embedding, dim=1) ** 2
