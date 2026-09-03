"""Reward computation for the m_dVrk HRL policy.

All reward logic lives here so that training, collection, and evaluation
share identical reward signals.
"""

from __future__ import annotations

import torch

from m_dVrk.hrl.constants import INVALID_COMMAND_PENALTY, SUCCESS_BONUS


def compute_tcc_reward(
    new_embs: torch.Tensor,
    goal_embedding: torch.Tensor,
    invalid_command_counts: torch.Tensor,
    success_mask: torch.Tensor | None = None,
    penalty: float = INVALID_COMMAND_PENALTY,
    success_bonus: float = SUCCESS_BONUS,
) -> torch.Tensor:
    """Compute the per-step TCC reward.

    Current formulation::

        reward = -||emb - goal||^2 * 1e-3
                 - invalid_commands * penalty
                 + task_success * success_bonus

    Args:
        new_embs: ``(num_envs, emb_dim)`` current frame embeddings.
        goal_embedding: ``(emb_dim,)`` or ``(1, emb_dim)`` mean goal embedding.
        invalid_command_counts: ``(num_envs,)`` float tensor counting invalid
            commands issued this step.
        success_mask: Optional ``(num_envs,)`` boolean tensor. The bonus is
            applied only to environments whose logical task success has been
            confirmed after the settle period.
        penalty: Scalar penalty per invalid command.
        success_bonus: Scalar terminal bonus for logical task completion.

    Returns:
        Float tensor of shape ``(num_envs,)`` containing per-env rewards.
    """
    rew_distance_raw = torch.norm(new_embs - goal_embedding, dim=1) ** 2
    rewards = -rew_distance_raw * 1e-3
    rewards = rewards - invalid_command_counts * penalty
    if success_mask is not None:
        rewards = rewards + success_mask.to(dtype=rewards.dtype) * success_bonus
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
