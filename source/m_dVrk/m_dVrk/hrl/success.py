"""Task success and progress counting for the m_dVrk HRL task.

These are pure functions (no side-effects) that can be called by training,
collection, evaluation, and offline diagnostics.
"""

from __future__ import annotations

import torch

from m_dVrk.hrl.constants import RING_NAMES, PEG_GREEN, PEG_RED, PEG_BLUE


def peg_ring_count(
    sm,
    peg_name: str,
    ring_names: list[str],
    num_envs: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Count how many rings are frozen on *peg_name* for each environment.

    Args:
        sm: ``SPARTANStateMachine`` instance.
        peg_name: Name of the peg to count rings on (e.g. ``"peg_green"``).
        ring_names: Ordered list of ring entity names.
        num_envs: Number of parallel environments.
        device: Tensor device.

    Returns:
        Long tensor of shape ``(num_envs,)`` with frozen ring counts.
    """
    counts = torch.zeros(num_envs, dtype=torch.long, device=device)
    for ring_name in ring_names:
        on_peg = torch.tensor(
            [sm.ring_support_peg[ring_name][i] == peg_name for i in range(num_envs)],
            dtype=torch.bool,
            device=device,
        )
        counts += (sm.frozen_rings_mask[ring_name] & on_peg).long()
    return counts


def task_success(
    sm,
    task_phase: str,
    ring_names: list[str],
    num_envs: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return a boolean tensor indicating per-environment task success.

    Phase 0 success: all 4 rings frozen on ``peg_green``.
    Phase 1 success: exactly 2 rings frozen on ``peg_red``, exactly 2 on
        ``peg_blue``, and all 4 rings frozen.

    Args:
        sm: ``SPARTANStateMachine`` instance.
        task_phase: ``"phase_0"`` or ``"phase_1"``.
        ring_names: Ordered list of ring entity names.
        num_envs: Number of parallel environments.
        device: Tensor device.

    Returns:
        Bool tensor of shape ``(num_envs,)``.
    """
    if task_phase == "phase_1":
        red_count  = peg_ring_count(sm, PEG_RED,  ring_names, num_envs, device)
        blue_count = peg_ring_count(sm, PEG_BLUE, ring_names, num_envs, device)
        all_frozen = torch.stack(
            [sm.frozen_rings_mask[name] for name in ring_names],
            dim=1,
        ).all(dim=1)
        return all_frozen & (red_count == 2) & (blue_count == 2)
    else:
        return peg_ring_count(sm, PEG_GREEN, ring_names, num_envs, device) == 4
