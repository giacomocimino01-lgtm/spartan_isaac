# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions specific to the dVRK Peg-and-Ring environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ring_out_of_bounds(
    env: ManagerBasedRLEnv,
    ring_names: list[str],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_min: float,
) -> torch.Tensor:
    """Termination: True per ogni env in cui almeno un ring è uscito dal workspace locale.

    Le posizioni sono calcolate in frame locale rispetto all'origine di ogni env,
    così il controllo è identico e indipendente per tutti gli ambienti paralleli.

    Args:
        env: L'ambiente IsaacLab.
        ring_names: Lista dei nomi degli oggetti ring nella scena (es. ["ring_red", ...]).
        x_bounds: (x_min, x_max) in frame locale dell'env.
        y_bounds: (y_min, y_max) in frame locale dell'env.
        z_min: quota minima in frame locale (sotto = caduto dal tavolo).

    Returns:
        Tensor booleano di shape (num_envs,): True dove l'episodio deve terminare.
    """
    env_origins = env.scene.env_origins  # [num_envs, 3]
    out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for ring_name in ring_names:
        pos_w = env.scene[ring_name].data.root_pos_w  # [num_envs, 3] world frame
        pos_local = pos_w - env_origins               # coordinate locali per ogni env

        out |= (
            (pos_local[:, 0] < x_bounds[0]) | (pos_local[:, 0] > x_bounds[1])
            | (pos_local[:, 1] < y_bounds[0]) | (pos_local[:, 1] > y_bounds[1])
            | (pos_local[:, 2] < z_min)
        )

    return out
