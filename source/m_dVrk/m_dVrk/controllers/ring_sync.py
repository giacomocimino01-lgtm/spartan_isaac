"""Ring synchronization utilities for the m_dVrk simulation.

This module contains:
  - get_scene_entity_positions_w: resolves any scene entity to per-env world
    positions, handling both rigid-body and XForm assets.
  - get_scene_entity_position_w: single-env convenience wrapper.
  - sync_attached_and_frozen_rings: writes ring root states to Isaac Sim for
    attached (held) and frozen (placed) rings.

These functions are called by the SPARTAN state machine and the HRL wrapper.
"""

from __future__ import annotations

import torch


def get_scene_entity_positions_w(env, entity_name: str) -> torch.Tensor:
    """Resolve a scene entity to one world-space position per environment.

    Handles both ``RigidObject`` assets (with ``data.root_pos_w``) and
    ``XFormPrimView`` assets (with ``get_world_poses``), including the case
    where there is a single shared asset replicated across environments.

    Args:
        env: IsaacLab ``ManagerBasedRLEnv``.
        entity_name: Name of the scene entity (e.g. ``"ring_red"``).

    Returns:
        Float tensor of shape ``(num_envs, 3)`` containing world positions.

    Raises:
        RuntimeError: If the entity cannot be resolved to ``num_envs``
            world positions.
    """
    entity = env.scene[entity_name]
    origins = env.scene.env_origins

    if hasattr(entity, "data") and hasattr(entity.data, "root_pos_w"):
        pos_w = entity.data.root_pos_w
        if pos_w.shape[0] == env.num_envs:
            return pos_w.clone()
        if pos_w.shape[0] == 1:
            local_pos = pos_w[0].clone() - origins[0]
            return origins + local_pos

    if hasattr(entity, "get_world_poses"):
        view_count = getattr(entity, "count", None)
        if view_count == env.num_envs:
            try:
                pos_w, _ = entity.get_world_poses(indices=list(range(env.num_envs)))
                if pos_w.shape[0] == env.num_envs:
                    return pos_w.clone()
            except Exception:
                pass

        pos_w, _ = entity.get_world_poses()
        if pos_w.shape[0] == env.num_envs:
            return pos_w.clone()
        if pos_w.shape[0] == 1:
            local_pos = pos_w[0].clone() - origins[0]
            return origins + local_pos

    raise RuntimeError(
        f"Cannot resolve scene entity '{entity_name}' to "
        f"{env.num_envs} world positions."
    )


def get_scene_entity_position_w(
    env,
    entity_name: str,
    env_id: int,
) -> torch.Tensor:
    """Return the world-space position of *entity_name* for a single env.

    Args:
        env: IsaacLab environment.
        entity_name: Scene entity name.
        env_id: Index of the target environment.

    Returns:
        Float tensor of shape ``(3,)``.
    """
    return get_scene_entity_positions_w(env, entity_name)[env_id]


# Internal aliases used by the state machine (preserving its call convention)
_get_scene_entity_positions_w = get_scene_entity_positions_w
_get_scene_entity_position_w = get_scene_entity_position_w


def sync_attached_and_frozen_rings(
    env,
    sm,
    active_env_mask: torch.Tensor,
) -> None:
    """Write ring root states to Isaac Sim for attached and frozen rings.

    For each ring:
      - If attached to the right arm: snap its position to the right tip
        (with a small lateral offset to avoid collision).
      - If attached to the left arm: snap to the left tip.
      - If frozen (placed on a peg): restore its recorded frozen pose.

    Only environments where *active_env_mask* is ``True`` are updated.
    Isaac-done environments that have already been internally reset by
    IsaacLab are excluded via the mask.

    Args:
        env: IsaacLab ``ManagerBasedRLEnv``.
        sm: ``SPARTANStateMachine`` instance.
        active_env_mask: Bool tensor of shape ``(num_envs,)``.
    """
    from m_dVrk.hrl.constants import RING_NAMES

    robot_r = env.scene["robot_right"]
    robot_l = env.scene["robot_left"]
    tip_r = robot_r.data.body_pos_w[:, sm.get_tip_idx_r()]
    tip_l = robot_l.data.body_pos_w[:, sm.get_tip_idx_l()]
    snap_offset_r = torch.tensor([-0.005, 0.0, 0.0], device=env.device)
    snap_offset_l = torch.tensor([+0.005, 0.0, 0.0], device=env.device)
    snap_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

    for ring_name in RING_NAMES:
        ring = env.scene[ring_name]
        new_s = ring.data.root_state_w.clone()

        mask_r = torch.tensor(
            [sm.attached_target_r[i] == ring_name for i in range(env.num_envs)],
            dtype=torch.bool,
            device=env.device,
        ) & active_env_mask

        mask_l = torch.tensor(
            [sm.attached_target_l[i] == ring_name for i in range(env.num_envs)],
            dtype=torch.bool,
            device=env.device,
        ) & active_env_mask

        mask_frozen = sm.frozen_rings_mask[ring_name] & active_env_mask
        write_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        if mask_r.any():
            new_s[mask_r, 0:3] = tip_r[mask_r] + snap_offset_r
            new_s[mask_r, 3:7] = snap_quat
            new_s[mask_r, 7:13] = 0.0
            write_mask |= mask_r

        if mask_l.any():
            new_s[mask_l, 0:3] = tip_l[mask_l] + snap_offset_l
            new_s[mask_l, 3:7] = snap_quat
            new_s[mask_l, 7:13] = 0.0
            write_mask |= mask_l

        if mask_frozen.any():
            new_s[mask_frozen, 0:7] = sm.frozen_rings_pose[ring_name][mask_frozen]
            new_s[mask_frozen, 7:13] = 0.0
            write_mask |= mask_frozen

        if write_mask.any():
            env_ids_to_write = write_mask.nonzero(as_tuple=False).flatten()
            ring.write_root_state_to_sim(new_s[env_ids_to_write], env_ids=env_ids_to_write)
