"""Pure observation-building functions for the m_dVrk HRL policy.

All observation logic lives here. Every script (training, collection,
evaluation, diagnostics) must use these functions instead of reimplementing
observation construction locally.

Dimension contract (single source of truth):
    emb_stack  : STACK_SIZE * EMB_DIM  =  3 * 32  =  96
    aux_obs    : AUX_DIM               =           62
    geom_obs   : GEOM_DIM              =           86
    total      : OBS_DIM               =          244
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from m_dVrk.hrl.constants import (
    VERB_MAP,
    TARGET_MAP,
    VERB_TO_ID,
    TARGET_TO_ID,
    IDLE_ACTION,
    AUX_DIM,
    GEOM_DIM,
    OBS_DIM,
)
from m_dVrk.hrl.constants import PEG_GREEN, PEG_RED, PEG_BLUE


# ---------------------------------------------------------------------------
# Manipulable rings mask
# ---------------------------------------------------------------------------

def get_manipulable_rings_mask(
    env,
    sm,
    ring_names: list[str],
    task_phase: str,
    num_envs: int,
) -> torch.Tensor:
    """Return a ``(num_envs, 4)`` float mask: 1.0 if the ring can be picked.

    A ring is considered manipulable if:
      - it is not currently held by either arm, AND
      - if it is on a peg: it is the topmost ring on that peg AND it has not
        already been placed on the task's final peg, AND
      - if it is free in the workspace: it is within reach (z > -0.1,
        xy-radius < 0.45 m).

    Args:
        env: IsaacLab ``ManagerBasedRLEnv``.
        sm: ``SPARTANStateMachine`` instance.
        ring_names: Ordered list of ring entity names (length 4).
        task_phase: ``"phase_0"`` or ``"phase_1"``.
        num_envs: Number of parallel environments.

    Returns:
        Float tensor of shape ``(num_envs, 4)`` on ``env.device``.
    """
    from m_dVrk.controllers.ring_sync import get_scene_entity_positions_w

    device = env.device
    manipulable_mask = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)

    for env_i in range(num_envs):
        for ring_idx, ring_name in enumerate(ring_names):
            # Already held — skip
            if (sm.attached_target_l[env_i] == ring_name
                    or sm.attached_target_r[env_i] == ring_name):
                continue

            current_peg = sm.ring_support_peg.get(ring_name, [None] * num_envs)[env_i]

            if current_peg is not None and current_peg != "None":
                # Check if already placed on the task's final peg
                placed_on_final_peg = (
                    (task_phase == "phase_0" and current_peg == PEG_GREEN)
                    or (task_phase == "phase_1" and current_peg in {PEG_RED, PEG_BLUE})
                )
                if sm.frozen_rings_mask[ring_name][env_i] and placed_on_final_peg:
                    continue

                # On a peg: manipulable only if it is the top ring
                top_rings = sm.get_top_rings_on_peg(env_i, current_peg, num_rings=1)
                top_ring = top_rings[0] if top_rings else None
                if top_ring == ring_name:
                    manipulable_mask[env_i, ring_idx] = 1.0
            else:
                if sm.frozen_rings_mask[ring_name][env_i]:
                    continue
                # Free in workspace: check geometric bounds
                ring_pos_w = get_scene_entity_positions_w(env, ring_name)[env_i]
                origin_w = env.scene.env_origins[env_i]
                rel_pos = ring_pos_w - origin_w

                in_workspace = (rel_pos[2] > -0.1) and (torch.norm(rel_pos[:2]) < 0.45)
                if in_workspace:
                    manipulable_mask[env_i, ring_idx] = 1.0

    return manipulable_mask


# ---------------------------------------------------------------------------
# Auxiliary observation (62 dims)
# ---------------------------------------------------------------------------

def _actions_to_onehot(actions_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a ``(num_envs, 4)`` action tensor to 28-dim one-hot encoding."""
    actions_tensor = actions_tensor.long()
    verb_l = F.one_hot(actions_tensor[:, 0], num_classes=len(VERB_MAP)).float()
    tgt_l  = F.one_hot(actions_tensor[:, 1], num_classes=len(TARGET_MAP)).float()
    verb_r = F.one_hot(actions_tensor[:, 2], num_classes=len(VERB_MAP)).float()
    tgt_r  = F.one_hot(actions_tensor[:, 3], num_classes=len(TARGET_MAP)).float()
    return torch.cat([verb_l, tgt_l, verb_r, tgt_r], dim=1)


def _current_triplets_to_action_tensor(
    sm,
    num_envs: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Encode the currently executing SPARTAN triplets as a ``(num_envs, 4)`` tensor."""
    out = IDLE_ACTION.to(device).unsqueeze(0).repeat(num_envs, 1)

    for i in range(num_envs):
        if sm.sub_state_l[i] != "IDLE" and sm.current_triplet_l[i] is not None:
            cmd_l = sm.current_triplet_l[i]
            out[i, 0] = VERB_TO_ID.get(cmd_l["verb"], VERB_TO_ID["idle"])
            out[i, 1] = TARGET_TO_ID.get(cmd_l["target"], TARGET_TO_ID["None"])

        if sm.sub_state_r[i] != "IDLE" and sm.current_triplet_r[i] is not None:
            cmd_r = sm.current_triplet_r[i]
            out[i, 2] = VERB_TO_ID.get(cmd_r["verb"], VERB_TO_ID["idle"])
            out[i, 3] = TARGET_TO_ID.get(cmd_r["target"], TARGET_TO_ID["None"])

    return out


def get_aux_obs(
    sm,
    prev_actions: torch.Tensor,
    last_override_flags: torch.Tensor,
    num_envs: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Build the 62-dim auxiliary observation vector.

    Layout::

        current active triplets      28   (one-hot of what SPARTAN executes)
        previous requested triplets  28   (one-hot of last PPO action)
        busy flags                    2
        override-busy flags           2
        gripper states                2
        total                        62

    Args:
        sm: ``SPARTANStateMachine`` instance.
        prev_actions: ``(num_envs, 4)`` tensor of previous (sanitized) actions.
        last_override_flags: ``(num_envs, 2)`` float tensor.
        num_envs: Number of parallel environments.
        device: Tensor device.

    Returns:
        Float tensor of shape ``(num_envs, 62)`` on *device*.
    """
    current_triplet_actions = _current_triplets_to_action_tensor(sm, num_envs, device)
    current_triplet_oh = _actions_to_onehot(current_triplet_actions)
    prev_action_oh = _actions_to_onehot(prev_actions)

    busy_l = torch.tensor(
        [s != "IDLE" for s in sm.sub_state_l],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)

    busy_r = torch.tensor(
        [s != "IDLE" for s in sm.sub_state_r],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)

    busy_flags = torch.cat([busy_l, busy_r], dim=1)

    gripper_states = torch.stack(
        [sm.current_gripper_state_l, sm.current_gripper_state_r],
        dim=1,
    )

    aux = torch.cat(
        [
            current_triplet_oh,   # 28
            prev_action_oh,       # 28
            busy_flags,           # 2
            last_override_flags,  # 2
            gripper_states,       # 2
        ],
        dim=1,
    )

    assert aux.shape[1] == AUX_DIM, (
        f"[observations] aux_obs dim mismatch: expected {AUX_DIM}, got {aux.shape[1]}"
    )
    return aux


# ---------------------------------------------------------------------------
# Geometric observation (86 dims)
# ---------------------------------------------------------------------------

def get_geom_obs(
    env,
    sm,
    ring_names: list[str],
    peg_names: list[str],
    tip_idx_r: int,
    tip_idx_l: int,
    cached_peg_pos_local: torch.Tensor,
    task_phase: str,
    num_envs: int,
) -> torch.Tensor:
    """Build the 86-dim geometric observation vector.

    Layout::

        tip_r local position             3
        tip_l local position             3
        ring local positions         4 * 3 = 12
        peg local positions          4 * 3 = 12
        right tip → each ring        4 * 3 = 12
        left  tip → each ring        4 * 3 = 12
        ring → matching peg          4 * 3 = 12
        attached_r flags             4
        attached_l flags             4
        frozen flags                 4
        peg inventory (normalised)   4
        manipulable flags            4
        total                       86

    Args:
        env: IsaacLab environment.
        sm: ``SPARTANStateMachine`` instance.
        ring_names: Ordered ring entity names (length 4).
        peg_names: Ordered peg entity names (length 4) for the HRL observation.
        tip_idx_r: Body index of the right PSM tool tip.
        tip_idx_l: Body index of the left PSM tool tip.
        cached_peg_pos_local: ``(num_envs, 4, 3)`` local peg positions
            (pre-cached at environment construction).
        task_phase: ``"phase_0"`` or ``"phase_1"``.
        num_envs: Number of parallel environments.

    Returns:
        Float tensor of shape ``(num_envs, 86)`` on ``env.device``.
    """
    from m_dVrk.controllers.ring_sync import get_scene_entity_positions_w

    device = env.device
    origins = env.scene.env_origins  # (N, 3)

    robot_r = env.scene["robot_right"]
    robot_l = env.scene["robot_left"]

    tip_r_w = robot_r.data.body_pos_w[:, tip_idx_r]
    tip_l_w = robot_l.data.body_pos_w[:, tip_idx_l]

    tip_r = tip_r_w - origins
    tip_l = tip_l_w - origins

    ring_pos_local = torch.stack(
        [get_scene_entity_positions_w(env, name) - origins for name in ring_names],
        dim=1,
    )  # (N, 4, 3)

    peg_pos_local = cached_peg_pos_local  # (N, 4, 3)

    rel_tip_r_to_rings = ring_pos_local - tip_r.unsqueeze(1)   # (N, 4, 3)
    rel_tip_l_to_rings = ring_pos_local - tip_l.unsqueeze(1)   # (N, 4, 3)
    rel_ring_to_matching_peg = peg_pos_local - ring_pos_local  # (N, 4, 3)

    attached_r = torch.zeros((num_envs, 4), device=device)
    attached_l = torch.zeros((num_envs, 4), device=device)
    for j, ring_name in enumerate(ring_names):
        for i in range(num_envs):
            attached_r[i, j] = 1.0 if sm.attached_target_r[i] == ring_name else 0.0
            attached_l[i, j] = 1.0 if sm.attached_target_l[i] == ring_name else 0.0

    frozen = torch.stack(
        [sm.frozen_rings_mask[name].float() for name in ring_names],
        dim=1,
    )  # (N, 4)

    inventory = torch.stack(
        [sm.peg_inventory[name].float() for name in peg_names],
        dim=1,
    ) / 4.0  # (N, 4) normalised

    manipulable_mask = get_manipulable_rings_mask(env, sm, ring_names, task_phase, num_envs)

    geom = torch.cat(
        [
            tip_r,                                          # 3
            tip_l,                                          # 3
            ring_pos_local.reshape(num_envs, -1),           # 12
            peg_pos_local.reshape(num_envs, -1),            # 12
            rel_tip_r_to_rings.reshape(num_envs, -1),       # 12
            rel_tip_l_to_rings.reshape(num_envs, -1),       # 12
            rel_ring_to_matching_peg.reshape(num_envs, -1), # 12
            attached_r,                                     # 4
            attached_l,                                     # 4
            frozen,                                         # 4
            inventory,                                      # 4
            manipulable_mask,                               # 4
        ],
        dim=1,
    )

    assert geom.shape[1] == GEOM_DIM, (
        f"[observations] geom_obs dim mismatch: expected {GEOM_DIM}, got {geom.shape[1]}"
    )
    return geom


# ---------------------------------------------------------------------------
# Full observation assembly
# ---------------------------------------------------------------------------

def build_obs(
    emb_buffer: torch.Tensor,
    aux_obs: torch.Tensor,
    geom_obs: torch.Tensor,
    num_envs: int,
) -> np.ndarray:
    """Concatenate embedding stack, aux obs, and geom obs into a flat array.

    Args:
        emb_buffer: ``(num_envs, stack_size, emb_dim)`` embedding stack.
        aux_obs: ``(num_envs, AUX_DIM)`` auxiliary observation.
        geom_obs: ``(num_envs, GEOM_DIM)`` geometric observation.
        num_envs: Number of parallel environments.

    Returns:
        NumPy float32 array of shape ``(num_envs, OBS_DIM)`` on CPU.
    """
    emb_obs = emb_buffer.view(num_envs, -1)
    obs = torch.cat([emb_obs, aux_obs, geom_obs], dim=1)

    assert obs.shape[1] == OBS_DIM, (
        f"[observations] obs dim mismatch: expected {OBS_DIM}, got {obs.shape[1]}"
    )
    return obs.cpu().numpy().astype(np.float32)
