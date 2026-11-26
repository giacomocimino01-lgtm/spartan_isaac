"""State-aware action masking for the existing four-head HRL action space.

The public action representation remains ``MultiDiscrete([4, 10, 4, 10])``.
The policy still produces the same 28 logits as a regular SB3 policy, but the
four logits groups are combined into a masked joint categorical distribution.
This makes sampling, deterministic prediction, entropy, and PPO log-probability
all agree on which complete actions are legal.
"""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.save_util import load_from_zip_file
from torch.distributions import Categorical

from m_dVrk.hrl.constants import (
    ARM_ROOT_XY,
    BOARD_CENTER_TARGET_ID,
    OUTSIDE_LEFT_TARGET_ID,
    OUTSIDE_RIGHT_TARGET_ID,
    MAX_RING_PLANAR_REACH,
    TARGET_TO_ID,
    VERB_TO_ID,
)


ACTION_MASK_MODES = ("guided", "goal_agnostic", "minimal")


# Full-observation offsets. Keep these aligned with hrl.observations.
_EMBEDDING_DIM = 96
_CURRENT_ACTION_OFFSET = _EMBEDDING_DIM
_PREVIOUS_ACTION_OFFSET = _CURRENT_ACTION_OFFSET + 28
_BUSY_OFFSET = _PREVIOUS_ACTION_OFFSET + 28
_GEOMETRY_OFFSET = _BUSY_OFFSET + 2 + 2 + 2
_ATTACHED_RIGHT_OFFSET = _GEOMETRY_OFFSET + 66
_ATTACHED_LEFT_OFFSET = _GEOMETRY_OFFSET + 70
_MANIPULABLE_OFFSET = _GEOMETRY_OFFSET + 82
_RING_POSITION_OFFSET = _GEOMETRY_OFFSET + 6
_PEG_POSITION_OFFSET = _GEOMETRY_OFFSET + 18

# Local robot-root positions from MDvrkEnvCfg. A planar radius of 0.22 m
# retains an overlap region around the board centre while excluding only the
# diagonally opposite corners that repeatedly trap the IK controller.
def _decode_action_one_hot(obs: torch.Tensor, offset: int) -> torch.Tensor:
    """Decode one ``[verb_l, target_l, verb_r, target_r]`` one-hot block."""
    verb_l = obs[:, offset : offset + 4].argmax(dim=1)
    target_l = obs[:, offset + 4 : offset + 14].argmax(dim=1)
    verb_r = obs[:, offset + 14 : offset + 18].argmax(dim=1)
    target_r = obs[:, offset + 18 : offset + 28].argmax(dim=1)
    return torch.stack((verb_l, target_l, verb_r, target_r), dim=1)


def _valid_destination_ids(
    task_phase: str,
    device: torch.device,
    action_mask_mode: str,
) -> torch.Tensor:
    if action_mask_mode in {"goal_agnostic", "minimal"}:
        # Every non-ring, non-None target is a physically valid destination.
        # In phase 2 the peg_gray slot is decoded as the board centre.
        ids = [TARGET_TO_ID[name] for name in (
            "peg_red", "peg_yellow", "peg_green", "peg_blue", "peg_gray"
        )]
        if task_phase == "phase_4":
            ids.append(OUTSIDE_LEFT_TARGET_ID)
        return torch.tensor(ids, device=device)
    if task_phase == "phase_0":
        names = ("peg_green",)
    elif task_phase == "phase_1":
        names = ("peg_red", "peg_blue")
    elif task_phase == "phase_2":
        return torch.tensor([BOARD_CENTER_TARGET_ID], device=device)
    elif task_phase == "phase_3":
        names = ("peg_red", "peg_yellow", "peg_green", "peg_blue")
    elif task_phase == "phase_4":
        return torch.tensor(
            [OUTSIDE_RIGHT_TARGET_ID, OUTSIDE_LEFT_TARGET_ID], device=device
        )
    else:
        raise ValueError(f"Unsupported task phase for action masking: {task_phase}")
    return torch.tensor([TARGET_TO_ID[name] for name in names], device=device)


def build_joint_action_mask(
    obs: torch.Tensor,
    action_table: torch.Tensor,
    task_phase: str,
    action_mask_mode: str = "guided",
) -> torch.Tensor:
    """Return a boolean mask over all complete MultiDiscrete actions.

    The guided profile follows the goal-aware sequence historically enforced
    by the SPARTAN controller:

    ``reach ring -> grasp same ring -> reach goal peg -> release same peg``.

    The goal-agnostic profile exposes every physical destination, including
    non-goal pegs, while retaining the generic manipulation grammar used by
    the guided profile.  It therefore removes task-solution knowledge without
    immediately exposing all no-op transitions.

    The minimal profile additionally permits idle-idle, intermediate reaches
    and correction of already placed top rings. It keeps only controller
    preconditions and physical/conflict guards.

    Busy arms are forced to repeat the command that the wrapper actually keeps
    executing. Distinct rings may be manipulated concurrently, preserving
    bimanual behavior. Two arms cannot target the same ring or release onto the
    same peg in the same decision.
    """
    if obs.ndim != 2:
        raise ValueError(f"Expected batched observations, got shape={tuple(obs.shape)}")
    if action_mask_mode not in ACTION_MASK_MODES:
        raise ValueError(
            f"Unsupported action mask mode: {action_mask_mode}; "
            f"expected one of {ACTION_MASK_MODES}"
        )

    batch_size = obs.shape[0]
    device = obs.device
    actions = action_table.to(device=device)
    num_actions = actions.shape[0]

    current = _decode_action_one_hot(obs, _CURRENT_ACTION_OFFSET)
    previous = _decode_action_one_hot(obs, _PREVIOUS_ACTION_OFFSET)
    busy = obs[:, _BUSY_OFFSET : _BUSY_OFFSET + 2] > 0.5
    attached_right = obs[:, _ATTACHED_RIGHT_OFFSET : _ATTACHED_RIGHT_OFFSET + 4] > 0.5
    attached_left = obs[:, _ATTACHED_LEFT_OFFSET : _ATTACHED_LEFT_OFFSET + 4] > 0.5
    manipulable = obs[:, _MANIPULABLE_OFFSET : _MANIPULABLE_OFFSET + 4] > 0.5
    ring_positions = obs[:, _RING_POSITION_OFFSET : _RING_POSITION_OFFSET + 12].reshape(
        batch_size, 4, 3
    )
    peg_positions = obs[:, _PEG_POSITION_OFFSET : _PEG_POSITION_OFFSET + 12].reshape(
        batch_size, 4, 3
    )

    idle_id = VERB_TO_ID["idle"]
    reach_id = VERB_TO_ID["reach"]
    grasp_id = VERB_TO_ID["grasp"]
    release_id = VERB_TO_ID["release"]
    none_id = TARGET_TO_ID["None"]
    ring_ids = torch.tensor(
        [TARGET_TO_ID[name] for name in ("ring_red", "ring_yellow", "ring_green", "ring_blue")],
        device=device,
    )
    peg_ids = _valid_destination_ids(task_phase, device, action_mask_mode)
    minimal_mask = action_mask_mode == "minimal"

    def arm_mask(arm: str, verb_col: int, target_col: int) -> torch.Tensor:
        verb = actions[:, verb_col].view(1, num_actions)
        target = actions[:, target_col].view(1, num_actions)
        arm_idx = 0 if arm == "left" else 1
        current_cols = (0, 1) if arm == "left" else (2, 3)
        previous_cols = current_cols
        attached = attached_left if arm == "left" else attached_right

        is_busy = busy[:, arm_idx].view(batch_size, 1)
        current_verb = current[:, current_cols[0]].view(batch_size, 1)
        current_target = current[:, current_cols[1]].view(batch_size, 1)
        previous_verb = previous[:, previous_cols[0]].view(batch_size, 1)
        previous_target = previous[:, previous_cols[1]].view(batch_size, 1)
        holds_ring = attached.any(dim=1).view(batch_size, 1)

        target_is_ring = (target.unsqueeze(-1) == ring_ids.view(1, 1, -1)).any(dim=-1)
        target_is_peg = (target.unsqueeze(-1) == peg_ids.view(1, 1, -1)).any(dim=-1)
        if task_phase == "phase_3" and arm == "right":
            # Empirical controller reachability: centre->red consistently
            # times out for the right PSM.  This is a physical affordance,
            # not a strategic ring-to-peg assignment.
            target_is_peg &= target != TARGET_TO_ID["peg_red"]
        if task_phase == "phase_4":
            # The two virtual regions lie outside opposite sides of the board
            # and are deliberately arm-relative. Opposite-side reaches are
            # outside the validated planar workspace.
            if arm == "right":
                target_is_peg &= target != OUTSIDE_LEFT_TARGET_ID
            else:
                target_is_peg &= target != OUTSIDE_RIGHT_TARGET_ID
        ring_index = target.clamp(min=0, max=3).expand(batch_size, -1)
        target_manipulable = torch.gather(manipulable, 1, ring_index) & target_is_ring
        target_ring_xy = torch.gather(
            ring_positions[:, :, :2],
            1,
            ring_index.unsqueeze(-1).expand(-1, -1, 2),
        )
        arm_root_xy = torch.tensor(
            ARM_ROOT_XY[f"{arm}_arm"], device=device, dtype=obs.dtype
        )
        target_reachable = (
            torch.linalg.vector_norm(target_ring_xy - arm_root_xy, dim=-1)
            <= MAX_RING_PLANAR_REACH
        ) & target_is_ring
        if task_phase == "phase_2" and arm == "right":
            # Scripted feasibility checks show that the left PSM can grasp from
            # both phase-2 stacks, while the right PSM consistently fails to
            # grasp from the red stack.  Keep only that empirically physical
            # restriction; choosing which arm serves the blue stack is left to
            # the policy.
            red_peg_xy = peg_positions[:, 0, :2].unsqueeze(1)
            blue_peg_xy = peg_positions[:, 3, :2].unsqueeze(1)
            distance_to_red = torch.linalg.vector_norm(
                target_ring_xy - red_peg_xy, dim=-1
            )
            distance_to_blue = torch.linalg.vector_norm(
                target_ring_xy - blue_peg_xy, dim=-1
            )
            source_is_red = distance_to_red <= distance_to_blue
            target_reachable &= ~source_is_red

        idle_ok = (verb == idle_id) & (target == none_id)
        reach_ring_ok = (verb == reach_id) & target_manipulable & target_reachable
        if not minimal_mask:
            reach_ring_ok &= ~holds_ring
        reach_peg_ok = (verb == reach_id) & target_is_peg
        if not minimal_mask:
            reach_peg_ok = reach_peg_ok & holds_ring
        grasp_ok = (
            (verb == grasp_id)
            & target_manipulable
            & target_reachable
            & ~holds_ring
            & (previous_verb == reach_id)
            & (previous_target == target)
        )
        release_ok = (
            (verb == release_id)
            & target_is_peg
            & holds_ring
            & (previous_verb == reach_id)
            & (previous_target == target)
        )
        idle_arm_valid = idle_ok | reach_ring_ok | reach_peg_ok | grasp_ok | release_ok
        busy_arm_valid = (verb == current_verb) & (target == current_target)
        return torch.where(is_busy, busy_arm_valid, idle_arm_valid)

    valid = arm_mask("left", 0, 1) & arm_mask("right", 2, 3)

    verb_l = actions[:, 0].view(1, num_actions)
    target_l = actions[:, 1].view(1, num_actions)
    verb_r = actions[:, 2].view(1, num_actions)
    target_r = actions[:, 3].view(1, num_actions)
    left_ring_command = ((verb_l == reach_id) | (verb_l == grasp_id)) & (target_l < 4)
    right_ring_command = ((verb_r == reach_id) | (verb_r == grasp_id)) & (target_r < 4)
    same_ring = left_ring_command & right_ring_command & (target_l == target_r)
    same_release_peg = (verb_l == release_id) & (verb_r == release_id) & (target_l == target_r)
    both_idle = (verb_l == idle_id) & (target_l == none_id) & (verb_r == idle_id) & (
        target_r == none_id
    )
    invalid_joint_action = same_ring | same_release_peg
    if not minimal_mask and task_phase != "phase_2":
        # Retain the existing behavior for already-trained phases.  In phase 2
        # idle-idle is deliberately legal so the mask does not encode a
        # strategic preference to act at every decision.
        task_has_work = (
            manipulable.any(dim=1) | attached_left.any(dim=1) | attached_right.any(dim=1)
        ).view(batch_size, 1)
        invalid_joint_action = invalid_joint_action | (both_idle & task_has_work)
    valid &= ~invalid_joint_action

    if not valid.any(dim=1).all():
        raise RuntimeError("Action mask produced an environment with no legal joint action")
    return valid


class MaskedJointMultiCategoricalDistribution(MultiCategoricalDistribution):
    """Joint masked distribution backed by the standard four groups of logits."""

    def __init__(
        self,
        action_dims: list[int],
        task_phase: str,
        action_mask_mode: str = "guided",
        soft_mask_penalty: float = 0.0,
    ):
        super().__init__(action_dims)
        self.task_phase = task_phase
        self.action_mask_mode = action_mask_mode
        self.soft_mask_penalty = float(soft_mask_penalty)
        self.joint_distribution: Categorical | None = None
        self._action_table: torch.Tensor | None = None
        self.last_new_action_probability: torch.Tensor | None = None
        self.last_new_action_logit_advantage: torch.Tensor | None = None
        self._soft_mask_probability_batches: list[torch.Tensor] = []
        self._soft_mask_logit_advantage_batches: list[torch.Tensor] = []

    def set_soft_mask_penalty(self, value: float) -> None:
        value = float(value)
        if value < 0:
            raise ValueError("soft_mask_penalty must be non-negative")
        self.soft_mask_penalty = value

    def reset_soft_mask_diagnostics(self) -> None:
        """Start a fresh aggregation window for rollout-only diagnostics."""
        self._soft_mask_probability_batches.clear()
        self._soft_mask_logit_advantage_batches.clear()

    def soft_mask_diagnostics(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return all per-environment measurements collected in this window."""
        probabilities = (
            torch.cat(self._soft_mask_probability_batches)
            if self._soft_mask_probability_batches
            else None
        )
        advantages = (
            torch.cat(self._soft_mask_logit_advantage_batches)
            if self._soft_mask_logit_advantage_batches
            else None
        )
        return probabilities, advantages

    def _get_action_table(self, device: torch.device) -> torch.Tensor:
        if self._action_table is None or self._action_table.device != device:
            axes = [torch.arange(dim, device=device) for dim in self.action_dims]
            self._action_table = torch.cartesian_prod(*axes)
        return self._action_table

    def proba_distribution(
        self,
        action_logits: torch.Tensor,
        obs: torch.Tensor | None = None,
    ) -> "MaskedJointMultiCategoricalDistribution":
        if obs is None:
            raise RuntimeError("Masked distribution requires the observation used to produce the logits")
        split_logits = torch.split(action_logits, self.action_dims, dim=1)
        action_table = self._get_action_table(action_logits.device)
        joint_logits = sum(
            head[:, action_table[:, idx]] for idx, head in enumerate(split_logits)
        )
        valid = build_joint_action_mask(
            obs, action_table, self.task_phase, self.action_mask_mode
        )
        newly_exposed = None
        if self.action_mask_mode == "minimal" and self.soft_mask_penalty > 0:
            guided_valid = build_joint_action_mask(
                obs, action_table, self.task_phase, "guided"
            )
            newly_exposed = valid & ~guided_valid
            guided_max = joint_logits.masked_fill(~guided_valid, -torch.inf).max(dim=1).values
            new_max = joint_logits.masked_fill(~newly_exposed, -torch.inf).max(dim=1).values
            finite = torch.isfinite(guided_max) & torch.isfinite(new_max)
            self.last_new_action_logit_advantage = (
                (new_max[finite] - guided_max[finite]).detach() if finite.any() else None
            )
            if self.last_new_action_logit_advantage is not None:
                self._soft_mask_logit_advantage_batches.append(
                    self.last_new_action_logit_advantage
                )
            joint_logits = joint_logits - newly_exposed.to(joint_logits.dtype) * self.soft_mask_penalty
        masked_logits = joint_logits.masked_fill(~valid, torch.finfo(joint_logits.dtype).min)
        self.joint_distribution = Categorical(logits=masked_logits)
        if newly_exposed is not None:
            self.last_new_action_probability = (
                self.joint_distribution.probs * newly_exposed.to(joint_logits.dtype)
            ).sum(dim=1).detach()
            self._soft_mask_probability_batches.append(self.last_new_action_probability)
        else:
            self.last_new_action_probability = None
            self.last_new_action_logit_advantage = None
        return self

    def _require_distribution(self) -> Categorical:
        if self.joint_distribution is None:
            raise RuntimeError("Distribution parameters have not been initialized")
        return self.joint_distribution

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.long()
        flat_index = actions[:, 0]
        for idx, dim in enumerate(self.action_dims[1:], start=1):
            flat_index = flat_index * dim + actions[:, idx]
        return self._require_distribution().log_prob(flat_index)

    def entropy(self) -> torch.Tensor:
        return self._require_distribution().entropy()

    def sample(self) -> torch.Tensor:
        indices = self._require_distribution().sample()
        return self._get_action_table(indices.device)[indices]

    def mode(self) -> torch.Tensor:
        indices = self._require_distribution().probs.argmax(dim=1)
        return self._get_action_table(indices.device)[indices]


class MaskedMultiDiscreteActorCriticPolicy(ActorCriticPolicy):
    """SB3 actor-critic policy with state-aware joint masking."""

    def __init__(
        self,
        *args,
        mask_task_phase: str = "phase_0",
        action_mask_mode: str = "guided",
        soft_mask_penalty: float = 0.0,
        **kwargs,
    ):
        self.mask_task_phase = mask_task_phase
        self.action_mask_mode = action_mask_mode
        self.soft_mask_penalty = float(soft_mask_penalty)
        if self.soft_mask_penalty < 0:
            raise ValueError("soft_mask_penalty must be non-negative")
        self._distribution_obs: torch.Tensor | None = None
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_space, spaces.MultiDiscrete):
            raise TypeError("MaskedMultiDiscreteActorCriticPolicy requires a MultiDiscrete action space")
        self.action_dist = MaskedJointMultiCategoricalDistribution(
            list(self.action_space.nvec),
            task_phase=self.mask_task_phase,
            action_mask_mode=self.action_mask_mode,
            soft_mask_penalty=self.soft_mask_penalty,
        )

    def set_soft_mask_penalty(self, value: float) -> None:
        self.action_dist.set_soft_mask_penalty(value)
        self.soft_mask_penalty = self.action_dist.soft_mask_penalty

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        if self._distribution_obs is None:
            raise RuntimeError("Missing observation context for masked action distribution")
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits, self._distribution_obs)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        self._distribution_obs = obs
        return super().forward(obs, deterministic=deterministic)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        self._distribution_obs = obs
        return super().evaluate_actions(obs, actions)

    def get_distribution(self, obs: torch.Tensor):
        self._distribution_obs = obs
        return super().get_distribution(obs)


def masked_policy_custom_objects(
    checkpoint_path: str,
    task_phase: str,
    action_mask_mode: str = "guided",
    soft_mask_penalty: float = 0.0,
) -> dict:
    """Build ``PPO.load`` overrides while preserving a checkpoint's network layout."""
    data, _, _ = load_from_zip_file(checkpoint_path, device="cpu")
    if data is None:
        raise ValueError(f"No SB3 metadata found in checkpoint: {checkpoint_path}")
    policy_kwargs = dict(data.get("policy_kwargs", {}))
    policy_kwargs["mask_task_phase"] = task_phase
    policy_kwargs["action_mask_mode"] = action_mask_mode
    policy_kwargs["soft_mask_penalty"] = float(soft_mask_penalty)
    return {
        "policy_class": MaskedMultiDiscreteActorCriticPolicy,
        "policy_kwargs": policy_kwargs,
    }
