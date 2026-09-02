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

from m_dVrk.hrl.constants import TARGET_TO_ID, VERB_TO_ID


# Full-observation offsets. Keep these aligned with hrl.observations.
_EMBEDDING_DIM = 96
_CURRENT_ACTION_OFFSET = _EMBEDDING_DIM
_PREVIOUS_ACTION_OFFSET = _CURRENT_ACTION_OFFSET + 28
_BUSY_OFFSET = _PREVIOUS_ACTION_OFFSET + 28
_GEOMETRY_OFFSET = _BUSY_OFFSET + 2 + 2 + 2
_ATTACHED_RIGHT_OFFSET = _GEOMETRY_OFFSET + 66
_ATTACHED_LEFT_OFFSET = _GEOMETRY_OFFSET + 70
_MANIPULABLE_OFFSET = _GEOMETRY_OFFSET + 82


def _decode_action_one_hot(obs: torch.Tensor, offset: int) -> torch.Tensor:
    """Decode one ``[verb_l, target_l, verb_r, target_r]`` one-hot block."""
    verb_l = obs[:, offset : offset + 4].argmax(dim=1)
    target_l = obs[:, offset + 4 : offset + 14].argmax(dim=1)
    verb_r = obs[:, offset + 14 : offset + 18].argmax(dim=1)
    target_r = obs[:, offset + 18 : offset + 28].argmax(dim=1)
    return torch.stack((verb_l, target_l, verb_r, target_r), dim=1)


def _valid_peg_ids(task_phase: str, device: torch.device) -> torch.Tensor:
    if task_phase == "phase_0":
        names = ("peg_green",)
    elif task_phase == "phase_1":
        names = ("peg_red", "peg_blue")
    else:
        raise ValueError(f"Unsupported task phase for action masking: {task_phase}")
    return torch.tensor([TARGET_TO_ID[name] for name in names], device=device)


def build_joint_action_mask(
    obs: torch.Tensor,
    action_table: torch.Tensor,
    task_phase: str,
) -> torch.Tensor:
    """Return a boolean mask over all complete MultiDiscrete actions.

    The mask follows the sequence enforced by the SPARTAN controller:

    ``reach ring -> grasp same ring -> reach goal peg -> release same peg``.

    Busy arms are forced to repeat the command that the wrapper actually keeps
    executing. Distinct rings may be manipulated concurrently, preserving
    bimanual behavior. Two arms cannot target the same ring or release onto the
    same peg in the same decision.
    """
    if obs.ndim != 2:
        raise ValueError(f"Expected batched observations, got shape={tuple(obs.shape)}")

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

    idle_id = VERB_TO_ID["idle"]
    reach_id = VERB_TO_ID["reach"]
    grasp_id = VERB_TO_ID["grasp"]
    release_id = VERB_TO_ID["release"]
    none_id = TARGET_TO_ID["None"]
    ring_ids = torch.tensor(
        [TARGET_TO_ID[name] for name in ("ring_red", "ring_yellow", "ring_green", "ring_blue")],
        device=device,
    )
    peg_ids = _valid_peg_ids(task_phase, device)

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
        ring_index = target.clamp(min=0, max=3).expand(batch_size, -1)
        target_manipulable = torch.gather(manipulable, 1, ring_index) & target_is_ring

        idle_ok = (verb == idle_id) & (target == none_id)
        reach_ring_ok = (verb == reach_id) & target_manipulable & ~holds_ring
        reach_peg_ok = (verb == reach_id) & target_is_peg & holds_ring
        grasp_ok = (
            (verb == grasp_id)
            & target_manipulable
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
    valid &= ~(same_ring | same_release_peg)

    if not valid.any(dim=1).all():
        raise RuntimeError("Action mask produced an environment with no legal joint action")
    return valid


class MaskedJointMultiCategoricalDistribution(MultiCategoricalDistribution):
    """Joint masked distribution backed by the standard four groups of logits."""

    def __init__(self, action_dims: list[int], task_phase: str):
        super().__init__(action_dims)
        self.task_phase = task_phase
        self.joint_distribution: Categorical | None = None
        self._action_table: torch.Tensor | None = None

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
        valid = build_joint_action_mask(obs, action_table, self.task_phase)
        masked_logits = joint_logits.masked_fill(~valid, torch.finfo(joint_logits.dtype).min)
        self.joint_distribution = Categorical(logits=masked_logits)
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

    def __init__(self, *args, mask_task_phase: str = "phase_0", **kwargs):
        self.mask_task_phase = mask_task_phase
        self._distribution_obs: torch.Tensor | None = None
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_space, spaces.MultiDiscrete):
            raise TypeError("MaskedMultiDiscreteActorCriticPolicy requires a MultiDiscrete action space")
        self.action_dist = MaskedJointMultiCategoricalDistribution(
            list(self.action_space.nvec),
            task_phase=self.mask_task_phase,
        )

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


def masked_policy_custom_objects(checkpoint_path: str, task_phase: str) -> dict:
    """Build ``PPO.load`` overrides while preserving a checkpoint's network layout."""
    data, _, _ = load_from_zip_file(checkpoint_path, device="cpu")
    if data is None:
        raise ValueError(f"No SB3 metadata found in checkpoint: {checkpoint_path}")
    policy_kwargs = dict(data.get("policy_kwargs", {}))
    policy_kwargs["mask_task_phase"] = task_phase
    return {
        "policy_class": MaskedMultiDiscreteActorCriticPolicy,
        "policy_kwargs": policy_kwargs,
    }
