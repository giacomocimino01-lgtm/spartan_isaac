"""PPO regularization that preserves a behavior-cloned policy."""

from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import explained_variance
from m_dVrk.hrl.masked_policy import MaskedJointMultiCategoricalDistribution


class BCRegularizedPPO(PPO):
    """Add BC KL and/or supervised replay to the standard SB3 PPO loss.

    KL anchoring constrains the policy on its current on-policy states.  BC
    replay additionally samples original demonstration observations and adds
    their exact joint-action negative log likelihood.  The latter preserves
    demonstrated task strategy even when PPO visits out-of-distribution states
    where the frozen BC policy is not a reliable teacher.
    """

    kl_reference_policy: BasePolicy | None = None
    kl_anchor_coefficient: float = 0.0
    bc_replay_observations: th.Tensor | None = None
    bc_replay_actions: th.Tensor | None = None
    bc_replay_decision_indices: th.Tensor | None = None
    bc_replay_steady_indices: th.Tensor | None = None
    bc_replay_coefficient: float = 0.0
    bc_replay_batch_size: int = 0
    bc_replay_decision_fraction: float = 0.0

    def set_kl_anchor(
        self,
        reference_policy: BasePolicy,
        coefficient: float,
    ) -> None:
        """Attach and freeze the reference policy used during PPO updates."""
        coefficient = float(coefficient)
        if coefficient <= 0:
            raise ValueError("KL anchor coefficient must be positive")
        reference_policy.to(self.device)
        reference_policy.set_training_mode(False)
        for parameter in reference_policy.parameters():
            parameter.requires_grad = False
            parameter.grad = None
        self.kl_reference_policy = reference_policy
        self.kl_anchor_coefficient = coefficient

    def set_bc_replay(
        self,
        observations: np.ndarray | th.Tensor,
        actions: np.ndarray | th.Tensor,
        coefficient: float,
        batch_size: int,
        decision_fraction: float = 0.0,
    ) -> None:
        """Attach BC demonstrations sampled during every PPO minibatch.

        ``decision_fraction`` reserves part of each batch for observations at
        which the demonstrated joint action differs from the preceding row.
        This prevents long low-level action repetitions from overwhelming the
        comparatively rare semantic command changes.
        """
        coefficient = float(coefficient)
        batch_size = int(batch_size)
        decision_fraction = float(decision_fraction)
        if coefficient <= 0:
            raise ValueError("BC replay coefficient must be positive")
        if batch_size <= 0:
            raise ValueError("BC replay batch size must be positive")
        if not 0.0 <= decision_fraction <= 1.0:
            raise ValueError("BC replay decision fraction must be in [0, 1]")
        observations = th.as_tensor(observations, dtype=th.float32, device="cpu")
        actions = th.as_tensor(actions, dtype=th.long, device="cpu")
        if observations.ndim != 2:
            raise ValueError("BC replay observations must be a rank-2 array")
        if actions.ndim != 2:
            raise ValueError("BC replay actions must be a rank-2 array")
        if len(observations) != len(actions) or len(observations) == 0:
            raise ValueError("BC replay observations/actions must have equal non-zero length")
        if tuple(observations.shape[1:]) != tuple(self.observation_space.shape):
            raise ValueError(
                "BC replay observation shape does not match the policy: "
                f"{tuple(observations.shape[1:])} != {self.observation_space.shape}"
            )
        if actions.shape[1] != len(self.action_space.nvec):
            raise ValueError(
                "BC replay action width does not match the MultiDiscrete policy"
            )
        self.bc_replay_observations = observations.contiguous()
        self.bc_replay_actions = actions.contiguous()
        decision_mask = th.zeros(len(actions), dtype=th.bool, device="cpu")
        decision_mask[0] = True
        if len(actions) > 1:
            decision_mask[1:] = (actions[1:] != actions[:-1]).any(dim=1)
        self.bc_replay_decision_indices = th.nonzero(
            decision_mask,
            as_tuple=False,
        ).flatten()
        self.bc_replay_steady_indices = th.nonzero(
            ~decision_mask,
            as_tuple=False,
        ).flatten()
        self.bc_replay_coefficient = coefficient
        self.bc_replay_batch_size = batch_size
        self.bc_replay_decision_fraction = decision_fraction

    @staticmethod
    def _sample_replay_indices(
        total_count: int,
        batch_size: int,
        decision_indices: th.Tensor | None,
        steady_indices: th.Tensor | None,
        decision_fraction: float,
    ) -> th.Tensor:
        """Sample a mixed replay batch, with replacement, on the CPU."""
        if decision_fraction <= 0 or decision_indices is None:
            return th.randint(total_count, (batch_size,), device="cpu")

        decision_count = min(
            batch_size,
            max(0, int(round(batch_size * decision_fraction))),
        )
        steady_count = batch_size - decision_count
        if len(decision_indices) == 0:
            decision_count = 0
            steady_count = batch_size
        if steady_indices is None or len(steady_indices) == 0:
            decision_count = batch_size
            steady_count = 0

        sampled: list[th.Tensor] = []
        if decision_count:
            offsets = th.randint(
                len(decision_indices),
                (decision_count,),
                device="cpu",
            )
            sampled.append(decision_indices[offsets])
        if steady_count:
            assert steady_indices is not None
            offsets = th.randint(
                len(steady_indices),
                (steady_count,),
                device="cpu",
            )
            sampled.append(steady_indices[offsets])
        indices = th.cat(sampled)
        return indices[th.randperm(len(indices))]

    def _excluded_save_params(self) -> list[str]:
        """Do not duplicate the frozen BC policy in every PPO checkpoint."""
        return [
            *super()._excluded_save_params(),
            "kl_reference_policy",
            "bc_replay_observations",
            "bc_replay_actions",
            "bc_replay_decision_indices",
            "bc_replay_steady_indices",
        ]

    @staticmethod
    def _joint_distribution(policy: BasePolicy):
        action_dist = getattr(policy, "action_dist", None)
        if not isinstance(action_dist, MaskedJointMultiCategoricalDistribution):
            raise TypeError(
                "BCRegularizedPPO KL anchoring requires "
                "MaskedJointMultiCategoricalDistribution "
                "for both the trainable and reference policies"
            )
        return action_dist._require_distribution()

    @staticmethod
    def _stable_forward_kl(
        reference_distribution,
        current_distribution,
        *,
        max_log_ratio: float = 1_000_000.0,
    ) -> th.Tensor:
        """Return ``KL(reference || current)`` without float32 infinities.

        The masked categorical distribution represents invalid joint actions
        with the smallest finite float32 logit.  ``torch``'s generic
        categorical KL can therefore produce ``inf`` as soon as a currently
        tiny probability underflows to zero, poisoning the complete PPO loss.

        Compute the same expectation in float64 log space.  Terms with zero
        reference probability have zero measure and are omitted.  The very
        loose log-ratio bound affects only an impossible support mismatch
        (for example an action masked by one policy but not the other);
        ordinary underflowed valid probabilities retain their full gradient.
        """
        if max_log_ratio <= 0:
            raise ValueError("max_log_ratio must be positive")
        reference_logits = reference_distribution.logits.to(dtype=th.float64)
        current_logits = current_distribution.logits.to(dtype=th.float64)
        reference_log_probs = th.log_softmax(reference_logits, dim=-1)
        current_log_probs = th.log_softmax(current_logits, dim=-1)
        reference_probs = th.softmax(reference_logits, dim=-1)
        log_ratio = (reference_log_probs - current_log_probs).clamp(
            min=-float(max_log_ratio),
            max=float(max_log_ratio),
        )
        terms = th.where(
            reference_probs > 0,
            reference_probs * log_ratio,
            th.zeros_like(log_ratio),
        )
        # The exact KL is non-negative; tiny negative values can only come
        # from finite precision or the symmetric clipping above.
        return terms.sum(dim=-1).clamp_min(0.0).to(
            dtype=current_distribution.logits.dtype
        )

    def train(self) -> None:
        """Update the policy with the standard PPO loss plus the BC KL anchor."""
        coefficient = float(getattr(self, "kl_anchor_coefficient", 0.0))
        reference_policy = getattr(self, "kl_reference_policy", None)
        replay_coefficient = float(getattr(self, "bc_replay_coefficient", 0.0))
        replay_observations = getattr(self, "bc_replay_observations", None)
        replay_actions = getattr(self, "bc_replay_actions", None)
        replay_batch_size = int(getattr(self, "bc_replay_batch_size", 0))
        replay_decision_fraction = float(
            getattr(self, "bc_replay_decision_fraction", 0.0)
        )
        replay_decision_indices = getattr(self, "bc_replay_decision_indices", None)
        replay_steady_indices = getattr(self, "bc_replay_steady_indices", None)
        use_kl_anchor = coefficient > 0
        use_bc_replay = replay_coefficient > 0
        if not use_kl_anchor and not use_bc_replay:
            super().train()
            return
        if use_kl_anchor and reference_policy is None:
            raise RuntimeError(
                "A positive KL anchor coefficient requires set_kl_anchor() "
                "before learn()"
            )
        if use_bc_replay and (
            replay_observations is None
            or replay_actions is None
            or replay_batch_size <= 0
        ):
            raise RuntimeError(
                "A positive BC replay coefficient requires set_bc_replay() "
                "before learn()"
            )

        self.policy.set_training_mode(True)
        if reference_policy is not None:
            reference_policy.set_training_mode(False)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        anchor_kls: list[float] = []
        anchor_losses: list[float] = []
        replay_nlls: list[float] = []
        replay_losses: list[float] = []
        replay_accuracies: list[float] = []

        continue_training = True
        approx_kl_divs: list[float] = []
        loss = th.zeros((), device=self.device)
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                )
                anchor_loss = th.zeros((), device=self.device)
                if use_kl_anchor:
                    current_joint = self._joint_distribution(self.policy)
                    with th.no_grad():
                        reference_policy.get_distribution(rollout_data.observations)
                        reference_joint = self._joint_distribution(reference_policy)
                    anchor_kl = self._stable_forward_kl(
                        reference_joint,
                        current_joint,
                    ).mean()
                    anchor_loss = coefficient * anchor_kl
                    anchor_kls.append(anchor_kl.detach().item())
                    anchor_losses.append(anchor_loss.detach().item())

                replay_loss = th.zeros((), device=self.device)
                if use_bc_replay:
                    replay_indices = self._sample_replay_indices(
                        len(replay_observations),
                        replay_batch_size,
                        replay_decision_indices,
                        replay_steady_indices,
                        replay_decision_fraction,
                    )
                    expert_observations = replay_observations[replay_indices].to(
                        self.device,
                        non_blocking=True,
                    )
                    expert_actions = replay_actions[replay_indices].to(
                        self.device,
                        non_blocking=True,
                    )
                    expert_distribution = self.policy.get_distribution(
                        expert_observations
                    )
                    replay_nll = -expert_distribution.log_prob(expert_actions).mean()
                    replay_loss = replay_coefficient * replay_nll
                    with th.no_grad():
                        replay_accuracy = (
                            expert_distribution.mode() == expert_actions
                        ).all(dim=1).float().mean()
                    replay_nlls.append(replay_nll.detach().item())
                    replay_losses.append(replay_loss.detach().item())
                    replay_accuracies.append(replay_accuracy.detach().item())

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio,
                    1 - clip_range,
                    1 + clip_range,
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + anchor_loss
                    + replay_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio
                    ).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching "
                            f"max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm,
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if use_kl_anchor:
            self.logger.record("train/kl_anchor", np.mean(anchor_kls))
            self.logger.record("train/kl_anchor_loss", np.mean(anchor_losses))
            self.logger.record("train/kl_anchor_coefficient", coefficient)
        if use_bc_replay:
            self.logger.record("train/bc_replay_nll", np.mean(replay_nlls))
            self.logger.record("train/bc_replay_loss", np.mean(replay_losses))
            self.logger.record(
                "train/bc_replay_exact_accuracy",
                np.mean(replay_accuracies),
            )
            self.logger.record(
                "train/bc_replay_coefficient",
                replay_coefficient,
            )
            self.logger.record(
                "train/bc_replay_decision_fraction",
                replay_decision_fraction,
            )
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std",
                th.exp(self.policy.log_std).mean().item(),
            )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


# Backward-compatible name used by the first KL-only probe.
KLAnchoredPPO = BCRegularizedPPO
