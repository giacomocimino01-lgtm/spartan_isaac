"""SB3 training callbacks for the m_dVrk PPO training loop.

All callbacks are defined here so they can be shared by training scripts
without being redefined inline.
"""

from __future__ import annotations

import csv
import math
import os
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback

from m_dVrk.hrl.curriculum import (
    AdaptiveMinimalMaskCurriculum,
    AdaptivePhaseEntryPoseCurriculum,
    AdaptiveRingRandomizationCurriculum,
    RingRandomizationCurriculum,
)
from m_dVrk.hrl.schedules import CosineDecaySchedule


class EpisodeCsvLoggerCallback(BaseCallback):
    """Log per-episode metrics to a CSV file during PPO training.

    Writes one row per completed episode with timing, reward, success, and
    task-specific ring placement counts.

    Args:
        file_path: Absolute path to the output CSV file (parent directory
            will be created if it does not exist).
        verbose: SB3 verbosity level (0 = silent).
    """

    def __init__(self, file_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.file_path = file_path
        self._file = None
        self._writer = None

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self._file = open(self.file_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "total_timesteps",
                "env_id",
                "episode_reward",
                "discounted_episode_return",
                "episode_length",
                "episode_time_seconds",
                "is_success",
                "rings_complete",
                "invalid_commands",
                "bimanual_steps",
                "left_active_steps",
                "right_active_steps",
                "joint_action_changes",
                "free_arm_step_slots",
                "terminal_distance",
                "green_peg_ring_count",
                "red_peg_ring_count",
                "yellow_peg_ring_count",
                "blue_peg_ring_count",
                "board_center_ring_count",
                "time_limit_truncated",
                "ring_randomization_scale",
                "phase_entry_pose_scale",
                "phase_entry_pose_sampled_scale",
            ],
        )
        self._writer.writeheader()
        self._file.flush()
        print(f"[PPO] Episode CSV logger active: {self.file_path}", flush=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None or self._writer is None:
            return True

        wrote_rows = False
        for env_id, info in enumerate(infos):
            episode = info.get("episode")
            if episode is None:
                continue

            self._writer.writerow(
                {
                    "total_timesteps": int(self.num_timesteps),
                    "env_id": env_id,
                    "episode_reward": float(episode.get("r", 0.0)),
                    "discounted_episode_return": float(
                        info.get("discounted_episode_return", float("nan"))
                    ),
                    "episode_length": int(episode.get("l", 0)),
                    "episode_time_seconds": float(episode.get("t", 0.0)),
                    "is_success": bool(info.get("is_success", False)),
                    "rings_complete": bool(info.get("rings_complete", False)),
                    "invalid_commands": int(
                        info.get("episode_invalid_commands", 0)
                    ),
                    "bimanual_steps": int(info.get("episode_bimanual_steps", 0)),
                    "left_active_steps": int(
                        info.get("episode_left_active_steps", 0)
                    ),
                    "right_active_steps": int(
                        info.get("episode_right_active_steps", 0)
                    ),
                    "joint_action_changes": int(
                        info.get("episode_joint_action_changes", 0)
                    ),
                    "free_arm_step_slots": int(
                        info.get("episode_free_arm_step_slots", 0)
                    ),
                    "terminal_distance": float(info.get("terminal_distance", float("nan"))),
                    "green_peg_ring_count": int(info.get("green_peg_ring_count", 0)),
                    "red_peg_ring_count": int(info.get("red_peg_ring_count", 0)),
                    "yellow_peg_ring_count": int(
                        info.get("yellow_peg_ring_count", 0)
                    ),
                    "blue_peg_ring_count": int(info.get("blue_peg_ring_count", 0)),
                    "board_center_ring_count": int(
                        info.get("board_center_ring_count", 0)
                    ),
                    "time_limit_truncated": bool(info.get("TimeLimit.truncated", False)),
                    "ring_randomization_scale": float(
                        info.get("ring_randomization_scale", float("nan"))
                    ),
                    "phase_entry_pose_scale": float(
                        info.get("phase_entry_pose_scale", float("nan"))
                    ),
                    "phase_entry_pose_sampled_scale": float(
                        info.get("phase_entry_pose_sampled_scale", float("nan"))
                    ),
                }
            )
            wrote_rows = True

        if wrote_rows and self._file is not None:
            self._file.flush()

        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


class BestTerminalDistanceCheckpointCallback(BaseCallback):
    """Save the model whenever the rolling mean terminal TCC distance improves.

    Args:
        save_dir: Directory where checkpoints are saved.
        file_prefix: Filename prefix for the saved model and metadata.
        window_size: Number of recent terminal distances used for the rolling mean.
        verbose: SB3 verbosity level.
    """

    def __init__(
        self,
        save_dir: str,
        file_prefix: str,
        window_size: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.save_dir = save_dir
        self.file_prefix = file_prefix
        self.window_size = max(1, int(window_size))
        self.best_mean_terminal_distance = float("inf")
        self.recent_terminal_distances: deque[float] = deque(maxlen=self.window_size)

    def _on_training_start(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None:
            return True

        for info in infos:
            episode = info.get("episode")
            terminal_distance = info.get("terminal_distance")
            if episode is None or terminal_distance is None:
                continue
            terminal_distance = float(terminal_distance)
            if not math.isfinite(terminal_distance):
                continue
            self.recent_terminal_distances.append(terminal_distance)

        if len(self.recent_terminal_distances) < self.window_size:
            return True

        mean_terminal_distance = sum(self.recent_terminal_distances) / len(
            self.recent_terminal_distances
        )
        if mean_terminal_distance >= self.best_mean_terminal_distance:
            return True

        self.best_mean_terminal_distance = mean_terminal_distance

        model_path        = os.path.join(self.save_dir, f"{self.file_prefix}.zip")
        vec_normalize_path = os.path.join(self.save_dir, f"{self.file_prefix}_vecnormalize.pkl")
        metric_path       = os.path.join(self.save_dir, f"{self.file_prefix}_metrics.txt")

        self.model.save(model_path)

        vec_normalize_env = self.model.get_vec_normalize_env()
        if vec_normalize_env is not None:
            vec_normalize_env.save(vec_normalize_path)

        with open(metric_path, "w", encoding="ascii") as f:
            f.write(f"num_timesteps={int(self.num_timesteps)}\n")
            f.write(f"window_size={len(self.recent_terminal_distances)}\n")
            f.write(f"mean_terminal_distance={mean_terminal_distance:.8f}\n")

        if self.verbose > 0:
            print(
                "[Checkpoint] New best terminal distance: "
                f"mean={mean_terminal_distance:.6f} "
                f"over last {len(self.recent_terminal_distances)} episodes"
            )

        return True


class FreezePolicyCallback(BaseCallback):
    """Freeze / unfreeze PPO actor weights around a BC-pretrained checkpoint.

    The actor (``policy_net`` + ``action_net``) is frozen at training start
    and unfrozen after *freeze_timesteps* steps, allowing the critic to warm
    up before the full joint update begins.

    Args:
        freeze_timesteps: Number of timesteps to keep the actor frozen.
            Set to 0 to disable freezing entirely.
        verbose: SB3 verbosity level.
    """

    def __init__(self, freeze_timesteps: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.freeze_timesteps = freeze_timesteps
        self.policy_frozen = False

    def _on_training_start(self) -> None:
        if self.freeze_timesteps > 0:
            self.freeze_policy()

    def _on_step(self) -> bool:
        if self.policy_frozen and self.num_timesteps >= self.freeze_timesteps:
            self.unfreeze_policy()
        return True

    def freeze_policy(self) -> None:
        """Freeze actor parameters (``policy_net`` and ``action_net``)."""
        print(
            f"[PPO] Freezing policy (actor) parameters for the first "
            f"{self.freeze_timesteps} timesteps.",
            flush=True,
        )
        freeze_count = 0
        frozen_names: list[str] = []
        for name, param in self.model.policy.named_parameters():
            if "policy_net" in name or "action_net" in name:
                param.requires_grad = False
                param.grad = None
                freeze_count += 1
                frozen_names.append(name)
        print(f"[PPO] Froze {freeze_count} parameters:", flush=True)
        for pname in frozen_names:
            print(f"      - {pname}", flush=True)
        self.policy_frozen = True

    def unfreeze_policy(self) -> None:
        """Re-enable gradient computation for the actor parameters."""
        print(
            f"[PPO] Unfreezing policy (actor) parameters after "
            f"{self.num_timesteps} timesteps.",
            flush=True,
        )
        unfreeze_count = 0
        for name, param in self.model.policy.named_parameters():
            if "policy_net" in name or "action_net" in name:
                param.requires_grad = True
                unfreeze_count += 1
        print(f"[PPO] Unfroze {unfreeze_count} parameters.", flush=True)
        self.policy_frozen = False


class EntropyCoefficientScheduleCallback(BaseCallback):
    """Update PPO's scalar entropy coefficient during training."""

    def __init__(self, schedule: CosineDecaySchedule, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.schedule = schedule
        self._last_logged_bucket = -1

    def _set_value(self) -> None:
        value = self.schedule.value_at(self.num_timesteps)
        self.model.ent_coef = value
        self.logger.record("train/ent_coef_schedule", value)

        bucket = int(10 * self.num_timesteps / max(1, self.schedule.total_timesteps))
        if self.verbose > 0 and bucket != self._last_logged_bucket:
            print(
                f"[PPO] Entropy schedule: timestep={self.num_timesteps} "
                f"ent_coef={value:.8g}",
                flush=True,
            )
            self._last_logged_bucket = bucket

    def _on_training_start(self) -> None:
        self._set_value()

    def _on_step(self) -> bool:
        self._set_value()
        return True

class SoftMaskPenaltyScheduleCallback(BaseCallback):
    """Anneal the logit penalty applied to minimal-only joint actions.

    The actions remain legal throughout training.  A positive penalty merely
    lowers their sampling probability relative to actions also admitted by
    the guided mask, avoiding an abrupt distribution shift when a pretrained
    guided policy is loaded under the minimal mask.
    """

    def __init__(self, schedule: CosineDecaySchedule, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.schedule = schedule
        self._last_logged_bucket = -1

    def _set_value(self) -> None:
        value = self.schedule.value_at(self.num_timesteps)
        setter = getattr(self.model.policy, "set_soft_mask_penalty", None)
        if setter is None:
            raise TypeError("Soft-mask schedule requires a masked policy")
        setter(value)
        self.model.policy_kwargs["soft_mask_penalty"] = value
        self.logger.record("train/soft_mask_penalty", value)

        bucket = int(10 * self.num_timesteps / max(1, self.schedule.total_timesteps))
        if self.verbose > 0 and bucket != self._last_logged_bucket:
            print(
                f"[PPO] Soft-mask schedule: timestep={self.num_timesteps} "
                f"logit_penalty={value:.6g}",
                flush=True,
            )
            self._last_logged_bucket = bucket

    def _on_training_start(self) -> None:
        self._set_value()

    def _on_step(self) -> bool:
        self._set_value()
        return True

    def _on_rollout_start(self) -> None:
        distribution = getattr(self.model.policy, "action_dist", None)
        reset = getattr(distribution, "reset_soft_mask_diagnostics", None)
        if reset is not None:
            reset()

    def _on_rollout_end(self) -> None:
        distribution = getattr(self.model.policy, "action_dist", None)
        diagnostics = getattr(distribution, "soft_mask_diagnostics", None)
        if diagnostics is not None:
            new_probability, logit_advantage = diagnostics()
        else:
            new_probability = getattr(distribution, "last_new_action_probability", None)
            logit_advantage = getattr(
                distribution, "last_new_action_logit_advantage", None
            )
        if new_probability is not None and new_probability.numel():
            self.logger.record(
                "train/soft_new_prob",
                float(new_probability.float().mean().item()),
            )
            self.logger.record(
                "train/soft_new_prob_max",
                float(new_probability.float().max().item()),
            )
        if logit_advantage is not None and logit_advantage.numel():
            self.logger.record(
                "train/soft_logit_adv",
                float(logit_advantage.float().mean().item()),
            )
            self.logger.record(
                "train/soft_logit_adv_max",
                float(logit_advantage.float().max().item()),
            )


class ManipulabilityObservationCurriculumCallback(BaseCallback):
    """Gradually expose completed top rings as physically manipulable."""

    def __init__(
        self,
        hrl_env: object,
        schedule: RingRandomizationCurriculum,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.hrl_env = hrl_env
        self.schedule = schedule
        self._last_logged_bucket = -1

    def _set_value(self) -> None:
        value = self.schedule.value_at(self.num_timesteps)
        setter = getattr(self.hrl_env, "set_manipulability_curriculum_scale", None)
        if setter is None:
            raise TypeError("Manipulability curriculum requires DVRKVisionHRLWrapper")
        setter(value)
        self.logger.record("train/manipulability_scale", value)

        bucket = int(10 * self.num_timesteps / max(1, self.schedule.total_timesteps))
        if self.verbose > 0 and bucket != self._last_logged_bucket:
            print(
                f"[PPO] Manipulability observation curriculum: "
                f"timestep={self.num_timesteps} scale={value:.4f}",
                flush=True,
            )
            self._last_logged_bucket = bucket

    def _on_training_start(self) -> None:
        self._set_value()

    def _on_step(self) -> bool:
        self._set_value()
        return True


class AdaptiveMinimalMaskCurriculumCallback(BaseCallback):
    """Advance observation semantics and soft masking from success windows."""

    def __init__(
        self,
        hrl_env: object,
        curriculum: AdaptiveMinimalMaskCurriculum,
        min_advance_timesteps: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.hrl_env = hrl_env
        self.curriculum = curriculum
        self.min_advance_timesteps = max(0, int(min_advance_timesteps))

    def _apply_state(self) -> None:
        env_setter = getattr(
            self.hrl_env, "set_manipulability_curriculum_scale", None
        )
        policy_setter = getattr(self.model.policy, "set_soft_mask_penalty", None)
        if env_setter is None or policy_setter is None:
            raise TypeError(
                "Adaptive minimal-mask curriculum requires the HRL wrapper "
                "and a masked policy"
            )
        env_setter(self.curriculum.current_scale)
        policy_setter(self.curriculum.current_penalty)
        self.model.policy_kwargs["soft_mask_penalty"] = (
            self.curriculum.current_penalty
        )

    def _record_state(self) -> None:
        self.logger.record(
            "curriculum/manipulability_scale", self.curriculum.current_scale
        )
        self.logger.record(
            "curriculum/soft_mask_penalty", self.curriculum.current_penalty
        )
        self.logger.record(
            "curriculum/recent_episode_count",
            self.curriculum.recent_episode_count,
        )
        rate = self.curriculum.recent_success_rate
        if rate is not None:
            self.logger.record("curriculum/recent_success_rate", rate)
        self.logger.record(
            "curriculum/minimal_complete", float(self.curriculum.complete)
        )

    def _on_training_start(self) -> None:
        self._apply_state()
        self._record_state()
        if self.verbose > 0:
            print(
                "[PPO] Adaptive minimal-mask curriculum: "
                f"scale={self.curriculum.current_scale:.3f} "
                f"penalty={self.curriculum.current_penalty:.3f} "
                f"scale_step={self.curriculum.scale_step:.3f} "
                f"penalty_step={self.curriculum.penalty_step:.3f} "
                f"threshold={self.curriculum.success_threshold:.2%} "
                f"window={self.curriculum.window_episodes} "
                f"cooldown={self.curriculum.cooldown_timesteps}",
                flush=True,
            )

    def _on_rollout_start(self) -> None:
        distribution = getattr(self.model.policy, "action_dist", None)
        reset = getattr(distribution, "reset_soft_mask_diagnostics", None)
        if reset is not None:
            reset()

    def _on_rollout_end(self) -> None:
        distribution = getattr(self.model.policy, "action_dist", None)
        diagnostics = getattr(distribution, "soft_mask_diagnostics", None)
        if diagnostics is None:
            return
        probabilities, advantages = diagnostics()
        if probabilities is not None and probabilities.numel():
            self.logger.record(
                "train/soft_new_prob", float(probabilities.float().mean().item())
            )
            self.logger.record(
                "train/soft_new_prob_max", float(probabilities.float().max().item())
            )
        if advantages is not None and advantages.numel():
            self.logger.record(
                "train/soft_logit_adv", float(advantages.float().mean().item())
            )
            self.logger.record(
                "train/soft_logit_adv_max", float(advantages.float().max().item())
            )

    def _on_step(self) -> bool:
        successes = [
            bool(info.get("is_success", False))
            for info in (self.locals.get("infos") or [])
            if info.get("episode") is not None
        ]
        transition = self.curriculum.observe(
            successes,
            timestep=self.num_timesteps,
            allow_advance=self.num_timesteps >= self.min_advance_timesteps,
        )
        if transition is not None:
            old_scale, old_penalty, new_scale, new_penalty = transition
            self._apply_state()
            if self.verbose > 0:
                print(
                    "[PPO] Adaptive minimal-mask curriculum advanced: "
                    f"timestep={self.num_timesteps} "
                    f"scale={old_scale:.3f}->{new_scale:.3f} "
                    f"penalty={old_penalty:.3f}->{new_penalty:.3f} "
                    f"complete={self.curriculum.complete}",
                    flush=True,
                )
        self._record_state()
        return True


class RingRandomizationCurriculumCallback(BaseCallback):
    """Update the ring-reset randomization scale during PPO training."""

    def __init__(
        self,
        raw_env: object,
        schedule: RingRandomizationCurriculum,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.raw_env = raw_env
        self.schedule = schedule
        self._last_logged_bucket = -1

    def _set_scale(self) -> None:
        scale = self.schedule.value_at(self.num_timesteps)
        setattr(self.raw_env, "ring_randomization_scale", scale)
        self.logger.record("curriculum/ring_randomization_scale", scale)

        bucket = int(scale * 10)
        if self.verbose > 0 and bucket != self._last_logged_bucket:
            print(
                f"[PPO] Ring curriculum: timestep={self.num_timesteps} "
                f"randomization_scale={scale:.4f}",
                flush=True,
            )
            self._last_logged_bucket = bucket

    def _on_training_start(self) -> None:
        self._set_scale()

    def _on_step(self) -> bool:
        self._set_scale()
        return True


class AdaptiveRingRandomizationCurriculumCallback(BaseCallback):
    """Advance ring difficulty from task-success windows, not wall-clock time."""

    def __init__(
        self,
        raw_env: object,
        curriculum: AdaptiveRingRandomizationCurriculum,
        min_advance_timesteps: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.raw_env = raw_env
        self.curriculum = curriculum
        self.min_advance_timesteps = max(0, int(min_advance_timesteps))

    def _record_state(self) -> None:
        self.logger.record(
            "curriculum/ring_randomization_scale",
            self.curriculum.current_scale,
        )
        rate = self.curriculum.recent_success_rate
        if rate is not None:
            self.logger.record("curriculum/recent_success_rate", rate)
        self.logger.record(
            "curriculum/recent_episode_count",
            self.curriculum.recent_episode_count,
        )

    def _on_training_start(self) -> None:
        setattr(
            self.raw_env,
            "ring_randomization_scale",
            self.curriculum.current_scale,
        )
        self._record_state()
        if self.verbose > 0:
            print(
                "[PPO] Adaptive ring curriculum: "
                f"start_scale={self.curriculum.current_scale:.4f} "
                f"step={self.curriculum.scale_step:.4f} "
                f"threshold={self.curriculum.success_threshold:.2%} "
                f"window={self.curriculum.window_episodes} "
                f"advance_after={self.min_advance_timesteps}",
                flush=True,
            )

    def _on_step(self) -> bool:
        outcomes: list[tuple[float, bool]] = []
        for info in self.locals.get("infos") or []:
            if info.get("episode") is None:
                continue
            episode_scale = info.get("ring_randomization_scale")
            if episode_scale is None:
                continue
            outcomes.append((float(episode_scale), bool(info.get("is_success", False))))

        old_scale = self.curriculum.current_scale
        advanced = self.curriculum.observe(
            outcomes,
            allow_advance=self.num_timesteps >= self.min_advance_timesteps,
        )
        if advanced:
            setattr(
                self.raw_env,
                "ring_randomization_scale",
                self.curriculum.current_scale,
            )
            if self.verbose > 0:
                print(
                    "[PPO] Adaptive ring curriculum advanced: "
                    f"timestep={self.num_timesteps} "
                    f"scale={old_scale:.4f}->{self.curriculum.current_scale:.4f}",
                    flush=True,
                )

        self._record_state()
        return True


class AdaptivePhaseEntryPoseCurriculumCallback(BaseCallback):
    """Advance canonical-to-parked reset difficulty after successful windows."""

    def __init__(
        self,
        raw_env: object,
        curriculum: AdaptivePhaseEntryPoseCurriculum,
        min_advance_timesteps: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.raw_env = raw_env
        self.curriculum = curriculum
        self.min_advance_timesteps = max(0, int(min_advance_timesteps))

    def _record_state(self) -> None:
        self.logger.record(
            "curriculum/phase_entry_pose_scale",
            self.curriculum.current_scale,
        )
        rate = self.curriculum.recent_success_rate
        if rate is not None:
            self.logger.record("curriculum/phase_entry_recent_success_rate", rate)
        self.logger.record(
            "curriculum/phase_entry_recent_episode_count",
            self.curriculum.recent_episode_count,
        )

    def _on_training_start(self) -> None:
        setattr(
            self.raw_env,
            "phase_entry_pose_scale",
            self.curriculum.current_scale,
        )
        self._record_state()
        if self.verbose > 0:
            print(
                "[PPO] Adaptive phase-entry pose curriculum: "
                f"start_scale={self.curriculum.current_scale:.3f} "
                f"step={self.curriculum.scale_step:.3f} "
                f"threshold={self.curriculum.success_threshold:.2%} "
                f"window={self.curriculum.window_episodes} "
                f"advance_after={self.min_advance_timesteps}",
                flush=True,
            )

    def _on_step(self) -> bool:
        outcomes: list[tuple[float, bool]] = []
        for info in self.locals.get("infos") or []:
            if info.get("episode") is None:
                continue
            episode_scale = info.get("phase_entry_pose_scale")
            if episode_scale is None:
                continue
            outcomes.append((float(episode_scale), bool(info.get("is_success", False))))

        old_scale = self.curriculum.current_scale
        advanced = self.curriculum.observe(
            outcomes,
            allow_advance=self.num_timesteps >= self.min_advance_timesteps,
        )
        if advanced:
            setattr(
                self.raw_env,
                "phase_entry_pose_scale",
                self.curriculum.current_scale,
            )
            if self.verbose > 0:
                print(
                    "[PPO] Adaptive phase-entry pose curriculum advanced: "
                    f"timestep={self.num_timesteps} "
                    f"scale={old_scale:.3f}->{self.curriculum.current_scale:.3f}",
                    flush=True,
                )

        self._record_state()
        return True
