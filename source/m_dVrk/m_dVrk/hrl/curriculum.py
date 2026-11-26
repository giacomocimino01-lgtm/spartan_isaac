"""Curriculum helpers for progressively randomized ring resets."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


def scaled_interval_around_anchor(
    anchor: float,
    full_interval: tuple[float, float],
    scale: float,
) -> tuple[float, float]:
    """Expand an interval from an anchor (scale 0) to full bounds (scale 1)."""

    if full_interval[0] > full_interval[1]:
        raise ValueError("full_interval lower bound must not exceed upper bound")
    if not 0.0 <= scale <= 1.0:
        raise ValueError("scale must be in [0, 1]")

    lower = anchor + scale * (full_interval[0] - anchor)
    upper = anchor + scale * (full_interval[1] - anchor)
    return lower, upper


def phase_entry_pose_sampling_bounds(
    current_scale: float,
    frontier_width: float | None,
) -> tuple[float, float]:
    """Return the interpolation interval sampled at one pose-curriculum stage.

    ``frontier_width=None`` preserves the original cumulative curriculum and
    samples from canonical zero to the current maximum.  A positive width
    focuses resets on the newest difficulty band while adjacent stages keep
    the transition continuous.
    """

    if not 0.0 <= current_scale <= 1.0:
        raise ValueError("current_scale must be in [0, 1]")
    if frontier_width is None:
        return 0.0, float(current_scale)
    if not 0.0 < frontier_width <= 1.0:
        raise ValueError("frontier_width must be in (0, 1]")
    return max(0.0, float(current_scale) - float(frontier_width)), float(
        current_scale
    )


def validate_phase_entry_neighborhood(
    probability: float,
    scale: float,
) -> tuple[float, float]:
    """Validate the mixture probability and normalized park-neighborhood size."""

    probability = float(probability)
    scale = float(scale)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("phase-entry neighborhood probability must be in [0, 1]")
    if not 0.0 <= scale <= 1.0:
        raise ValueError("phase-entry neighborhood scale must be in [0, 1]")
    return probability, scale


@dataclass(frozen=True)
class RingRandomizationCurriculum:
    """Hold fixed poses, linearly expand their sampling area, then stay full."""

    total_timesteps: int
    start_timesteps: int
    full_timesteps: int

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if not 0 <= self.start_timesteps < self.full_timesteps <= self.total_timesteps:
            raise ValueError(
                "curriculum boundaries must satisfy "
                "0 <= start_timesteps < full_timesteps <= total_timesteps"
            )

    def value_at(self, timestep: int) -> float:
        step = max(0, min(int(timestep), self.total_timesteps))
        if step <= self.start_timesteps:
            return 0.0
        if step >= self.full_timesteps:
            return 1.0
        return (step - self.start_timesteps) / (
            self.full_timesteps - self.start_timesteps
        )


@dataclass
class AdaptiveRingRandomizationCurriculum:
    """Advance ring-reset difficulty only after sustained task success."""

    start_scale: float = 0.75
    scale_step: float = 0.05
    success_threshold: float = 0.85
    window_episodes: int = 512
    scale_tolerance: float = 1e-4
    current_scale: float = field(init=False)
    _recent_successes: deque[bool] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.start_scale < 1.0:
            raise ValueError("start_scale must be in [0, 1)")
        if not 0.0 < self.scale_step <= 1.0:
            raise ValueError("scale_step must be in (0, 1]")
        if not 0.0 < self.success_threshold <= 1.0:
            raise ValueError("success_threshold must be in (0, 1]")
        if self.window_episodes <= 0:
            raise ValueError("window_episodes must be positive")
        if self.scale_tolerance < 0.0:
            raise ValueError("scale_tolerance must be non-negative")

        self.current_scale = float(self.start_scale)
        self._recent_successes = deque(maxlen=int(self.window_episodes))

    @property
    def recent_episode_count(self) -> int:
        return len(self._recent_successes)

    @property
    def recent_success_rate(self) -> float | None:
        if not self._recent_successes:
            return None
        return sum(self._recent_successes) / len(self._recent_successes)

    def observe(
        self,
        outcomes: list[tuple[float, bool]],
        *,
        allow_advance: bool = True,
    ) -> bool:
        """Consume outcomes sampled at the current scale and maybe advance once."""

        for episode_scale, success in outcomes:
            if abs(float(episode_scale) - self.current_scale) <= self.scale_tolerance:
                self._recent_successes.append(bool(success))

        rate = self.recent_success_rate
        if (
            not allow_advance
            or self.current_scale >= 1.0
            or len(self._recent_successes) < self.window_episodes
            or rate is None
            or rate < self.success_threshold
        ):
            return False

        self.current_scale = min(
            1.0, round(self.current_scale + self.scale_step, 10)
        )
        self._recent_successes.clear()
        return True


@dataclass
class AdaptivePhaseEntryPoseCurriculum(AdaptiveRingRandomizationCurriculum):
    """Advance the maximum canonical-to-parked reset interpolation by success."""

    start_scale: float = 0.0
    scale_step: float = 0.25
    success_threshold: float = 0.85
    window_episodes: int = 128


@dataclass
class AdaptiveMinimalMaskCurriculum:
    """Open minimal-mask semantics only after sustained task success.

    The observation scale is advanced first.  Once it reaches one, the logit
    penalty on minimal-only actions is reduced.  A final successful window at
    scale one and penalty zero marks the curriculum complete.
    """

    start_scale: float = 0.55
    scale_step: float = 0.05
    initial_penalty: float = 8.0
    penalty_step: float = 1.0
    success_threshold: float = 0.85
    window_episodes: int = 256
    cooldown_timesteps: int = 32_768
    current_scale: float = field(init=False)
    current_penalty: float = field(init=False)
    complete: bool = field(init=False, default=False)
    _recent_successes: deque[bool] = field(init=False, repr=False)
    _ignore_until_timestep: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.start_scale <= 1.0:
            raise ValueError("start_scale must be in [0, 1]")
        if not 0.0 < self.scale_step <= 1.0:
            raise ValueError("scale_step must be in (0, 1]")
        if self.initial_penalty < 0.0:
            raise ValueError("initial_penalty must be non-negative")
        if self.penalty_step <= 0.0:
            raise ValueError("penalty_step must be positive")
        if not 0.0 < self.success_threshold <= 1.0:
            raise ValueError("success_threshold must be in (0, 1]")
        if self.window_episodes <= 0:
            raise ValueError("window_episodes must be positive")
        if self.cooldown_timesteps < 0:
            raise ValueError("cooldown_timesteps must be non-negative")

        self.current_scale = float(self.start_scale)
        self.current_penalty = float(self.initial_penalty)
        self._recent_successes = deque(maxlen=int(self.window_episodes))

    @property
    def recent_episode_count(self) -> int:
        return len(self._recent_successes)

    @property
    def recent_success_rate(self) -> float | None:
        if not self._recent_successes:
            return None
        return sum(self._recent_successes) / len(self._recent_successes)

    def observe(
        self,
        successes: list[bool],
        *,
        timestep: int,
        allow_advance: bool = True,
    ) -> tuple[float, float, float, float] | None:
        """Consume outcomes and return ``old_scale, old_penalty, new...`` on advance."""
        if self.complete or timestep < self._ignore_until_timestep:
            return None

        self._recent_successes.extend(bool(value) for value in successes)
        rate = self.recent_success_rate
        if (
            not allow_advance
            or len(self._recent_successes) < self.window_episodes
            or rate is None
            or rate < self.success_threshold
        ):
            return None

        old_scale = self.current_scale
        old_penalty = self.current_penalty
        if self.current_scale < 1.0:
            self.current_scale = min(
                1.0, round(self.current_scale + self.scale_step, 10)
            )
        elif self.current_penalty > 0.0:
            self.current_penalty = max(
                0.0, round(self.current_penalty - self.penalty_step, 10)
            )
        else:
            # This window validates the fully minimal configuration.
            self.complete = True

        self._recent_successes.clear()
        self._ignore_until_timestep = int(timestep) + self.cooldown_timesteps
        return old_scale, old_penalty, self.current_scale, self.current_penalty
