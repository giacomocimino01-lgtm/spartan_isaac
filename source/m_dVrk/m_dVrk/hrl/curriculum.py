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
