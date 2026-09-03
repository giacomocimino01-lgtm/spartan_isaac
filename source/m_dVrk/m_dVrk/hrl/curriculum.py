"""Curriculum helpers for progressively randomized ring resets."""

from __future__ import annotations

from dataclasses import dataclass


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
