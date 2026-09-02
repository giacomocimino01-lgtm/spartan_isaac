"""Serializable schedules used by PPO fine-tuning."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LowHighCosineSchedule:
    """Low LR during actor freeze, linear rise, plateau, then cosine decay."""

    total_timesteps: int
    freeze_timesteps: int
    ramp_timesteps: int
    decay_start_timesteps: int
    low_value: float
    high_value: float
    final_value: float

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if min(self.low_value, self.high_value, self.final_value) < 0:
            raise ValueError("Schedule values must be non-negative")

    def value_at(self, timestep: int) -> float:
        step = max(0, min(int(timestep), self.total_timesteps))
        freeze_end = max(0, min(self.freeze_timesteps, self.total_timesteps))
        ramp_end = min(freeze_end + max(0, self.ramp_timesteps), self.total_timesteps)
        decay_start = max(ramp_end, min(self.decay_start_timesteps, self.total_timesteps))

        if step < freeze_end:
            return self.low_value
        if step < ramp_end and ramp_end > freeze_end:
            ratio = (step - freeze_end) / (ramp_end - freeze_end)
            return self.low_value + ratio * (self.high_value - self.low_value)
        if step < decay_start:
            return self.high_value
        if decay_start >= self.total_timesteps:
            return self.high_value

        ratio = (step - decay_start) / (self.total_timesteps - decay_start)
        cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return self.final_value + (self.high_value - self.final_value) * cosine

    def __call__(self, progress_remaining: float) -> float:
        timestep = round((1.0 - float(progress_remaining)) * self.total_timesteps)
        return self.value_at(timestep)


@dataclass(frozen=True)
class CosineDecaySchedule:
    """Hold a value, then cosine-decay it toward a final value."""

    total_timesteps: int
    decay_start_timesteps: int
    initial_value: float
    final_value: float

    def value_at(self, timestep: int) -> float:
        step = max(0, min(int(timestep), self.total_timesteps))
        decay_start = max(0, min(self.decay_start_timesteps, self.total_timesteps))
        if step < decay_start or decay_start >= self.total_timesteps:
            return self.initial_value
        ratio = (step - decay_start) / (self.total_timesteps - decay_start)
        cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return self.final_value + (self.initial_value - self.final_value) * cosine

