"""Phase selection from phase-specific TCC start embeddings.

This module intentionally implements the simplest routing hypothesis: encode
the same observation with every candidate phase encoder and select the phase
whose raw embedding is closest to that encoder's mean start embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class RawPhaseSelection:
    """Result of a raw nearest-start phase selection."""

    phase_names: tuple[str, ...]
    selected_indices: torch.Tensor
    selected_phases: tuple[str, ...]
    squared_distances: torch.Tensor
    margins: torch.Tensor


@dataclass
class TCCSuccessGate:
    """Debounce a phase-success decision from raw TCC goal distance.

    A transition is emitted only after the squared embedding distance remains
    below ``threshold`` for ``consecutive_steps`` observations.  ``min_steps``
    prevents a visually ambiguous reset frame from immediately completing a
    phase.
    """

    threshold: float
    consecutive_steps: int = 5
    min_steps: int = 20
    streak: int = 0

    def __post_init__(self) -> None:
        if self.threshold < 0:
            raise ValueError("TCC success threshold must be non-negative")
        if self.consecutive_steps <= 0:
            raise ValueError("TCC consecutive_steps must be positive")
        if self.min_steps < 0:
            raise ValueError("TCC min_steps must be non-negative")

    def update(
        self,
        squared_distance: float,
        phase_step: int,
        *,
        eligible: bool = True,
    ) -> bool:
        """Consume one observation and return whether the gate has fired.

        ``eligible`` lets an online controller require a task-agnostic safe
        transition state (for example, idle arms with no object attached).
        Ineligible observations reset the debounce streak even if their visual
        embedding is close to the goal.  The default preserves the standalone
        distance-only behavior.
        """
        if (
            not eligible
            or phase_step < self.min_steps
            or squared_distance > self.threshold
        ):
            self.streak = 0
            return False
        self.streak += 1
        return self.streak >= self.consecutive_steps

    def reset(self) -> None:
        """Clear accumulated evidence before entering another phase."""
        self.streak = 0


def mean_start_embedding(video_embeddings: Iterable[torch.Tensor]) -> torch.Tensor:
    """Return an equally weighted mean start embedding across videos.

    Every input tensor contains the embeddings of the selected initial frames
    from one video and has shape ``(frames, embedding_dim)``. Frames are first
    averaged within each video, then video means are averaged. This prevents a
    short or long video from receiving a different weight.
    """

    video_means: list[torch.Tensor] = []
    embedding_dim: int | None = None

    for embeddings in video_embeddings:
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(
                "Each video must provide a non-empty (frames, embedding_dim) tensor"
            )
        if embedding_dim is None:
            embedding_dim = int(embeddings.shape[1])
        elif embeddings.shape[1] != embedding_dim:
            raise ValueError("All videos must use the same embedding dimension")
        video_means.append(embeddings.mean(dim=0))

    if not video_means:
        raise ValueError("At least one video is required to compute a start embedding")

    return torch.stack(video_means, dim=0).mean(dim=0)


def select_phase_by_nearest_start(
    embeddings_by_phase: Mapping[str, torch.Tensor],
    start_embeddings: Mapping[str, torch.Tensor],
    phase_order: Sequence[str] | None = None,
) -> RawPhaseSelection:
    """Select the nearest phase start using raw squared Euclidean distance.

    ``embeddings_by_phase[name]`` must contain the current observations encoded
    by phase ``name``. No normalization or calibration is applied: this is the
    direct test of the raw-distance routing hypothesis.
    """

    names = tuple(phase_order or embeddings_by_phase.keys())
    if not names:
        raise ValueError("At least one candidate phase is required")
    if len(set(names)) != len(names):
        raise ValueError("Candidate phase names must be unique")

    distance_columns: list[torch.Tensor] = []
    batch_size: int | None = None

    for name in names:
        if name not in embeddings_by_phase:
            raise KeyError(f"Missing current embeddings for phase '{name}'")
        if name not in start_embeddings:
            raise KeyError(f"Missing start embedding for phase '{name}'")

        current = embeddings_by_phase[name]
        start = start_embeddings[name]
        if current.ndim == 1:
            current = current.unsqueeze(0)
        if current.ndim != 2:
            raise ValueError(
                f"Current embeddings for '{name}' must have shape (batch, dim)"
            )
        if start.ndim != 1:
            raise ValueError(f"Start embedding for '{name}' must have shape (dim,)")
        if current.shape[1] != start.shape[0]:
            raise ValueError(
                f"Embedding dimension mismatch for '{name}': "
                f"{current.shape[1]} != {start.shape[0]}"
            )
        if batch_size is None:
            batch_size = int(current.shape[0])
        elif current.shape[0] != batch_size:
            raise ValueError("Every phase encoder must receive the same batch size")

        distance_columns.append(torch.sum((current - start) ** 2, dim=1))

    squared_distances = torch.stack(distance_columns, dim=1)
    selected_indices = torch.argmin(squared_distances, dim=1)
    selected_phases = tuple(names[index] for index in selected_indices.tolist())

    if len(names) == 1:
        margins = torch.full_like(squared_distances[:, 0], float("inf"))
    else:
        two_smallest = torch.topk(
            squared_distances, k=2, dim=1, largest=False
        ).values
        margins = two_smallest[:, 1] - two_smallest[:, 0]

    return RawPhaseSelection(
        phase_names=names,
        selected_indices=selected_indices,
        selected_phases=selected_phases,
        squared_distances=squared_distances,
        margins=margins,
    )
