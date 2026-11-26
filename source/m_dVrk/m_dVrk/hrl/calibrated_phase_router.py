"""Calibrated TCC transition router from Sections 5--9 of the report.

The active phase ``k`` is known.  At every frame the router evaluates all
candidate phases ``j != k`` and forms

    log T(k -> j | x) = log E_k(x) + log P(start_j | x) + log pi(j | k).

There is deliberately no preliminary end-score gate: all three switching
conditions are evaluated on every frame.  A switch is emitted only when the
same candidate passes the total-score threshold, runner-up margin, and
standalone start-evidence floor for ``n`` consecutive frames.

The module has no Isaac Sim dependency.  Calibration and runtime need only
PyTorch, which keeps the fitted artifact usable in standalone tests and in the
full-task evaluator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


_EPS = 1.0e-12


@dataclass(frozen=True)
class CalibratedRouterDataset:
    """Precomputed embeddings and endpoint annotations for whole trajectories.

    Every phase-specific encoder must encode the same ``N`` frames.  The first
    ``L`` and last ``L`` frames of each annotated segment are marked by
    ``start_mask`` and ``end_mask``.  ``start_negative_masks`` may exclude the
    guard frames immediately preceding a phase start from that phase's
    negative examples.
    """

    embeddings_by_phase: Mapping[str, torch.Tensor]
    phase_labels: Sequence[str]
    trajectory_ids: Sequence[str]
    start_mask: torch.Tensor
    end_mask: torch.Tensor
    start_negative_masks: Mapping[str, torch.Tensor] | None = None

    def validate(self, phase_names: Sequence[str] | None = None) -> tuple[str, ...]:
        names = tuple(phase_names or self.embeddings_by_phase.keys())
        if not names or len(names) != len(set(names)):
            raise ValueError("phase_names must contain unique phase identifiers")
        missing = sorted(set(names) - set(self.embeddings_by_phase))
        if missing:
            raise KeyError(f"Missing embeddings for encoders: {missing}")

        count = len(self.phase_labels)
        if count == 0 or len(self.trajectory_ids) != count:
            raise ValueError("phase_labels and trajectory_ids must have equal non-zero length")
        if self.start_mask.shape != (count,) or self.end_mask.shape != (count,):
            raise ValueError("start_mask and end_mask must have shape (N,)")
        if self.start_mask.dtype != torch.bool or self.end_mask.dtype != torch.bool:
            raise ValueError("start_mask and end_mask must be boolean")
        unknown_labels = sorted(set(self.phase_labels) - set(names))
        if unknown_labels:
            raise ValueError(f"Unknown phase labels: {unknown_labels}")

        for name in names:
            values = self.embeddings_by_phase[name]
            if values.ndim != 2 or values.shape[0] != count:
                raise ValueError(f"Embeddings for {name!r} must have shape (N, D)")
            if not torch.isfinite(values).all():
                raise ValueError(f"Embeddings for {name!r} contain non-finite values")
        if self.start_negative_masks is not None:
            missing_masks = sorted(set(names) - set(self.start_negative_masks))
            if missing_masks:
                raise KeyError(f"Missing start-negative masks for: {missing_masks}")
            for name in names:
                mask = self.start_negative_masks[name]
                if mask.shape != (count,) or mask.dtype != torch.bool:
                    raise ValueError(
                        f"start_negative_masks[{name!r}] must be boolean with shape (N,)"
                    )
        return names

    def to_payload(self) -> dict[str, object]:
        return {
            "format": "calibrated_tcc_router_dataset_v1",
            "embeddings_by_phase": {
                name: values.detach().cpu() for name, values in self.embeddings_by_phase.items()
            },
            "phase_labels": list(self.phase_labels),
            "trajectory_ids": list(self.trajectory_ids),
            "start_mask": self.start_mask.detach().cpu(),
            "end_mask": self.end_mask.detach().cpu(),
            "start_negative_masks": (
                None
                if self.start_negative_masks is None
                else {
                    name: mask.detach().cpu()
                    for name, mask in self.start_negative_masks.items()
                }
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CalibratedRouterDataset":
        embeddings = payload.get("embeddings_by_phase")
        if not isinstance(embeddings, Mapping):
            raise ValueError("Dataset payload lacks embeddings_by_phase")
        negative_masks = payload.get("start_negative_masks")
        if negative_masks is not None and not isinstance(negative_masks, Mapping):
            raise ValueError("start_negative_masks must be a mapping or None")
        return cls(
            embeddings_by_phase={
                str(name): torch.as_tensor(values, dtype=torch.float32)
                for name, values in embeddings.items()
            },
            phase_labels=tuple(str(value) for value in payload["phase_labels"]),
            trajectory_ids=tuple(str(value) for value in payload["trajectory_ids"]),
            start_mask=torch.as_tensor(payload["start_mask"], dtype=torch.bool),
            end_mask=torch.as_tensor(payload["end_mask"], dtype=torch.bool),
            start_negative_masks=(
                None
                if negative_masks is None
                else {
                    str(name): torch.as_tensor(mask, dtype=torch.bool)
                    for name, mask in negative_masks.items()
                }
            ),
        )


@dataclass(frozen=True)
class CalibratedRouterArtifact:
    """Serializable calibration and optional sparse transition prior."""

    phase_names: tuple[str, ...]
    goal_prototypes: Mapping[str, torch.Tensor]
    start_prototypes: Mapping[str, torch.Tensor]
    end_scales: Mapping[str, float]
    start_scales: Mapping[str, float]
    start_weights: Mapping[str, float]
    start_biases: Mapping[str, float]
    transition_priors: Mapping[tuple[str, str], float] = field(default_factory=dict)
    transition_counts: Mapping[tuple[str, str], int] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        names = set(self.phase_names)
        if not names or len(names) != len(self.phase_names):
            raise ValueError("Artifact phase names must be non-empty and unique")
        mappings = (
            self.goal_prototypes,
            self.start_prototypes,
            self.end_scales,
            self.start_scales,
            self.start_weights,
            self.start_biases,
        )
        if any(set(values) != names for values in mappings):
            raise ValueError("Every artifact mapping must contain every phase exactly once")
        for name in self.phase_names:
            goal = torch.as_tensor(self.goal_prototypes[name])
            start = torch.as_tensor(self.start_prototypes[name])
            if goal.ndim != 1 or start.shape != goal.shape:
                raise ValueError(f"Prototype shapes are incompatible for {name!r}")
            if not torch.isfinite(goal).all() or not torch.isfinite(start).all():
                raise ValueError(f"Prototype for {name!r} contains non-finite values")
            for label, value in (
                ("end scale", self.end_scales[name]),
                ("start scale", self.start_scales[name]),
            ):
                if not math.isfinite(float(value)) or float(value) <= 0:
                    raise ValueError(f"{label} for {name!r} must be finite and positive")
            if not math.isfinite(float(self.start_weights[name])):
                raise ValueError(f"Start weight for {name!r} must be finite")
            if float(self.start_weights[name]) > 1.0e-8:
                raise ValueError(f"Start weight for {name!r} must be non-positive")
            if not math.isfinite(float(self.start_biases[name])):
                raise ValueError(f"Start bias for {name!r} must be finite")
        for edge, value in self.transition_priors.items():
            _validate_transition_edge(edge, names)
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(
                    f"Positive sparse transition prior required for {edge}: {value}"
                )
        for edge, value in self.transition_counts.items():
            _validate_transition_edge(edge, names)
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(
                    f"Positive integer transition count required for {edge}: {value}"
                )
        for source in self.phase_names:
            row = [
                float(value)
                for (row_source, _), value in self.transition_priors.items()
                if row_source == source
            ]
            if row and not math.isclose(sum(row), 1.0, rel_tol=1.0e-6, abs_tol=1.0e-8):
                raise ValueError(
                    f"Transition priors leaving {source!r} must sum to one"
                )

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "format": "calibrated_tcc_router_v2",
            "phase_names": list(self.phase_names),
            "goal_prototypes": {
                name: value.detach().cpu() for name, value in self.goal_prototypes.items()
            },
            "start_prototypes": {
                name: value.detach().cpu() for name, value in self.start_prototypes.items()
            },
            "end_scales": dict(self.end_scales),
            "start_scales": dict(self.start_scales),
            "start_weights": dict(self.start_weights),
            "start_biases": dict(self.start_biases),
            "transition_priors": _nested_edge_mapping(self.transition_priors),
            "transition_counts": _nested_edge_mapping(self.transition_counts),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CalibratedRouterArtifact":
        artifact_format = payload.get("format")
        if artifact_format not in {
            "calibrated_tcc_router_v1",
            "calibrated_tcc_router_v2",
        }:
            raise ValueError("Unsupported calibrated-router artifact format")

        def tensor_mapping(key: str) -> dict[str, torch.Tensor]:
            values = payload.get(key)
            if not isinstance(values, Mapping):
                raise ValueError(f"Artifact lacks {key}")
            return {
                str(name): torch.as_tensor(value, dtype=torch.float32)
                for name, value in values.items()
            }

        def float_mapping(key: str) -> dict[str, float]:
            values = payload.get(key)
            if not isinstance(values, Mapping):
                raise ValueError(f"Artifact lacks {key}")
            return {str(name): float(value) for name, value in values.items()}

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("Artifact metadata must be a mapping")
        artifact = cls(
            phase_names=tuple(str(value) for value in payload["phase_names"]),
            goal_prototypes=tensor_mapping("goal_prototypes"),
            start_prototypes=tensor_mapping("start_prototypes"),
            end_scales=float_mapping("end_scales"),
            start_scales=float_mapping("start_scales"),
            start_weights=float_mapping("start_weights"),
            start_biases=float_mapping("start_biases"),
            transition_priors=(
                {}
                if artifact_format == "calibrated_tcc_router_v1"
                else _edge_mapping_from_nested(payload.get("transition_priors", {}), float)
            ),
            transition_counts=(
                {}
                if artifact_format == "calibrated_tcc_router_v1"
                else _edge_mapping_from_nested(payload.get("transition_counts", {}), int)
            ),
            metadata=dict(metadata),
        )
        artifact.validate()
        return artifact


@dataclass(frozen=True)
class CalibratedRouterConfig:
    """Decision parameters selected on validation trajectories."""

    transition_thresholds: Mapping[str, float]
    margin: float = 0.5
    start_log_threshold: float = math.log(0.5)
    consecutive_steps: int = 3
    start_log_thresholds: Mapping[str, float] = field(default_factory=dict)
    consecutive_steps_by_phase: Mapping[str, int] = field(default_factory=dict)

    def validate(self, phase_names: Sequence[str]) -> None:
        unknown = sorted(set(self.transition_thresholds) - set(phase_names))
        if unknown:
            raise ValueError(f"Transition thresholds contain unknown phases: {unknown}")
        if not all(math.isfinite(float(value)) for value in self.transition_thresholds.values()):
            raise ValueError("Transition thresholds must be finite")
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError("margin must be finite and non-negative")
        if not math.isfinite(self.start_log_threshold) or self.start_log_threshold > 0:
            raise ValueError("start_log_threshold must be finite and no greater than zero")
        if self.consecutive_steps <= 0:
            raise ValueError("consecutive_steps must be positive")
        unknown_start = sorted(set(self.start_log_thresholds) - set(phase_names))
        unknown_consecutive = sorted(
            set(self.consecutive_steps_by_phase) - set(phase_names)
        )
        if unknown_start or unknown_consecutive:
            raise ValueError(
                "Per-phase router parameters contain unknown phases: "
                f"{sorted(set(unknown_start + unknown_consecutive))}"
            )
        if any(
            not math.isfinite(float(value)) or float(value) > 0
            for value in self.start_log_thresholds.values()
        ):
            raise ValueError("Per-phase start thresholds must be finite and <= 0")
        if any(int(value) <= 0 for value in self.consecutive_steps_by_phase.values()):
            raise ValueError("Per-phase consecutive-step counts must be positive")


@dataclass(frozen=True)
class CalibratedRouterEvidence:
    """Per-frame score decomposition and the three report conditions."""

    candidate_phases: tuple[str, ...]
    end_log_scores: torch.Tensor
    start_log_probabilities: torch.Tensor
    transition_log_scores: torch.Tensor
    selected_indices: torch.Tensor
    selected_phases: tuple[str, ...]
    runner_up_indices: torch.Tensor
    runner_up_phases: tuple[str | None, ...]
    margins: torch.Tensor
    passes_total_threshold: torch.Tensor
    passes_margin: torch.Tensor
    passes_start_floor: torch.Tensor
    supports_transition: torch.Tensor
    end_score_mode: str = "prototype"
    end_probabilities: torch.Tensor | None = None
    end_progress: torch.Tensor | None = None
    end_within_reference_variance: torch.Tensor | None = None
    end_between_reference_variance: torch.Tensor | None = None
    end_total_variance: torch.Tensor | None = None


@dataclass(frozen=True)
class CalibratedRouterDecision:
    decisions: tuple[str, ...]
    selected_phases: tuple[str | None, ...]
    confirmation_streaks: torch.Tensor
    evidence: CalibratedRouterEvidence


@dataclass(frozen=True)
class HandoffTrajectory:
    """Embeddings around one ground-truth pre-parking phase boundary."""

    source_embeddings: torch.Tensor
    target_embeddings: torch.Tensor
    boundary_index: int

    def validate(self) -> None:
        source = torch.as_tensor(self.source_embeddings)
        target = torch.as_tensor(self.target_embeddings)
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("Handoff embeddings must have shape (T, D)")
        if source.shape[0] != target.shape[0] or source.shape[0] == 0:
            raise ValueError("Source and target handoff trajectories must align")
        if not 0 <= int(self.boundary_index) < source.shape[0]:
            raise ValueError("boundary_index lies outside the handoff trajectory")
        if not torch.isfinite(source).all() or not torch.isfinite(target).all():
            raise ValueError("Handoff embeddings contain non-finite values")


@dataclass(frozen=True)
class HandoffCalibrationStats:
    trajectories: int
    endpoint_frames: int
    end_positive_frames: int
    start_positive_frames: int
    start_negative_frames: int
    source_phase: str
    target_phase: str
    end_prototype_frames: int | None = None
    end_scale_frames: int | None = None
    start_prototype_frames: int | None = None
    start_scale_frames: int | None = None
    end_scale_positive_frames: int | None = None
    start_prototype_positive_frames: int | None = None


@dataclass(frozen=True)
class SoftAlignmentReferenceBank:
    """Variable-length phase references for TCC soft temporal alignment."""

    phase_names: tuple[str, ...]
    references_by_phase: Mapping[str, Sequence[torch.Tensor]]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.phase_names or len(set(self.phase_names)) != len(self.phase_names):
            raise ValueError("Reference-bank phase names must be non-empty and unique")
        if set(self.references_by_phase) != set(self.phase_names):
            raise ValueError("Reference bank must contain every declared phase exactly once")
        for phase in self.phase_names:
            references = tuple(self.references_by_phase[phase])
            if not references:
                raise ValueError(f"Phase {phase!r} needs at least one temporal reference")
            dimensions: set[int] = set()
            for index, raw_reference in enumerate(references):
                reference = torch.as_tensor(raw_reference)
                if reference.ndim != 2 or reference.shape[0] < 2:
                    raise ValueError(
                        f"Reference {index} for {phase!r} must have shape (M, D), M >= 2"
                    )
                if not torch.isfinite(reference).all():
                    raise ValueError(f"Reference {index} for {phase!r} is not finite")
                dimensions.add(int(reference.shape[1]))
            if len(dimensions) != 1:
                raise ValueError(f"References for {phase!r} have inconsistent dimensions")

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "format": "tcc_soft_alignment_reference_bank_v1",
            "phase_names": list(self.phase_names),
            "references_by_phase": {
                phase: [torch.as_tensor(value).detach().cpu().float() for value in values]
                for phase, values in self.references_by_phase.items()
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "SoftAlignmentReferenceBank":
        if payload.get("format") != "tcc_soft_alignment_reference_bank_v1":
            raise ValueError("Unsupported soft-alignment reference-bank format")
        raw_references = payload.get("references_by_phase")
        if not isinstance(raw_references, Mapping):
            raise ValueError("Reference-bank payload lacks references_by_phase")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("Reference-bank metadata must be a mapping")
        bank = cls(
            phase_names=tuple(str(value) for value in payload["phase_names"]),
            references_by_phase={
                str(phase): tuple(
                    torch.as_tensor(reference, dtype=torch.float32)
                    for reference in references
                )
                for phase, references in raw_references.items()
            },
            metadata=dict(metadata),
        )
        bank.validate()
        return bank


@dataclass(frozen=True)
class SoftAlignmentEndConfig:
    """Parameters of the training-free temporal soft-matching end score."""

    temperature: float = 1.0
    end_region_mode: str = "fraction"
    end_threshold: float = 0.9
    end_last_frames: int = 3
    end_last_seconds: float = 1.0
    reference_capture_hz: float = 2.0
    epsilon: float = 1.0e-12

    def validate(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Soft-alignment temperature must be finite and positive")
        if self.end_region_mode not in {"fraction", "last_frames", "last_seconds"}:
            raise ValueError(
                "Soft-alignment end region mode must be fraction, last_frames, "
                "or last_seconds"
            )
        if not math.isfinite(self.end_threshold) or not 0.0 <= self.end_threshold < 1.0:
            raise ValueError("Soft-alignment end threshold must lie in [0, 1)")
        if self.end_last_frames <= 0:
            raise ValueError("Soft-alignment end_last_frames must be positive")
        if not math.isfinite(self.end_last_seconds) or self.end_last_seconds <= 0:
            raise ValueError("Soft-alignment end_last_seconds must be finite and positive")
        if not math.isfinite(self.reference_capture_hz) or self.reference_capture_hz <= 0:
            raise ValueError("Soft-alignment reference_capture_hz must be finite and positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Soft-alignment epsilon must be finite and positive")

    def end_frame_count(self, reference_length: int) -> int | None:
        """Return the end-window size, or ``None`` for normalized-time mode."""

        if reference_length < 2:
            raise ValueError("Soft-alignment references need at least two frames")
        if self.end_region_mode == "fraction":
            return None
        if self.end_region_mode == "last_frames":
            requested = self.end_last_frames
        else:
            requested = math.ceil(self.end_last_seconds * self.reference_capture_hz)
        return min(int(requested), reference_length)


@dataclass(frozen=True)
class SoftAlignmentEndEvidence:
    """End probability, normalized progress, and temporal ambiguity."""

    end_probabilities: torch.Tensor
    log_end_probabilities: torch.Tensor
    progress: torch.Tensor
    within_reference_variance: torch.Tensor
    between_reference_variance: torch.Tensor
    total_variance: torch.Tensor
    per_reference_end_probabilities: torch.Tensor
    per_reference_progress: torch.Tensor
    per_reference_variance: torch.Tensor


def soft_alignment_end_evidence(
    query_embeddings: torch.Tensor,
    references: Sequence[torch.Tensor],
    config: SoftAlignmentEndConfig | None = None,
) -> SoftAlignmentEndEvidence:
    """Soft-match a query batch to complete references on normalized time.

    Each reference contributes equally, independently of its number of frames.
    ``total_variance`` combines ambiguity inside each alignment distribution
    with disagreement between the per-reference progress estimates.
    """

    config = config or SoftAlignmentEndConfig()
    config.validate()
    query = torch.as_tensor(query_embeddings, dtype=torch.float32)
    if query.ndim == 1:
        query = query.unsqueeze(0)
    if query.ndim != 2 or query.shape[0] == 0:
        raise ValueError("Soft-alignment query must have shape (B, D) with B > 0")
    if not torch.isfinite(query).all():
        raise ValueError("Soft-alignment query contains non-finite values")
    if not references:
        raise ValueError("Soft alignment needs at least one reference")

    end_columns: list[torch.Tensor] = []
    progress_columns: list[torch.Tensor] = []
    variance_columns: list[torch.Tensor] = []
    for index, raw_reference in enumerate(references):
        reference = torch.as_tensor(
            raw_reference, dtype=query.dtype, device=query.device
        )
        if reference.ndim != 2 or reference.shape[0] < 2:
            raise ValueError(f"Reference {index} must have shape (M, D), M >= 2")
        if reference.shape[1] != query.shape[1]:
            raise ValueError(
                f"Reference {index} dimension {reference.shape[1]} != query {query.shape[1]}"
            )
        if not torch.isfinite(reference).all():
            raise ValueError(f"Reference {index} contains non-finite values")

        squared_distances = torch.sum(
            (query[:, None, :] - reference[None, :, :]) ** 2, dim=2
        )
        weights = torch.softmax(-squared_distances / config.temperature, dim=1)
        normalized_time = torch.linspace(
            0.0, 1.0, reference.shape[0], dtype=query.dtype, device=query.device
        )
        end_frame_count = config.end_frame_count(int(reference.shape[0]))
        if end_frame_count is None:
            end_mask = normalized_time > config.end_threshold
        else:
            end_mask = torch.zeros(
                reference.shape[0], dtype=torch.bool, device=query.device
            )
            end_mask[-end_frame_count:] = True
        if not bool(end_mask.any()):
            raise ValueError(
                f"End-region configuration selects no frames in reference {index}"
            )
        progress = torch.sum(weights * normalized_time[None, :], dim=1)
        variance = torch.sum(
            weights * (normalized_time[None, :] - progress[:, None]) ** 2, dim=1
        )
        end_columns.append(torch.sum(weights[:, end_mask], dim=1))
        progress_columns.append(progress)
        variance_columns.append(variance)

    per_reference_end = torch.stack(end_columns, dim=1)
    per_reference_progress = torch.stack(progress_columns, dim=1)
    per_reference_variance = torch.stack(variance_columns, dim=1)
    end_probabilities = per_reference_end.mean(dim=1)
    progress = per_reference_progress.mean(dim=1)
    within_variance = per_reference_variance.mean(dim=1)
    between_variance = torch.mean(
        (per_reference_progress - progress[:, None]) ** 2, dim=1
    )
    total_variance = within_variance + between_variance
    return SoftAlignmentEndEvidence(
        end_probabilities=end_probabilities,
        log_end_probabilities=torch.log(
            torch.clamp(end_probabilities, min=config.epsilon)
        ),
        progress=progress,
        within_reference_variance=within_variance,
        between_reference_variance=between_variance,
        total_variance=total_variance,
        per_reference_end_probabilities=per_reference_end,
        per_reference_progress=per_reference_progress,
        per_reference_variance=per_reference_variance,
    )


def _validate_transition_edge(edge: tuple[str, str], phase_names: set[str]) -> None:
    if not isinstance(edge, tuple) or len(edge) != 2:
        raise ValueError(f"Transition edge must be a (source, target) tuple: {edge!r}")
    source, target = edge
    if source not in phase_names or target not in phase_names:
        raise ValueError(f"Transition edge has an unknown phase: {edge}")
    if source == target:
        raise ValueError(f"Self-transition is not a routing edge: {edge}")


def _nested_edge_mapping(
    values: Mapping[tuple[str, str], float | int],
) -> dict[str, dict[str, float | int]]:
    nested: dict[str, dict[str, float | int]] = {}
    for (source, target), value in values.items():
        nested.setdefault(source, {})[target] = value
    return nested


def _edge_mapping_from_nested(
    values: object, conversion: type[float] | type[int]
) -> dict[tuple[str, str], float | int]:
    if not isinstance(values, Mapping):
        raise ValueError("Transition edge payload must be a nested mapping")
    edges: dict[tuple[str, str], float | int] = {}
    for source, row in values.items():
        if not isinstance(row, Mapping):
            raise ValueError(f"Transition-prior row for {source!r} must be a mapping")
        for target, value in row.items():
            edges[(str(source), str(target))] = conversion(value)
    return edges


def transition_priors_from_counts(
    transition_counts: Mapping[tuple[str, str], int],
    phase_names: Sequence[str],
) -> dict[tuple[str, str], float]:
    """Normalize annotated transition counts into a sparse Markov prior.

    Missing edges have probability zero. A phase with no annotated outgoing
    transition has no row, which is appropriate for a terminal phase.
    """

    names = set(phase_names)
    row_totals: dict[str, int] = {}
    cleaned: dict[tuple[str, str], int] = {}
    for edge, raw_count in transition_counts.items():
        _validate_transition_edge(edge, names)
        if isinstance(raw_count, bool) or int(raw_count) != raw_count:
            raise ValueError(f"Transition count must be an integer for {edge}")
        count = int(raw_count)
        if count < 0:
            raise ValueError(f"Transition count must be non-negative for {edge}")
        if count == 0:
            continue
        cleaned[edge] = cleaned.get(edge, 0) + count
        row_totals[edge[0]] = row_totals.get(edge[0], 0) + count
    return {
        edge: count / row_totals[edge[0]] for edge, count in cleaned.items()
    }


def normalize_transition_priors(
    transition_priors: Mapping[tuple[str, str], float],
    phase_names: Sequence[str],
) -> dict[tuple[str, str], float]:
    """Normalize non-negative sparse weights independently for each source."""

    names = set(phase_names)
    row_totals: dict[str, float] = {}
    cleaned: dict[tuple[str, str], float] = {}
    for edge, raw_value in transition_priors.items():
        _validate_transition_edge(edge, names)
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"Transition-prior weight must be finite and non-negative: {edge}"
            )
        if value == 0:
            continue
        cleaned[edge] = value
        row_totals[edge[0]] = row_totals.get(edge[0], 0.0) + value
    return {
        edge: value / row_totals[edge[0]] for edge, value in cleaned.items()
    }


def _phase_mask(labels: Sequence[str], phase: str) -> torch.Tensor:
    return torch.tensor([label == phase for label in labels], dtype=torch.bool)


def _endpoint_prototype_and_scale(
    values: torch.Tensor,
    prototype_mask: torch.Tensor,
    trajectory_ids: Sequence[str],
    min_scale: float,
    *,
    scale_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Pool prototype frames and estimate scale leave-one-trajectory-out."""

    if scale_mask is None:
        scale_mask = prototype_mask
    selected = values[prototype_mask]
    if not selected.numel():
        raise ValueError("Every phase needs at least one annotated endpoint frame")
    if not values[scale_mask].numel():
        raise ValueError("Every phase needs at least one endpoint scale frame")
    prototype = selected.mean(dim=0)
    endpoint_trajectories = tuple(
        dict.fromkeys(
            str(trajectory_ids[index])
            for index in scale_mask.nonzero(as_tuple=False).flatten().tolist()
        )
    )
    if len(endpoint_trajectories) < 2:
        raise ValueError("Leave-one-trajectory-out scale needs at least two trajectories")

    distances: list[torch.Tensor] = []
    for trajectory in endpoint_trajectories:
        trajectory_mask = torch.tensor(
            [str(value) == trajectory for value in trajectory_ids], dtype=torch.bool
        )
        held_out = scale_mask & trajectory_mask
        remaining = prototype_mask & ~trajectory_mask
        if not remaining.any():
            continue
        loo_prototype = values[remaining].mean(dim=0)
        distances.append(torch.sum((values[held_out] - loo_prototype) ** 2, dim=1))
    if not distances:
        raise ValueError("Could not compute leave-one-trajectory-out endpoint distances")
    scale = max(float(torch.median(torch.cat(distances)).item()), min_scale)
    return prototype, scale


def _fit_monotone_logistic(
    distances: torch.Tensor,
    labels: torch.Tensor,
    l2: float,
) -> tuple[float, float]:
    """Fit sigmoid(weight * distance + bias), constraining weight <= 0."""

    x = distances.detach().to(dtype=torch.float64, device="cpu")
    y = labels.detach().to(dtype=torch.float64, device="cpu")
    positives = y > 0.5
    positive_count = int(positives.sum().item())
    negative_count = int((~positives).sum().item())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Each start detector needs positive and negative frames")
    sample_weights = torch.where(
        positives,
        torch.full_like(y, 0.5 / positive_count),
        torch.full_like(y, 0.5 / negative_count),
    )

    center = torch.median(x)
    spread = torch.quantile(torch.abs(x - center), 0.75).clamp_min(1.0e-6)
    normalized = (x - center) / spread
    design = torch.stack((-normalized, torch.ones_like(normalized)), dim=1)
    parameters = torch.tensor([1.0, 0.0], dtype=torch.float64)
    for _ in range(60):
        logits = design @ parameters
        probabilities = torch.sigmoid(logits)
        gradient = design.T @ (sample_weights * (probabilities - y))
        gradient[0] += l2 * parameters[0]
        curvature = sample_weights * probabilities * (1.0 - probabilities)
        hessian = design.T @ (curvature[:, None] * design)
        hessian[0, 0] += l2
        hessian += torch.eye(2, dtype=torch.float64) * 1.0e-8
        updated = parameters - torch.linalg.solve(hessian, gradient)
        updated[0] = updated[0].clamp_min(0.0)
        if torch.max(torch.abs(updated - parameters)) < 1.0e-9:
            parameters = updated
            break
        parameters = updated
    weight = -parameters[0] / spread
    bias = parameters[1] + parameters[0] * center / spread
    return float(weight.item()), float(bias.item())


def fit_calibrated_router(
    data: CalibratedRouterDataset,
    phase_names: Sequence[str] | None = None,
    *,
    min_scale: float = 1.0e-8,
    logistic_l2: float = 1.0e-3,
    transition_counts: Mapping[tuple[str, str], int] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> CalibratedRouterArtifact:
    """Fit prototypes, LOTO scales, and class-balanced start detectors."""

    if min_scale <= 0 or not math.isfinite(min_scale):
        raise ValueError("min_scale must be finite and positive")
    if logistic_l2 < 0 or not math.isfinite(logistic_l2):
        raise ValueError("logistic_l2 must be finite and non-negative")
    names = data.validate(phase_names)
    counts = dict(transition_counts or {})
    priors = transition_priors_from_counts(counts, names)
    embeddings = {
        name: data.embeddings_by_phase[name].detach().cpu().float() for name in names
    }
    start_mask = data.start_mask.detach().cpu()
    end_mask = data.end_mask.detach().cpu()

    goals: dict[str, torch.Tensor] = {}
    starts: dict[str, torch.Tensor] = {}
    end_scales: dict[str, float] = {}
    start_scales: dict[str, float] = {}
    weights: dict[str, float] = {}
    biases: dict[str, float] = {}

    for phase in names:
        own_phase = _phase_mask(data.phase_labels, phase)
        positive_start = own_phase & start_mask
        positive_end = own_phase & end_mask
        start, start_scale = _endpoint_prototype_and_scale(
            embeddings[phase], positive_start, data.trajectory_ids, min_scale
        )
        goal, end_scale = _endpoint_prototype_and_scale(
            embeddings[phase], positive_end, data.trajectory_ids, min_scale
        )
        starts[phase] = start
        goals[phase] = goal
        start_scales[phase] = start_scale
        end_scales[phase] = end_scale

        allowed_negatives = (
            torch.ones_like(own_phase)
            if data.start_negative_masks is None
            else data.start_negative_masks[phase].detach().cpu()
        )
        negative = (~own_phase) & allowed_negatives
        detector_rows = positive_start | negative
        distances = (
            torch.sum((embeddings[phase] - start) ** 2, dim=1) / start_scale
        )
        weight, bias = _fit_monotone_logistic(
            distances[detector_rows], positive_start[detector_rows].float(), logistic_l2
        )
        weights[phase] = weight
        biases[phase] = bias

    artifact = CalibratedRouterArtifact(
        phase_names=names,
        goal_prototypes=goals,
        start_prototypes=starts,
        end_scales=end_scales,
        start_scales=start_scales,
        start_weights=weights,
        start_biases=biases,
        transition_priors=priors,
        transition_counts={
            edge: int(value) for edge, value in counts.items() if int(value) > 0
        },
        metadata={
            "fit_trajectories": len(set(str(value) for value in data.trajectory_ids)),
            "min_scale": min_scale,
            "logistic_l2": logistic_l2,
            **dict(metadata or {}),
        },
    )
    artifact.validate()
    return artifact


def fit_handoff_calibration(
    base_artifact: CalibratedRouterArtifact,
    trajectories: Mapping[str, HandoffTrajectory],
    source_phase: str,
    target_phase: str,
    *,
    endpoint_frames: int,
    guard_frames: int,
    end_prototype_frames: int | None = None,
    end_scale_frames: int | None = None,
    start_prototype_frames: int | None = None,
    start_scale_frames: int | None = None,
    min_scale: float = 1.0e-8,
    logistic_l2: float = 1.0e-3,
    metadata: Mapping[str, object] | None = None,
) -> tuple[CalibratedRouterArtifact, HandoffCalibrationStats]:
    """Recalibrate one transition from policy-produced pre-parking boundaries.

    By default, ``endpoint_frames`` controls all prototype and scale windows.
    The four optional window arguments decouple the source goal prototype,
    source end scale, target start prototype and target start scale. Earlier
    source-policy frames, excluding a guard window, are the deployment-relevant
    negatives for the target start detector. TCC encoders and every unrelated
    artifact component remain frozen.
    """

    base_artifact.validate()
    if source_phase == target_phase:
        raise ValueError("A handoff must connect two different phases")
    names = set(base_artifact.phase_names)
    _validate_transition_edge((source_phase, target_phase), names)
    if endpoint_frames <= 0:
        raise ValueError("endpoint_frames must be positive")
    end_prototype_frames = int(end_prototype_frames or endpoint_frames)
    end_scale_frames = int(end_scale_frames or endpoint_frames)
    start_prototype_frames = int(start_prototype_frames or endpoint_frames)
    start_scale_frames = int(start_scale_frames or endpoint_frames)
    calibration_windows = {
        "end_prototype_frames": end_prototype_frames,
        "end_scale_frames": end_scale_frames,
        "start_prototype_frames": start_prototype_frames,
        "start_scale_frames": start_scale_frames,
    }
    if any(value <= 0 for value in calibration_windows.values()):
        raise ValueError("Handoff calibration window lengths must be positive")
    if guard_frames < 0:
        raise ValueError("guard_frames cannot be negative")
    if min_scale <= 0 or not math.isfinite(min_scale):
        raise ValueError("min_scale must be finite and positive")
    if logistic_l2 < 0 or not math.isfinite(logistic_l2):
        raise ValueError("logistic_l2 must be finite and non-negative")
    if len(trajectories) < 2:
        raise ValueError("Handoff calibration needs at least two trajectories")

    source_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    trajectory_ids: list[str] = []
    end_prototype_mask_parts: list[torch.Tensor] = []
    end_scale_mask_parts: list[torch.Tensor] = []
    start_prototype_mask_parts: list[torch.Tensor] = []
    start_scale_mask_parts: list[torch.Tensor] = []
    negative_mask_parts: list[torch.Tensor] = []
    for trajectory_id, trajectory in trajectories.items():
        trajectory.validate()
        source = trajectory.source_embeddings.detach().cpu().float()
        target = trajectory.target_embeddings.detach().cpu().float()
        boundary = int(trajectory.boundary_index)
        if source.shape[1] != int(base_artifact.goal_prototypes[source_phase].numel()):
            raise ValueError(f"Source embedding dimension mismatch in {trajectory_id!r}")
        if target.shape[1] != int(base_artifact.start_prototypes[target_phase].numel()):
            raise ValueError(f"Target embedding dimension mismatch in {trajectory_id!r}")
        end_begin = boundary - max(end_prototype_frames, end_scale_frames) + 1
        start_end = boundary + max(start_prototype_frames, start_scale_frames)
        if end_begin < 0 or start_end > source.shape[0]:
            raise ValueError(
                f"Trajectory {trajectory_id!r} lacks the requested calibration "
                f"windows {calibration_windows} around its boundary"
            )
        count = source.shape[0]
        indices = torch.arange(count)
        source_rows.append(source)
        target_rows.append(target)
        trajectory_ids.extend([str(trajectory_id)] * count)
        end_prototype_mask_parts.append(
            (indices >= boundary - end_prototype_frames + 1)
            & (indices <= boundary)
        )
        end_scale_mask_parts.append(
            (indices >= boundary - end_scale_frames + 1)
            & (indices <= boundary)
        )
        start_prototype_mask_parts.append(
            (indices >= boundary) & (indices < boundary + start_prototype_frames)
        )
        start_scale_mask_parts.append(
            (indices >= boundary) & (indices < boundary + start_scale_frames)
        )
        negative_mask_parts.append(indices < max(boundary - guard_frames, 0))

    source_values = torch.cat(source_rows)
    target_values = torch.cat(target_rows)
    end_prototype_mask = torch.cat(end_prototype_mask_parts)
    end_scale_mask = torch.cat(end_scale_mask_parts)
    start_prototype_mask = torch.cat(start_prototype_mask_parts)
    start_scale_mask = torch.cat(start_scale_mask_parts)
    negative_mask = torch.cat(negative_mask_parts)
    if bool((start_scale_mask & negative_mask).any()):
        raise RuntimeError("Positive and negative handoff masks overlap")

    goal, end_scale = _endpoint_prototype_and_scale(
        source_values,
        end_prototype_mask,
        trajectory_ids,
        min_scale,
        scale_mask=end_scale_mask,
    )
    start, start_scale = _endpoint_prototype_and_scale(
        target_values,
        start_prototype_mask,
        trajectory_ids,
        min_scale,
        scale_mask=start_scale_mask,
    )
    distances = torch.sum((target_values - start) ** 2, dim=1) / start_scale
    detector_rows = start_scale_mask | negative_mask
    weight, bias = _fit_monotone_logistic(
        distances[detector_rows],
        start_scale_mask[detector_rows].float(),
        logistic_l2,
    )

    goals = dict(base_artifact.goal_prototypes)
    starts = dict(base_artifact.start_prototypes)
    end_scales = dict(base_artifact.end_scales)
    start_scales = dict(base_artifact.start_scales)
    weights = dict(base_artifact.start_weights)
    biases = dict(base_artifact.start_biases)
    goals[source_phase] = goal
    starts[target_phase] = start
    end_scales[source_phase] = end_scale
    start_scales[target_phase] = start_scale
    weights[target_phase] = weight
    biases[target_phase] = bias

    updated_metadata = dict(base_artifact.metadata)
    updated_metadata["handoff_calibration"] = {
        "source_phase": source_phase,
        "target_phase": target_phase,
        "trajectory_ids": sorted(str(value) for value in trajectories),
        "endpoint_frames": endpoint_frames,
        **calibration_windows,
        "guard_frames": guard_frames,
        "min_scale": min_scale,
        "logistic_l2": logistic_l2,
        **dict(metadata or {}),
    }
    artifact = CalibratedRouterArtifact(
        phase_names=base_artifact.phase_names,
        goal_prototypes=goals,
        start_prototypes=starts,
        end_scales=end_scales,
        start_scales=start_scales,
        start_weights=weights,
        start_biases=biases,
        transition_priors=dict(base_artifact.transition_priors),
        transition_counts=dict(base_artifact.transition_counts),
        metadata=updated_metadata,
    )
    artifact.validate()
    stats = HandoffCalibrationStats(
        trajectories=len(trajectories),
        endpoint_frames=endpoint_frames,
        end_positive_frames=int(end_prototype_mask.sum().item()),
        start_positive_frames=int(start_scale_mask.sum().item()),
        start_negative_frames=int(negative_mask.sum().item()),
        source_phase=source_phase,
        target_phase=target_phase,
        end_prototype_frames=end_prototype_frames,
        end_scale_frames=end_scale_frames,
        start_prototype_frames=start_prototype_frames,
        start_scale_frames=start_scale_frames,
        end_scale_positive_frames=int(end_scale_mask.sum().item()),
        start_prototype_positive_frames=int(start_prototype_mask.sum().item()),
    )
    return artifact, stats


def replay_handoff_trajectories(
    artifact: CalibratedRouterArtifact,
    trajectories: Mapping[str, HandoffTrajectory],
    source_phase: str,
    target_phase: str,
    *,
    transition_threshold: float,
    start_probability_threshold: float,
    consecutive_steps: int,
    timing_tolerance: int,
    earliest_allowed_delay: int | None = None,
    end_reference_bank: SoftAlignmentReferenceBank | None = None,
    soft_end_config: SoftAlignmentEndConfig | None = None,
) -> dict[str, object]:
    """Evaluate one sparse-prior handoff against held-out symbolic boundaries."""

    if not 0.0 < start_probability_threshold <= 1.0:
        raise ValueError("start_probability_threshold must be in (0, 1]")
    if consecutive_steps <= 0 or timing_tolerance < 0:
        raise ValueError("consecutive_steps must be positive and tolerance non-negative")
    router = CalibratedPhaseRouter(
        artifact,
        CalibratedRouterConfig(
            transition_thresholds={source_phase: transition_threshold},
            margin=0.0,
            start_log_threshold=math.log(start_probability_threshold),
            consecutive_steps=consecutive_steps,
        ),
        transition_priors={(source_phase, target_phase): 1.0},
        end_reference_bank=end_reference_bank,
        soft_end_config=soft_end_config,
    )
    rows: list[dict[str, object]] = []
    timing_hits = 0
    early_switches = 0
    late_switches = 0
    misses = 0
    for trajectory_id, trajectory in trajectories.items():
        trajectory.validate()
        router.reset(1)
        switch_index: int | None = None
        supported_frames = 0
        longest_streak = 0
        for index in range(trajectory.source_embeddings.shape[0]):
            decision = router.update(
                source_phase,
                {
                    source_phase: trajectory.source_embeddings[index],
                    target_phase: trajectory.target_embeddings[index],
                },
                candidate_phases=(target_phase,),
            )
            if bool(decision.evidence.supports_transition[0]):
                supported_frames += 1
            longest_streak = max(
                longest_streak, int(decision.confirmation_streaks[0].item())
            )
            if decision.decisions[0] == "switch":
                switch_index = index
                break
        boundary = int(trajectory.boundary_index)
        delay = None if switch_index is None else switch_index - boundary
        earliest = (
            -timing_tolerance
            if earliest_allowed_delay is None
            else int(earliest_allowed_delay)
        )
        if switch_index is None:
            outcome = "miss"
            misses += 1
        elif earliest <= delay <= timing_tolerance:
            outcome = "timing_hit"
            timing_hits += 1
        elif delay < earliest:
            outcome = "early_switch"
            early_switches += 1
        else:
            outcome = "late_switch"
            late_switches += 1
        rows.append(
            {
                "trajectory": str(trajectory_id),
                "boundary_index": boundary,
                "switch_index": switch_index,
                "delay": delay,
                "outcome": outcome,
                "supported_frames": supported_frames,
                "longest_streak": longest_streak,
            }
        )
    count = len(rows)
    return {
        "trajectories": count,
        "timing_hits": timing_hits,
        "early_switches": early_switches,
        "late_switches": late_switches,
        "misses": misses,
        "timing_hit_rate": timing_hits / max(count, 1),
        "miss_rate": misses / max(count, 1),
        "rows": rows,
    }


class CalibratedPhaseRouter:
    """Stateful report router with three score conditions and debounce."""

    def __init__(
        self,
        artifact: CalibratedRouterArtifact,
        config: CalibratedRouterConfig,
        transition_priors: Mapping[tuple[str, str], float] | None = None,
        end_reference_bank: SoftAlignmentReferenceBank | None = None,
        soft_end_config: SoftAlignmentEndConfig | None = None,
    ) -> None:
        artifact.validate()
        config.validate(artifact.phase_names)
        self.artifact = artifact
        self.config = config
        raw_priors = (
            artifact.transition_priors
            if transition_priors is None
            else transition_priors
        )
        self.transition_priors = normalize_transition_priors(
            raw_priors, artifact.phase_names
        )
        if end_reference_bank is not None:
            end_reference_bank.validate()
            unknown_reference_phases = sorted(
                set(end_reference_bank.phase_names) - set(artifact.phase_names)
            )
            if unknown_reference_phases:
                raise ValueError(
                    "Reference bank contains unknown phases: "
                    f"{unknown_reference_phases}"
                )
            for phase in end_reference_bank.phase_names:
                expected = int(artifact.goal_prototypes[phase].numel())
                actual = int(
                    torch.as_tensor(end_reference_bank.references_by_phase[phase][0]).shape[1]
                )
                if actual != expected:
                    raise ValueError(
                        f"Reference embedding dimension mismatch for {phase!r}: "
                        f"{actual} != {expected}"
                    )
        if soft_end_config is not None:
            soft_end_config.validate()
        self.end_reference_bank = end_reference_bank
        self.soft_end_config = soft_end_config or SoftAlignmentEndConfig()
        self._pending: list[str | None] = []
        self._streaks = torch.zeros(0, dtype=torch.long)

    def reset(self, batch_size: int = 0) -> None:
        self._pending = [None] * batch_size
        self._streaks = torch.zeros(batch_size, dtype=torch.long)

    @staticmethod
    def _as_batch(value: torch.Tensor, phase: str) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2:
            raise ValueError(f"Embedding for {phase!r} must have shape (B, D)")
        return value

    def score(
        self,
        current_phase: str,
        embeddings_by_phase: Mapping[str, torch.Tensor],
        candidate_phases: Sequence[str] | None = None,
    ) -> CalibratedRouterEvidence:
        """Evaluate end, start, and transition scores for every candidate."""

        if current_phase not in self.artifact.phase_names:
            raise ValueError(f"Unknown current phase: {current_phase}")
        if current_phase not in self.config.transition_thresholds:
            raise ValueError(f"Missing transition threshold for {current_phase}")
        requested_candidates = tuple(
            candidate_phases
            or (name for name in self.artifact.phase_names if name != current_phase)
        )
        if not requested_candidates or len(requested_candidates) != len(set(requested_candidates)):
            raise ValueError("candidate_phases must be non-empty and unique")
        if current_phase in requested_candidates:
            raise ValueError("The active phase cannot be a transition candidate")
        unknown = sorted(set(requested_candidates) - set(self.artifact.phase_names))
        if unknown:
            raise ValueError(f"Unknown candidate phases: {unknown}")
        prior_row = {
            target: probability
            for (source, target), probability in self.transition_priors.items()
            if source == current_phase
        }
        candidates = (
            tuple(name for name in requested_candidates if name in prior_row)
            if prior_row
            else requested_candidates
        )
        if not candidates:
            raise ValueError(
                f"Sparse transition prior leaves no candidate from {current_phase!r}"
            )

        required = {current_phase, *candidates}
        missing = sorted(required - set(embeddings_by_phase))
        if missing:
            raise KeyError(f"Missing embeddings for phases: {missing}")
        batches = {
            name: self._as_batch(embeddings_by_phase[name], name) for name in required
        }
        sizes = {int(value.shape[0]) for value in batches.values()}
        devices = {value.device for value in batches.values()}
        if len(sizes) != 1:
            raise ValueError("Every encoder embedding must have the same batch size")
        if len(devices) != 1:
            raise ValueError("Every encoder embedding must be on the same device")
        batch_size = sizes.pop()
        device = batches[current_phase].device
        for name, values in batches.items():
            expected = int(self.artifact.goal_prototypes[name].numel())
            if values.shape[1] != expected:
                raise ValueError(
                    f"Embedding dimension mismatch for {name!r}: {values.shape[1]} != {expected}"
                )

        current = batches[current_phase]
        soft_end: SoftAlignmentEndEvidence | None = None
        if (
            self.end_reference_bank is not None
            and current_phase in self.end_reference_bank.references_by_phase
        ):
            soft_end = soft_alignment_end_evidence(
                current,
                self.end_reference_bank.references_by_phase[current_phase],
                self.soft_end_config,
            )
            end_log_scores = soft_end.log_end_probabilities
            end_probabilities = soft_end.end_probabilities
            end_score_mode = "soft_alignment"
        else:
            goal = self.artifact.goal_prototypes[current_phase].to(device)
            end_log_scores = -torch.sum((current - goal) ** 2, dim=1) / float(
                self.artifact.end_scales[current_phase]
            )
            end_probabilities = torch.exp(end_log_scores)
            end_score_mode = "prototype"

        start_columns: list[torch.Tensor] = []
        transition_columns: list[torch.Tensor] = []
        for candidate in candidates:
            values = batches[candidate]
            start = self.artifact.start_prototypes[candidate].to(device)
            delta = torch.sum((values - start) ** 2, dim=1) / float(
                self.artifact.start_scales[candidate]
            )
            logits = (
                delta * float(self.artifact.start_weights[candidate])
                + float(self.artifact.start_biases[candidate])
            )
            log_start = F.logsigmoid(logits)
            log_prior = math.log(float(prior_row.get(candidate, 1.0)))
            start_columns.append(log_start)
            transition_columns.append(end_log_scores + log_start + log_prior)

        start_log_probabilities = torch.stack(start_columns, dim=1)
        transition_log_scores = torch.stack(transition_columns, dim=1)
        selected_indices = transition_log_scores.argmax(dim=1)
        row = torch.arange(batch_size, device=device)
        best_scores = transition_log_scores[row, selected_indices]
        best_start = start_log_probabilities[row, selected_indices]
        selected_phases = tuple(candidates[index] for index in selected_indices.tolist())

        if len(candidates) == 1:
            runner_up_indices = torch.full_like(selected_indices, -1)
            runner_up_phases: tuple[str | None, ...] = (None,) * batch_size
            margins = torch.full_like(best_scores, float("inf"))
        else:
            best_two = torch.topk(transition_log_scores, k=2, dim=1)
            runner_up_indices = best_two.indices[:, 1]
            runner_up_phases = tuple(
                candidates[index] for index in runner_up_indices.tolist()
            )
            margins = best_two.values[:, 0] - best_two.values[:, 1]

        passes_total = best_scores > float(
            self.config.transition_thresholds[current_phase]
        )
        passes_margin = margins > self.config.margin
        start_log_threshold = self.config.start_log_thresholds.get(
            current_phase, self.config.start_log_threshold
        )
        passes_start = best_start > float(start_log_threshold)
        supports = passes_total & passes_margin & passes_start
        return CalibratedRouterEvidence(
            candidate_phases=candidates,
            end_log_scores=end_log_scores,
            start_log_probabilities=start_log_probabilities,
            transition_log_scores=transition_log_scores,
            selected_indices=selected_indices,
            selected_phases=selected_phases,
            runner_up_indices=runner_up_indices,
            runner_up_phases=runner_up_phases,
            margins=margins,
            passes_total_threshold=passes_total,
            passes_margin=passes_margin,
            passes_start_floor=passes_start,
            supports_transition=supports,
            end_score_mode=end_score_mode,
            end_probabilities=end_probabilities,
            end_progress=(None if soft_end is None else soft_end.progress),
            end_within_reference_variance=(
                None if soft_end is None else soft_end.within_reference_variance
            ),
            end_between_reference_variance=(
                None if soft_end is None else soft_end.between_reference_variance
            ),
            end_total_variance=(
                None if soft_end is None else soft_end.total_variance
            ),
        )

    def update(
        self,
        current_phase: str,
        embeddings_by_phase: Mapping[str, torch.Tensor],
        candidate_phases: Sequence[str] | None = None,
    ) -> CalibratedRouterDecision:
        """Apply the three conditions and confirm one candidate over time."""

        evidence = self.score(current_phase, embeddings_by_phase, candidate_phases)
        batch_size = int(evidence.supports_transition.shape[0])
        if len(self._pending) != batch_size:
            self.reset(batch_size)

        decisions: list[str] = []
        selected: list[str | None] = []
        for index in range(batch_size):
            if not bool(evidence.supports_transition[index]):
                self._pending[index] = None
                self._streaks[index] = 0
                decisions.append("stay")
                selected.append(None)
                continue
            candidate = evidence.selected_phases[index]
            if self._pending[index] == candidate:
                self._streaks[index] += 1
            else:
                self._pending[index] = candidate
                self._streaks[index] = 1
            required_steps = self.config.consecutive_steps_by_phase.get(
                current_phase, self.config.consecutive_steps
            )
            if self._streaks[index] >= int(required_steps):
                decisions.append("switch")
                selected.append(candidate)
            else:
                decisions.append("stay")
                selected.append(None)
        return CalibratedRouterDecision(
            decisions=tuple(decisions),
            selected_phases=tuple(selected),
            confirmation_streaks=self._streaks.clone(),
            evidence=evidence,
        )
