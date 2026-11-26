from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "source" / "m_dVrk" / "m_dVrk" / "hrl" / "calibrated_phase_router.py"
)
SPEC = importlib.util.spec_from_file_location(
    "calibrated_phase_router_under_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
router = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = router
SPEC.loader.exec_module(router)


PHASES = ("phase_a", "phase_b", "phase_c")


def make_fit_data() -> object:
    generator = torch.Generator().manual_seed(4)
    embeddings = {name: [] for name in PHASES}
    labels: list[str] = []
    trajectories: list[str] = []
    starts: list[bool] = []
    ends: list[bool] = []
    for trajectory_index in range(6):
        trajectory = f"trajectory_{trajectory_index}"
        for label_index, label in enumerate(PHASES):
            for frame_index in range(5):
                labels.append(label)
                trajectories.append(trajectory)
                starts.append(frame_index < 2)
                ends.append(frame_index >= 3)
                for encoder_index, encoder in enumerate(PHASES):
                    if encoder == label:
                        center = torch.tensor(
                            [5.0 * label_index, float(frame_index)], dtype=torch.float32
                        )
                    else:
                        center = torch.tensor(
                            [20.0 + 5.0 * label_index, 10.0 + encoder_index],
                            dtype=torch.float32,
                        )
                    noise = torch.randn(2, generator=generator) * 0.03
                    embeddings[encoder].append(center + noise)
    return router.CalibratedRouterDataset(
        embeddings_by_phase={name: torch.stack(rows) for name, rows in embeddings.items()},
        phase_labels=tuple(labels),
        trajectory_ids=tuple(trajectories),
        start_mask=torch.tensor(starts, dtype=torch.bool),
        end_mask=torch.tensor(ends, dtype=torch.bool),
    )


def manual_artifact() -> object:
    artifact = router.CalibratedRouterArtifact(
        phase_names=PHASES,
        goal_prototypes={name: torch.tensor([0.0]) for name in PHASES},
        start_prototypes={name: torch.tensor([0.0]) for name in PHASES},
        end_scales={name: 1.0 for name in PHASES},
        start_scales={name: 1.0 for name in PHASES},
        start_weights={name: -1.0 for name in PHASES},
        start_biases={name: 2.0 for name in PHASES},
    )
    artifact.validate()
    return artifact


class CalibrationTests(unittest.TestCase):
    def test_fits_report_artifact_and_round_trips(self) -> None:
        counts = {
            ("phase_a", "phase_b"): 6,
            ("phase_a", "phase_c"): 2,
            ("phase_b", "phase_c"): 6,
        }
        artifact = router.fit_calibrated_router(
            make_fit_data(), transition_counts=counts
        )
        self.assertEqual(artifact.phase_names, PHASES)
        self.assertEqual(artifact.transition_counts, counts)
        self.assertAlmostEqual(
            artifact.transition_priors[("phase_a", "phase_b")], 0.75
        )
        self.assertAlmostEqual(
            artifact.transition_priors[("phase_a", "phase_c")], 0.25
        )
        for phase in PHASES:
            self.assertGreater(artifact.start_scales[phase], 0.0)
            self.assertGreater(artifact.end_scales[phase], 0.0)
            self.assertLessEqual(artifact.start_weights[phase], 0.0)

        restored = router.CalibratedRouterArtifact.from_payload(artifact.to_payload())
        self.assertEqual(restored.phase_names, artifact.phase_names)
        self.assertEqual(restored.transition_counts, artifact.transition_counts)
        self.assertEqual(restored.transition_priors, artifact.transition_priors)
        for phase in PHASES:
            torch.testing.assert_close(
                restored.start_prototypes[phase], artifact.start_prototypes[phase]
            )
            torch.testing.assert_close(
                restored.goal_prototypes[phase], artifact.goal_prototypes[phase]
            )

    def test_requires_two_trajectories_for_loto_scale(self) -> None:
        data = make_fit_data()
        one_trajectory = torch.tensor(
            [value == "trajectory_0" for value in data.trajectory_ids], dtype=torch.bool
        )
        reduced = router.CalibratedRouterDataset(
            embeddings_by_phase={
                name: values[one_trajectory] for name, values in data.embeddings_by_phase.items()
            },
            phase_labels=tuple(
                label for label, keep in zip(data.phase_labels, one_trajectory, strict=True) if keep
            ),
            trajectory_ids=tuple(
                value for value, keep in zip(data.trajectory_ids, one_trajectory, strict=True) if keep
            ),
            start_mask=data.start_mask[one_trajectory],
            end_mask=data.end_mask[one_trajectory],
        )
        with self.assertRaises(ValueError):
            router.fit_calibrated_router(reduced)

    def test_handoff_update_changes_only_source_end_and_target_start(self) -> None:
        artifact = manual_artifact()
        trajectories = {}
        for episode in range(4):
            source = torch.tensor(
                [[3.0], [3.0], [2.0], [1.0], [0.10], [0.05],
                 [0.05], [0.10], [1.0], [2.0]]
            )
            target = torch.tensor(
                [[4.0], [4.0], [4.0], [3.0], [2.0], [0.05],
                 [0.10], [0.20], [1.0], [2.0]]
            )
            trajectories[f"episode_{episode}"] = router.HandoffTrajectory(
                source_embeddings=source,
                target_embeddings=target,
                boundary_index=5,
            )

        updated, stats = router.fit_handoff_calibration(
            artifact,
            trajectories,
            "phase_a",
            "phase_b",
            endpoint_frames=2,
            guard_frames=1,
        )
        self.assertEqual(stats.trajectories, 4)
        self.assertEqual(stats.end_positive_frames, 8)
        self.assertEqual(stats.start_positive_frames, 8)
        self.assertEqual(stats.start_negative_frames, 16)
        self.assertLessEqual(updated.start_weights["phase_b"], 0.0)
        torch.testing.assert_close(
            updated.goal_prototypes["phase_b"], artifact.goal_prototypes["phase_b"]
        )
        torch.testing.assert_close(
            updated.start_prototypes["phase_c"], artifact.start_prototypes["phase_c"]
        )
        self.assertIn("handoff_calibration", updated.metadata)

        metrics = router.replay_handoff_trajectories(
            updated,
            trajectories,
            "phase_a",
            "phase_b",
            transition_threshold=-2.0,
            start_probability_threshold=0.3,
            consecutive_steps=2,
            timing_tolerance=2,
        )
        self.assertEqual(metrics["timing_hits"], 4)
        self.assertEqual(metrics["early_switches"], 0)

    def test_handoff_update_supports_hybrid_prototype_and_scale_windows(self) -> None:
        artifact = manual_artifact()
        trajectories = {}
        for episode in range(4):
            trajectories[f"episode_{episode}"] = router.HandoffTrajectory(
                source_embeddings=torch.tensor(
                    [[3.0], [3.0], [2.0], [1.0], [0.10], [0.05],
                     [0.05], [0.10], [1.0], [2.0]]
                ),
                target_embeddings=torch.tensor(
                    [[4.0], [4.0], [4.0], [3.0], [2.0], [0.05],
                     [0.10], [0.20], [1.0], [2.0]]
                ),
                boundary_index=5,
            )

        updated, stats = router.fit_handoff_calibration(
            artifact,
            trajectories,
            "phase_a",
            "phase_b",
            endpoint_frames=1,
            guard_frames=1,
            end_prototype_frames=1,
            end_scale_frames=3,
            start_prototype_frames=3,
            start_scale_frames=3,
        )

        self.assertEqual(stats.end_positive_frames, 4)
        self.assertEqual(stats.end_scale_positive_frames, 12)
        self.assertEqual(stats.start_prototype_positive_frames, 12)
        self.assertEqual(stats.start_positive_frames, 12)
        torch.testing.assert_close(
            updated.goal_prototypes["phase_a"], torch.tensor([0.05])
        )
        torch.testing.assert_close(
            updated.start_prototypes["phase_b"], torch.tensor([0.11666667])
        )
        self.assertGreater(updated.end_scales["phase_a"], 0.0)
        self.assertGreater(updated.start_scales["phase_b"], 0.0)
        calibration = updated.metadata["handoff_calibration"]
        self.assertEqual(calibration["end_prototype_frames"], 1)
        self.assertEqual(calibration["end_scale_frames"], 3)
        self.assertEqual(calibration["start_prototype_frames"], 3)
        self.assertEqual(calibration["start_scale_frames"], 3)


class SoftAlignmentTests(unittest.TestCase):
    def test_end_probability_progress_and_variance_match_softmax_definition(self) -> None:
        reference = torch.tensor([[0.0], [1.0], [2.0]])
        query = torch.tensor([2.0])
        config = router.SoftAlignmentEndConfig(
            temperature=1.0, end_threshold=0.5
        )

        evidence = router.soft_alignment_end_evidence(query, (reference,), config)
        weights = torch.softmax(torch.tensor([-4.0, -1.0, 0.0]), dim=0)
        times = torch.tensor([0.0, 0.5, 1.0])
        expected_progress = torch.sum(weights * times)
        expected_variance = torch.sum(weights * (times - expected_progress) ** 2)

        torch.testing.assert_close(evidence.end_probabilities, weights[-1:])
        torch.testing.assert_close(
            evidence.log_end_probabilities, torch.log(weights[-1:])
        )
        torch.testing.assert_close(evidence.progress, expected_progress.unsqueeze(0))
        torch.testing.assert_close(
            evidence.total_variance, expected_variance.unsqueeze(0)
        )
        torch.testing.assert_close(
            evidence.between_reference_variance, torch.zeros(1)
        )

    def test_temporal_variance_flags_repeated_visual_state(self) -> None:
        config = router.SoftAlignmentEndConfig(
            temperature=0.01, end_threshold=0.9
        )
        ambiguous = router.soft_alignment_end_evidence(
            torch.tensor([0.0]),
            (torch.tensor([[0.0], [5.0], [0.0]]),),
            config,
        )
        unique = router.soft_alignment_end_evidence(
            torch.tensor([0.0]),
            (torch.tensor([[5.0], [5.0], [0.0]]),),
            config,
        )

        self.assertGreater(
            float(ambiguous.total_variance[0]), float(unique.total_variance[0])
        )
        torch.testing.assert_close(
            ambiguous.end_probabilities, torch.tensor([0.5]), atol=1.0e-6, rtol=0.0
        )
        torch.testing.assert_close(
            unique.end_probabilities, torch.tensor([1.0]), atol=1.0e-6, rtol=0.0
        )

    def test_last_frames_end_region_sums_exact_tail_mass(self) -> None:
        reference = torch.arange(5, dtype=torch.float32).unsqueeze(1)
        query = torch.tensor([4.0])
        config = router.SoftAlignmentEndConfig(
            temperature=1.0,
            end_region_mode="last_frames",
            end_last_frames=2,
        )

        evidence = router.soft_alignment_end_evidence(query, (reference,), config)
        weights = torch.softmax(-torch.tensor([16.0, 9.0, 4.0, 1.0, 0.0]), dim=0)
        torch.testing.assert_close(
            evidence.end_probabilities, weights[-2:].sum().unsqueeze(0)
        )

    def test_last_seconds_maps_to_capture_rate_tail_frames(self) -> None:
        reference = torch.arange(5, dtype=torch.float32).unsqueeze(1)
        query = torch.tensor([4.0])
        seconds = router.SoftAlignmentEndConfig(
            temperature=1.0,
            end_region_mode="last_seconds",
            end_last_seconds=1.0,
            reference_capture_hz=2.0,
        )
        frames = router.SoftAlignmentEndConfig(
            temperature=1.0,
            end_region_mode="last_frames",
            end_last_frames=2,
        )

        seconds_evidence = router.soft_alignment_end_evidence(
            query, (reference,), seconds
        )
        frames_evidence = router.soft_alignment_end_evidence(query, (reference,), frames)
        torch.testing.assert_close(
            seconds_evidence.end_probabilities, frames_evidence.end_probabilities
        )

    def test_reference_bank_round_trip_and_router_integration(self) -> None:
        bank = router.SoftAlignmentReferenceBank(
            phase_names=("phase_a",),
            references_by_phase={
                "phase_a": (torch.tensor([[0.0], [1.0], [2.0]]),)
            },
            metadata={"capture_hz": 2.0},
        )
        restored = router.SoftAlignmentReferenceBank.from_payload(bank.to_payload())
        self.assertEqual(restored.phase_names, ("phase_a",))
        self.assertEqual(restored.metadata["capture_hz"], 2.0)

        configured = router.CalibratedPhaseRouter(
            manual_artifact(),
            router.CalibratedRouterConfig(
                transition_thresholds={name: -10.0 for name in PHASES},
                margin=0.0,
                start_log_threshold=math.log(0.01),
                consecutive_steps=1,
            ),
            transition_priors={("phase_a", "phase_b"): 1.0},
            end_reference_bank=restored,
            soft_end_config=router.SoftAlignmentEndConfig(
                temperature=1.0, end_threshold=0.5
            ),
        )
        evidence = configured.score(
            "phase_a",
            {
                "phase_a": torch.tensor([2.0]),
                "phase_b": torch.tensor([0.0]),
            },
        )
        expected = torch.softmax(torch.tensor([-4.0, -1.0, 0.0]), dim=0)[-1]
        self.assertEqual(evidence.end_score_mode, "soft_alignment")
        torch.testing.assert_close(evidence.end_probabilities, expected.unsqueeze(0))
        torch.testing.assert_close(evidence.end_log_scores, torch.log(expected).unsqueeze(0))
        self.assertIsNotNone(evidence.end_progress)
        self.assertIsNotNone(evidence.end_total_variance)


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = router.CalibratedRouterConfig(
            transition_thresholds={name: -0.5 for name in PHASES},
            margin=0.5,
            start_log_threshold=math.log(0.5),
            consecutive_steps=3,
        )
        self.router = router.CalibratedPhaseRouter(manual_artifact(), self.config)

    @staticmethod
    def b_wins(current: float = 0.0) -> dict[str, torch.Tensor]:
        return {
            "phase_a": torch.tensor([current]),
            "phase_b": torch.tensor([0.0]),
            "phase_c": torch.tensor([2.0]),
        }

    @staticmethod
    def c_wins(current: float = 0.0) -> dict[str, torch.Tensor]:
        return {
            "phase_a": torch.tensor([current]),
            "phase_b": torch.tensor([2.0]),
            "phase_c": torch.tensor([0.0]),
        }

    def test_score_is_exact_report_sum_and_excludes_current_phase(self) -> None:
        evidence = self.router.score("phase_a", self.b_wins())
        self.assertEqual(evidence.candidate_phases, ("phase_b", "phase_c"))
        expected_b = torch.nn.functional.logsigmoid(torch.tensor(2.0))
        expected_c = torch.nn.functional.logsigmoid(torch.tensor(-2.0))
        torch.testing.assert_close(evidence.end_log_scores, torch.tensor([-0.0]))
        torch.testing.assert_close(
            evidence.transition_log_scores,
            torch.stack((expected_b, expected_c)).unsqueeze(0),
        )
        self.assertEqual(evidence.selected_phases, ("phase_b",))
        self.assertTrue(bool(evidence.passes_total_threshold[0]))
        self.assertTrue(bool(evidence.passes_margin[0]))
        self.assertTrue(bool(evidence.passes_start_floor[0]))

    def test_sparse_prior_excludes_unannotated_edges(self) -> None:
        sparse = router.CalibratedPhaseRouter(
            manual_artifact(),
            self.config,
            transition_priors={
                ("phase_a", "phase_b"): 4.0,
                ("phase_b", "phase_c"): 4.0,
            },
        )
        evidence = sparse.score("phase_a", self.c_wins())
        self.assertEqual(evidence.candidate_phases, ("phase_b",))
        self.assertEqual(evidence.selected_phases, ("phase_b",))
        self.assertTrue(math.isinf(float(evidence.margins[0])))

        next_evidence = sparse.score(
            "phase_b",
            {
                "phase_a": torch.tensor([0.0]),
                "phase_b": torch.tensor([0.0]),
                "phase_c": torch.tensor([0.0]),
            },
        )
        self.assertEqual(next_evidence.candidate_phases, ("phase_c",))

    def test_per_phase_start_floor_and_confirmation_override_defaults(self) -> None:
        configured = router.CalibratedPhaseRouter(
            manual_artifact(),
            router.CalibratedRouterConfig(
                transition_thresholds={name: -0.5 for name in PHASES},
                margin=0.5,
                start_log_threshold=math.log(0.95),
                consecutive_steps=5,
                start_log_thresholds={"phase_a": math.log(0.5)},
                consecutive_steps_by_phase={"phase_a": 2},
            ),
        )
        first = configured.update("phase_a", self.b_wins())
        second = configured.update("phase_a", self.b_wins())
        self.assertEqual(first.decisions, ("stay",))
        self.assertEqual(second.decisions, ("switch",))

    def test_all_three_conditions_are_required(self) -> None:
        weak_end = self.router.score("phase_a", self.b_wins(current=1.0))
        self.assertFalse(bool(weak_end.passes_total_threshold[0]))
        self.assertTrue(bool(weak_end.passes_margin[0]))
        self.assertTrue(bool(weak_end.passes_start_floor[0]))
        self.assertFalse(bool(weak_end.supports_transition[0]))

        ambiguous = self.router.score(
            "phase_a",
            {
                "phase_a": torch.tensor([0.0]),
                "phase_b": torch.tensor([0.0]),
                "phase_c": torch.tensor([0.1]),
            },
        )
        self.assertFalse(bool(ambiguous.passes_margin[0]))
        self.assertFalse(bool(ambiguous.supports_transition[0]))

        permissive_total = router.CalibratedPhaseRouter(
            manual_artifact(),
            router.CalibratedRouterConfig(
                transition_thresholds={name: -2.0 for name in PHASES},
                margin=0.5,
                start_log_threshold=math.log(0.5),
                consecutive_steps=3,
            ),
        ).score(
            "phase_a",
            {
                "phase_a": torch.tensor([0.0]),
                "phase_b": torch.tensor([1.5]),
                "phase_c": torch.tensor([3.0]),
            },
        )
        self.assertTrue(bool(permissive_total.passes_total_threshold[0]))
        self.assertTrue(bool(permissive_total.passes_margin[0]))
        self.assertFalse(bool(permissive_total.passes_start_floor[0]))
        self.assertFalse(bool(permissive_total.supports_transition[0]))

    def test_switches_only_after_same_candidate_is_confirmed(self) -> None:
        first = self.router.update("phase_a", self.b_wins())
        second = self.router.update("phase_a", self.b_wins())
        third = self.router.update("phase_a", self.b_wins())
        self.assertEqual(first.decisions, ("stay",))
        self.assertEqual(second.decisions, ("stay",))
        self.assertEqual(third.decisions, ("switch",))
        self.assertEqual(third.selected_phases, ("phase_b",))
        self.assertEqual(int(third.confirmation_streaks[0]), 3)

    def test_candidate_change_restarts_confirmation(self) -> None:
        self.router.update("phase_a", self.b_wins())
        before_change = self.router.update("phase_a", self.b_wins())
        after_change = self.router.update("phase_a", self.c_wins())
        self.assertEqual(int(before_change.confirmation_streaks[0]), 2)
        self.assertEqual(int(after_change.confirmation_streaks[0]), 1)
        self.assertEqual(after_change.decisions, ("stay",))

    def test_failed_condition_resets_confirmation(self) -> None:
        self.router.update("phase_a", self.b_wins())
        reset = self.router.update("phase_a", self.b_wins(current=1.0))
        restarted = self.router.update("phase_a", self.b_wins())
        self.assertEqual(int(reset.confirmation_streaks[0]), 0)
        self.assertEqual(int(restarted.confirmation_streaks[0]), 1)


if __name__ == "__main__":
    unittest.main()
