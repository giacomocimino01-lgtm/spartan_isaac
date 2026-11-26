from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "source" / "m_dVrk" / "m_dVrk" / "hrl" / "phase_router.py"
SPEC = importlib.util.spec_from_file_location("phase_router_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
phase_router = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = phase_router
SPEC.loader.exec_module(phase_router)


class MeanStartEmbeddingTests(unittest.TestCase):
    def test_weights_videos_equally(self) -> None:
        result = phase_router.mean_start_embedding(
            [
                torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
                torch.tensor([[5.0, 5.0]]),
            ]
        )
        torch.testing.assert_close(result, torch.tensor([3.0, 3.0]))

    def test_rejects_empty_collection(self) -> None:
        with self.assertRaises(ValueError):
            phase_router.mean_start_embedding([])


class RawNearestStartTests(unittest.TestCase):
    def test_selects_smallest_raw_squared_distance(self) -> None:
        result = phase_router.select_phase_by_nearest_start(
            {
                "phase_0": torch.tensor([[0.0, 1.0], [8.0, 8.0]]),
                "phase_1": torch.tensor([[9.0, 9.0], [5.0, 6.0]]),
            },
            {
                "phase_0": torch.tensor([0.0, 0.0]),
                "phase_1": torch.tensor([5.0, 5.0]),
            },
            phase_order=["phase_0", "phase_1"],
        )
        self.assertEqual(result.selected_phases, ("phase_0", "phase_1"))
        torch.testing.assert_close(
            result.squared_distances,
            torch.tensor([[1.0, 32.0], [128.0, 1.0]]),
        )
        torch.testing.assert_close(result.margins, torch.tensor([31.0, 127.0]))

    def test_does_not_normalize_encoder_scales(self) -> None:
        result = phase_router.select_phase_by_nearest_start(
            {
                "wide": torch.tensor([2.0]),
                "compact": torch.tensor([100.0]),
            },
            {
                "wide": torch.tensor([0.0]),
                "compact": torch.tensor([99.0]),
            },
        )
        self.assertEqual(result.selected_phases, ("compact",))
        torch.testing.assert_close(result.squared_distances, torch.tensor([[4.0, 1.0]]))

    def test_rejects_mismatched_batch_sizes(self) -> None:
        with self.assertRaises(ValueError):
            phase_router.select_phase_by_nearest_start(
                {
                    "phase_0": torch.zeros((2, 2)),
                    "phase_1": torch.zeros((1, 2)),
                },
                {
                    "phase_0": torch.zeros(2),
                    "phase_1": torch.zeros(2),
                },
            )


if __name__ == "__main__":
    unittest.main()
