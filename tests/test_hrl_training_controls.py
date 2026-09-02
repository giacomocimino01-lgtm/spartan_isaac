from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "source" / "m_dVrk" / "m_dVrk"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# Avoid importing m_dVrk.__init__, which imports IsaacLab during test collection.
m_dvrk_package = types.ModuleType("m_dVrk")
m_dvrk_package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault("m_dVrk", m_dvrk_package)
hrl_package = types.ModuleType("m_dVrk.hrl")
hrl_package.__path__ = [str(PACKAGE_ROOT / "hrl")]
sys.modules.setdefault("m_dVrk.hrl", hrl_package)

constants = load_module("m_dVrk.hrl.constants", PACKAGE_ROOT / "hrl" / "constants.py")
schedules = load_module("m_dVrk.hrl.schedules", PACKAGE_ROOT / "hrl" / "schedules.py")
masked_policy = load_module("m_dVrk.hrl.masked_policy", PACKAGE_ROOT / "hrl" / "masked_policy.py")
checkpoints = load_module("m_dVrk.hrl.checkpoints", PACKAGE_ROOT / "hrl" / "checkpoints.py")


def put_action_one_hot(obs: torch.Tensor, offset: int, action: list[int]) -> None:
    obs[:, offset + action[0]] = 1
    obs[:, offset + 4 + action[1]] = 1
    obs[:, offset + 14 + action[2]] = 1
    obs[:, offset + 18 + action[3]] = 1


def base_observation(batch_size: int = 1) -> torch.Tensor:
    obs = torch.zeros((batch_size, constants.OBS_DIM), dtype=torch.float32)
    idle = [
        constants.VERB_TO_ID["idle"],
        constants.TARGET_TO_ID["None"],
        constants.VERB_TO_ID["idle"],
        constants.TARGET_TO_ID["None"],
    ]
    put_action_one_hot(obs, 96, idle)
    put_action_one_hot(obs, 124, idle)
    obs[:, 240:244] = 1
    return obs


def action_index(action: list[int]) -> int:
    return ((action[0] * 10 + action[1]) * 4 + action[2]) * 10 + action[3]


class ScheduleTests(unittest.TestCase):
    def test_low_high_cosine_boundaries(self) -> None:
        schedule = schedules.LowHighCosineSchedule(
            total_timesteps=5_000_000,
            freeze_timesteps=500_000,
            ramp_timesteps=250_000,
            decay_start_timesteps=3_500_000,
            low_value=1e-5,
            high_value=3e-5,
            final_value=3e-6,
        )
        self.assertAlmostEqual(schedule.value_at(0), 1e-5)
        self.assertAlmostEqual(schedule.value_at(500_000), 1e-5)
        self.assertAlmostEqual(schedule.value_at(625_000), 2e-5)
        self.assertAlmostEqual(schedule.value_at(750_000), 3e-5)
        self.assertAlmostEqual(schedule.value_at(3_500_000), 3e-5)
        self.assertAlmostEqual(schedule.value_at(5_000_000), 3e-6)
        self.assertAlmostEqual(schedule(1.0), 1e-5)
        self.assertAlmostEqual(schedule(0.0), 3e-6)

    def test_entropy_cosine_decay(self) -> None:
        schedule = schedules.CosineDecaySchedule(100, 70, 0.001, 0.0)
        self.assertEqual(schedule.value_at(50), 0.001)
        self.assertEqual(schedule.value_at(70), 0.001)
        self.assertAlmostEqual(schedule.value_at(100), 0.0)


class CheckpointTests(unittest.TestCase):
    def test_step_checkpoint_vecnormalize_name(self) -> None:
        candidates = checkpoints.vecnormalize_candidates(
            "/tmp/dvrk_ppo_phase_0_run_4992000_steps.zip"
        )
        self.assertEqual(
            candidates[0].name,
            "dvrk_ppo_phase_0_run_vecnormalize_4992000_steps.pkl",
        )

    def test_final_checkpoint_vecnormalize_name(self) -> None:
        candidates = checkpoints.vecnormalize_candidates("/tmp/dvrk_ppo_finale.zip")
        self.assertEqual(candidates[0].name, "dvrk_ppo_finale_vecnormalize.pkl")


class ActionMaskTests(unittest.TestCase):
    def setUp(self) -> None:
        axes = [torch.arange(dim) for dim in (4, 10, 4, 10)]
        self.action_table = torch.cartesian_prod(*axes)

    def mask(self, obs: torch.Tensor) -> torch.Tensor:
        return masked_policy.build_joint_action_mask(obs, self.action_table, "phase_0")

    def test_idle_state_allows_distinct_bimanual_reaches(self) -> None:
        mask = self.mask(base_observation())
        reach = constants.VERB_TO_ID["reach"]
        red = constants.TARGET_TO_ID["ring_red"]
        blue = constants.TARGET_TO_ID["ring_blue"]
        self.assertTrue(mask[0, action_index([reach, red, reach, blue])])
        self.assertFalse(mask[0, action_index([reach, red, reach, red])])

    def test_grasp_requires_previous_reach_to_same_ring(self) -> None:
        obs = base_observation()
        reach = constants.VERB_TO_ID["reach"]
        grasp = constants.VERB_TO_ID["grasp"]
        idle = constants.VERB_TO_ID["idle"]
        none = constants.TARGET_TO_ID["None"]
        red = constants.TARGET_TO_ID["ring_red"]
        previous = [reach, red, idle, none]
        obs[:, 124:152] = 0
        put_action_one_hot(obs, 124, previous)
        mask = self.mask(obs)
        self.assertTrue(mask[0, action_index([grasp, red, idle, none])])
        self.assertFalse(mask[0, action_index([grasp, constants.TARGET_TO_ID["ring_blue"], idle, none])])

    def test_release_requires_attachment_and_previous_peg_reach(self) -> None:
        obs = base_observation()
        reach = constants.VERB_TO_ID["reach"]
        release = constants.VERB_TO_ID["release"]
        idle = constants.VERB_TO_ID["idle"]
        none = constants.TARGET_TO_ID["None"]
        green = constants.TARGET_TO_ID["peg_green"]
        obs[:, 228] = 1  # left arm holds ring_red
        obs[:, 124:152] = 0
        put_action_one_hot(obs, 124, [reach, green, idle, none])
        mask = self.mask(obs)
        self.assertTrue(mask[0, action_index([release, green, idle, none])])

    def test_busy_arm_is_forced_to_executing_command(self) -> None:
        obs = base_observation()
        reach = constants.VERB_TO_ID["reach"]
        idle = constants.VERB_TO_ID["idle"]
        none = constants.TARGET_TO_ID["None"]
        red = constants.TARGET_TO_ID["ring_red"]
        blue = constants.TARGET_TO_ID["ring_blue"]
        obs[:, 96:124] = 0
        put_action_one_hot(obs, 96, [reach, red, idle, none])
        obs[:, 152] = 1
        mask = self.mask(obs)
        self.assertTrue(mask[0, action_index([reach, red, idle, none])])
        self.assertFalse(mask[0, action_index([reach, blue, idle, none])])

    def test_masked_policy_keeps_standard_parameter_layout(self) -> None:
        observation_space = spaces.Box(-np.inf, np.inf, (244,), dtype=np.float32)
        action_space = spaces.MultiDiscrete([4, 10, 4, 10])
        lr_schedule = lambda _: 3e-5
        kwargs = {
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "activation_fn": torch.nn.ReLU,
        }
        standard = ActorCriticPolicy(observation_space, action_space, lr_schedule, **kwargs)
        masked = masked_policy.MaskedMultiDiscreteActorCriticPolicy(
            observation_space, action_space, lr_schedule, **kwargs
        )
        self.assertEqual(list(standard.state_dict()), list(masked.state_dict()))
        masked.load_state_dict(standard.state_dict())

        obs = base_observation(batch_size=8)
        actions, _, log_prob = masked(obs)
        mask = self.mask(obs)
        indices = [action_index(row.tolist()) for row in actions]
        self.assertTrue(all(bool(mask[i, index]) for i, index in enumerate(indices)))
        _, evaluated_log_prob, entropy = masked.evaluate_actions(obs, actions)
        self.assertTrue(torch.isfinite(log_prob).all())
        self.assertTrue(torch.isfinite(evaluated_log_prob).all())
        self.assertTrue(torch.isfinite(entropy).all())


if __name__ == "__main__":
    unittest.main()
