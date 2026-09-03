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
controllers_package = types.ModuleType("m_dVrk.controllers")
controllers_package.__path__ = [str(PACKAGE_ROOT / "controllers")]
sys.modules.setdefault("m_dVrk.controllers", controllers_package)

# The state-machine timeout test exercises pure transition logic and does not
# require Isaac Sim. Stub its import-only dependencies during this unit test.
isaaclab_package = types.ModuleType("isaaclab")
isaaclab_utils_package = types.ModuleType("isaaclab.utils")
isaaclab_math_module = types.ModuleType("isaaclab.utils.math")
isaaclab_utils_package.math = isaaclab_math_module
isaaclab_package.utils = isaaclab_utils_package
sys.modules.setdefault("isaaclab", isaaclab_package)
sys.modules.setdefault("isaaclab.utils", isaaclab_utils_package)
sys.modules.setdefault("isaaclab.utils.math", isaaclab_math_module)

ring_sync_module = types.ModuleType("m_dVrk.controllers.ring_sync")
ring_sync_module.get_scene_entity_positions_w = lambda *args, **kwargs: None
ring_sync_module.get_scene_entity_position_w = lambda *args, **kwargs: None
ring_sync_module._get_scene_entity_position_w = lambda *args, **kwargs: None
sys.modules.setdefault("m_dVrk.controllers.ring_sync", ring_sync_module)

constants = load_module("m_dVrk.hrl.constants", PACKAGE_ROOT / "hrl" / "constants.py")
schedules = load_module("m_dVrk.hrl.schedules", PACKAGE_ROOT / "hrl" / "schedules.py")
masked_policy = load_module("m_dVrk.hrl.masked_policy", PACKAGE_ROOT / "hrl" / "masked_policy.py")
checkpoints = load_module("m_dVrk.hrl.checkpoints", PACKAGE_ROOT / "hrl" / "checkpoints.py")
rewards = load_module("m_dVrk.hrl.rewards", PACKAGE_ROOT / "hrl" / "rewards.py")
state_machine = load_module(
    "m_dVrk.controllers.spartan_state_machine",
    PACKAGE_ROOT / "controllers" / "spartan_state_machine.py",
)


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
    ring_positions = torch.tensor(
        [
            [0.335, -0.055, 0.717],
            [0.362, -0.055, 0.717],
            [0.388, -0.055, 0.717],
            [0.415, -0.055, 0.717],
        ],
        dtype=torch.float32,
    )
    obs[:, 164:176] = ring_positions.flatten()
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


class RewardTests(unittest.TestCase):
    def test_success_bonus_is_applied_only_to_successful_environments(self) -> None:
        embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        goal = torch.zeros(2)
        invalid = torch.zeros(2)
        result = rewards.compute_tcc_reward(
            embeddings,
            goal,
            invalid,
            success_mask=torch.tensor([True, False]),
        )
        self.assertAlmostEqual(result[0].item(), 99.999, places=5)
        self.assertAlmostEqual(result[1].item(), -0.001, places=5)

    def test_success_bonus_does_not_replace_invalid_command_penalty(self) -> None:
        result = rewards.compute_tcc_reward(
            torch.zeros((1, 2)),
            torch.zeros(2),
            torch.ones(1),
            success_mask=torch.ones(1, dtype=torch.bool),
        )
        self.assertEqual(result.item(), 99.0)


class StateMachineTimeoutTests(unittest.TestCase):
    def test_reach_timeout_recovers_then_returns_idle(self) -> None:
        machine = state_machine.SPARTANStateMachine.__new__(
            state_machine.SPARTANStateMachine
        )
        machine.device = torch.device("cpu")
        machine.debug_mode = False
        machine._get_reach_entry_position = lambda subject, env_id: torch.ones(3)
        machine.get_target_coordinates = lambda target, subject, env_id: torch.ones(3)
        machine._tip_reached = lambda tip, dest: False
        machine._get_fixed_speed_cmd = lambda tip, dest: dest - tip

        active = ["reach"]
        sub_state = ["DESCEND"]
        step_counter = [machine.REACH_PHASE_TIMEOUT_STEPS]
        current = [{"verb": "reach", "subject": "left_arm", "target": "ring_red"}]
        last_target = [None]
        attached_target = [None]
        target_pos = torch.zeros((1, 3))
        gripper_state = torch.ones(1)
        release_target = [None]
        release_committed = [False]
        tip = torch.zeros((1, 3))
        act = torch.zeros((1, 7))
        grip = torch.ones((1, 2))

        machine._process_arm_logic(
            0,
            active,
            sub_state,
            step_counter,
            current,
            last_target,
            attached_target,
            target_pos,
            gripper_state,
            release_target,
            release_committed,
            tip,
            act,
            grip,
            "left_arm",
        )
        self.assertEqual(sub_state[0], "RECOVER_TO_ENTRY")
        self.assertIsNotNone(current[0])

        step_counter[0] = machine.REACH_RECOVERY_TIMEOUT_STEPS
        machine._process_arm_logic(
            0,
            active,
            sub_state,
            step_counter,
            current,
            last_target,
            attached_target,
            target_pos,
            gripper_state,
            release_target,
            release_committed,
            tip,
            act,
            grip,
            "left_arm",
        )
        self.assertEqual(sub_state[0], "IDLE")
        self.assertIsNone(current[0])


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

    def test_opposite_corner_ring_is_masked_per_arm(self) -> None:
        obs = base_observation()
        # Far upper-left corner: reachable from the right robot root but not
        # from the left robot root at x=0.50.
        obs[:, 164:167] = torch.tensor([0.318, 0.007, 0.717])
        mask = self.mask(obs)
        reach = constants.VERB_TO_ID["reach"]
        idle = constants.VERB_TO_ID["idle"]
        red = constants.TARGET_TO_ID["ring_red"]
        none = constants.TARGET_TO_ID["None"]
        self.assertFalse(mask[0, action_index([reach, red, idle, none])])
        self.assertTrue(mask[0, action_index([idle, none, reach, red])])

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
