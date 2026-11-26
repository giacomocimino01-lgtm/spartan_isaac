from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import gymnasium as gym
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
ring_sync_module.sync_attached_and_frozen_rings = lambda *args, **kwargs: None
sys.modules.setdefault("m_dVrk.controllers.ring_sync", ring_sync_module)

constants = load_module("m_dVrk.hrl.constants", PACKAGE_ROOT / "hrl" / "constants.py")
schedules = load_module("m_dVrk.hrl.schedules", PACKAGE_ROOT / "hrl" / "schedules.py")
curriculum = load_module(
    "m_dVrk.hrl.curriculum", PACKAGE_ROOT / "hrl" / "curriculum.py"
)
masked_policy = load_module("m_dVrk.hrl.masked_policy", PACKAGE_ROOT / "hrl" / "masked_policy.py")
anchored_ppo = load_module("m_dVrk.hrl.anchored_ppo", PACKAGE_ROOT / "hrl" / "anchored_ppo.py")
phase_router = load_module("m_dVrk.hrl.phase_router", PACKAGE_ROOT / "hrl" / "phase_router.py")
expert_scheduling = load_module(
    "m_dVrk.hrl.expert_scheduling", PACKAGE_ROOT / "hrl" / "expert_scheduling.py"
)
checkpoints = load_module("m_dVrk.hrl.checkpoints", PACKAGE_ROOT / "hrl" / "checkpoints.py")
rewards = load_module("m_dVrk.hrl.rewards", PACKAGE_ROOT / "hrl" / "rewards.py")
success = load_module("m_dVrk.hrl.success", PACKAGE_ROOT / "hrl" / "success.py")
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


class Phase2BimanualSchedulerTests(unittest.TestCase):
    def test_grasps_advance_independently_when_reaches_finish_asynchronously(self) -> None:
        scheduler = expert_scheduling.Phase2BimanualScheduler([("ring_red", "ring_blue")])
        self.assertEqual(scheduler.issue_ready(left_idle=True, right_idle=True), (("reach", "left_arm", "ring_red"), ("reach", "right_arm", "ring_blue")))
        self.assertEqual(scheduler.issue_ready(left_idle=True, right_idle=False), (("grasp", "left_arm", "ring_red"),))
        self.assertEqual(scheduler.issue_ready(left_idle=False, right_idle=True), (("grasp", "right_arm", "ring_blue"),))
        self.assertEqual(scheduler.issue_ready(left_idle=True, right_idle=True), (("reach", "left_arm", constants.BOARD_CENTER),))
        self.assertEqual(scheduler.issue_ready(left_idle=True, right_idle=True), (("release", "left_arm", constants.BOARD_CENTER),))
        self.assertEqual(scheduler.issue_ready(left_idle=True, right_idle=True), (("reach", "right_arm", constants.BOARD_CENTER),))
        self.assertEqual(scheduler.issue_ready(left_idle=True, right_idle=True), (("release", "right_arm", constants.BOARD_CENTER),))
        self.assertEqual(scheduler.issue_ready(left_idle=True, right_idle=True), ())
        self.assertTrue(scheduler.complete)

    def test_scheduler_waits_for_busy_arm_before_serial_destination(self) -> None:
        scheduler = expert_scheduling.Phase2BimanualScheduler([("ring_yellow", "ring_green")])
        scheduler.issue_ready(left_idle=True, right_idle=True)
        scheduler.issue_ready(left_idle=True, right_idle=False)
        scheduler.issue_ready(left_idle=False, right_idle=True)
        self.assertEqual(scheduler.issue_ready(left_idle=True, right_idle=False), ())
        self.assertEqual(scheduler.issue_ready(left_idle=True, right_idle=True), (("reach", "left_arm", constants.BOARD_CENTER),))


class Phase3BimanualSchedulerTests(unittest.TestCase):
    def test_exposes_second_ring_before_overlapping_transfers(self) -> None:
        scheduler = expert_scheduling.Phase3BimanualScheduler([("ring_red", "peg_red", "ring_blue", "peg_green")])
        self.assertEqual(scheduler.issue_ready(left_idle=True, right_idle=True), (("reach", "left_arm", "ring_red"),))
        self.assertEqual(scheduler.issue_ready(left_idle=True, right_idle=True), (("grasp", "left_arm", "ring_red"),))
        self.assertEqual(scheduler.issue_ready(left_idle=True, right_idle=True), (("reach", "left_arm", "peg_red"), ("reach", "right_arm", "ring_blue")))
        self.assertEqual(scheduler.issue_ready(left_idle=True, right_idle=False), (("release", "left_arm", "peg_red"),))
        self.assertEqual(scheduler.issue_ready(left_idle=False, right_idle=True), (("grasp", "right_arm", "ring_blue"),))
        self.assertEqual(scheduler.issue_ready(left_idle=True, right_idle=True), (("reach", "right_arm", "peg_green"),))
        self.assertEqual(scheduler.issue_ready(left_idle=True, right_idle=True), (("release", "right_arm", "peg_green"),))
        self.assertEqual(scheduler.issue_ready(left_idle=True, right_idle=True), ())
        self.assertTrue(scheduler.complete)


class Phase4BimanualSchedulerTests(unittest.TestCase):
    def test_both_arms_complete_independent_outside_transfers(self) -> None:
        scheduler = expert_scheduling.Phase4BimanualScheduler([
            (
                "ring_red",
                constants.OUTSIDE_LEFT,
                "ring_yellow",
                constants.OUTSIDE_RIGHT,
            )
        ])
        expected = (
            ("reach", "left_arm", "ring_red"),
            ("reach", "right_arm", "ring_yellow"),
        )
        self.assertEqual(
            scheduler.issue_ready(left_idle=True, right_idle=True), expected
        )
        self.assertEqual(
            scheduler.issue_ready(left_idle=True, right_idle=False),
            (("grasp", "left_arm", "ring_red"),),
        )
        self.assertEqual(
            scheduler.issue_ready(left_idle=False, right_idle=True),
            (("grasp", "right_arm", "ring_yellow"),),
        )
        self.assertEqual(
            scheduler.issue_ready(left_idle=True, right_idle=True),
            (
                ("reach", "left_arm", constants.OUTSIDE_LEFT),
                ("reach", "right_arm", constants.OUTSIDE_RIGHT),
            ),
        )
        self.assertEqual(
            scheduler.issue_ready(left_idle=True, right_idle=True),
            (
                ("release", "left_arm", constants.OUTSIDE_LEFT),
                ("release", "right_arm", constants.OUTSIDE_RIGHT),
            ),
        )
        self.assertEqual(
            scheduler.issue_ready(left_idle=True, right_idle=True), ()
        )
        self.assertTrue(scheduler.complete)


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

    def test_soft_mask_cosine_decay_reaches_zero_before_training_end(self) -> None:
        schedule = schedules.CosineDecaySchedule(1_500_000, 250_000, 8.0, 0.0)
        self.assertEqual(schedule.value_at(0), 8.0)
        self.assertEqual(schedule.value_at(250_000), 8.0)
        self.assertGreater(schedule.value_at(1_000_000), 0.0)
        self.assertEqual(schedule.value_at(1_500_000), 0.0)
        self.assertEqual(schedule.value_at(2_000_000), 0.0)

    def test_ring_curriculum_boundaries(self) -> None:
        schedule = curriculum.RingRandomizationCurriculum(
            total_timesteps=5_000_000,
            start_timesteps=500_000,
            full_timesteps=3_500_000,
        )
        self.assertEqual(schedule.value_at(0), 0.0)
        self.assertEqual(schedule.value_at(500_000), 0.0)
        self.assertAlmostEqual(schedule.value_at(2_000_000), 0.5)
        self.assertEqual(schedule.value_at(3_500_000), 1.0)
        self.assertEqual(schedule.value_at(5_000_000), 1.0)

    def test_ring_sampling_interval_expands_from_fixed_pose(self) -> None:
        full = (0.318, 0.432)
        anchor = 0.335
        self.assertEqual(
            curriculum.scaled_interval_around_anchor(anchor, full, 0.0),
            (anchor, anchor),
        )
        self.assertEqual(
            curriculum.scaled_interval_around_anchor(anchor, full, 1.0),
            full,
        )
        lower, upper = curriculum.scaled_interval_around_anchor(anchor, full, 0.5)
        self.assertAlmostEqual(lower, 0.3265)
        self.assertAlmostEqual(upper, 0.3835)

    def test_phase_entry_pose_frontier_sampling_bounds(self) -> None:
        self.assertEqual(
            curriculum.phase_entry_pose_sampling_bounds(0.25, None),
            (0.0, 0.25),
        )
        self.assertEqual(
            curriculum.phase_entry_pose_sampling_bounds(0.25, 0.05),
            (0.20, 0.25),
        )
        self.assertEqual(
            curriculum.phase_entry_pose_sampling_bounds(0.03, 0.05),
            (0.0, 0.03),
        )
        with self.assertRaises(ValueError):
            curriculum.phase_entry_pose_sampling_bounds(0.25, 0.0)

    def test_phase_entry_park_neighborhood_validation(self) -> None:
        self.assertEqual(
            curriculum.validate_phase_entry_neighborhood(0.4, 1.0),
            (0.4, 1.0),
        )
        self.assertEqual(
            curriculum.validate_phase_entry_neighborhood(0.0, 0.0),
            (0.0, 0.0),
        )
        with self.assertRaises(ValueError):
            curriculum.validate_phase_entry_neighborhood(1.01, 1.0)
        with self.assertRaises(ValueError):
            curriculum.validate_phase_entry_neighborhood(0.4, -0.01)

    def test_ring_curriculum_rejects_invalid_boundaries(self) -> None:
        with self.assertRaises(ValueError):
            curriculum.RingRandomizationCurriculum(100, 80, 80)

    def test_adaptive_curriculum_advances_after_success_window(self) -> None:
        adaptive = curriculum.AdaptiveRingRandomizationCurriculum(
            start_scale=0.75,
            scale_step=0.05,
            success_threshold=0.75,
            window_episodes=4,
        )
        advanced = adaptive.observe(
            [(0.75, True), (0.75, True), (0.75, True), (0.75, False)]
        )
        self.assertTrue(advanced)
        self.assertAlmostEqual(adaptive.current_scale, 0.80)
        self.assertEqual(adaptive.recent_episode_count, 0)

    def test_adaptive_curriculum_holds_below_threshold(self) -> None:
        adaptive = curriculum.AdaptiveRingRandomizationCurriculum(
            success_threshold=0.75,
            window_episodes=4,
        )
        advanced = adaptive.observe(
            [(0.75, True), (0.75, False), (0.75, False), (0.75, True)]
        )
        self.assertFalse(advanced)
        self.assertEqual(adaptive.current_scale, 0.75)
        self.assertAlmostEqual(adaptive.recent_success_rate, 0.5)

    def test_adaptive_curriculum_ignores_inflight_old_scale(self) -> None:
        adaptive = curriculum.AdaptiveRingRandomizationCurriculum(
            start_scale=0.75,
            scale_step=0.05,
            success_threshold=1.0,
            window_episodes=2,
        )
        self.assertTrue(adaptive.observe([(0.75, True), (0.75, True)]))
        self.assertAlmostEqual(adaptive.current_scale, 0.80)
        self.assertFalse(adaptive.observe([(0.75, True), (0.75, True)]))
        self.assertEqual(adaptive.recent_episode_count, 0)

    def test_adaptive_curriculum_respects_advance_hold(self) -> None:
        adaptive = curriculum.AdaptiveRingRandomizationCurriculum(
            success_threshold=1.0,
            window_episodes=2,
        )
        self.assertFalse(
            adaptive.observe(
                [(0.75, True), (0.75, True)], allow_advance=False
            )
        )
        self.assertTrue(adaptive.observe([], allow_advance=True))

    def test_phase_entry_pose_curriculum_advances_in_quarter_steps(self) -> None:
        adaptive = curriculum.AdaptivePhaseEntryPoseCurriculum(
            success_threshold=1.0,
            window_episodes=2,
        )
        self.assertEqual(adaptive.current_scale, 0.0)
        for expected in (0.25, 0.50, 0.75, 1.0):
            scale = adaptive.current_scale
            self.assertTrue(adaptive.observe([(scale, True), (scale, True)]))
            self.assertAlmostEqual(adaptive.current_scale, expected)
        self.assertFalse(adaptive.observe([(1.0, True), (1.0, True)]))

    def test_adaptive_minimal_mask_opens_scale_before_penalty(self) -> None:
        adaptive = curriculum.AdaptiveMinimalMaskCurriculum(
            start_scale=0.90,
            scale_step=0.10,
            initial_penalty=2.0,
            penalty_step=1.0,
            success_threshold=0.75,
            window_episodes=4,
            cooldown_timesteps=0,
        )
        transition = adaptive.observe(
            [True, True, True, False], timestep=100
        )
        self.assertIsNotNone(transition)
        self.assertEqual(adaptive.current_scale, 1.0)
        self.assertEqual(adaptive.current_penalty, 2.0)

        adaptive.observe([True] * 4, timestep=200)
        self.assertEqual(adaptive.current_scale, 1.0)
        self.assertEqual(adaptive.current_penalty, 1.0)

    def test_adaptive_minimal_mask_requires_final_validation_window(self) -> None:
        adaptive = curriculum.AdaptiveMinimalMaskCurriculum(
            start_scale=1.0,
            initial_penalty=1.0,
            penalty_step=1.0,
            success_threshold=1.0,
            window_episodes=2,
            cooldown_timesteps=0,
        )
        adaptive.observe([True, True], timestep=100)
        self.assertEqual(adaptive.current_penalty, 0.0)
        self.assertFalse(adaptive.complete)
        adaptive.observe([True, True], timestep=200)
        self.assertTrue(adaptive.complete)

    def test_adaptive_minimal_mask_holds_below_threshold(self) -> None:
        adaptive = curriculum.AdaptiveMinimalMaskCurriculum(
            success_threshold=0.75,
            window_episodes=4,
        )
        self.assertIsNone(
            adaptive.observe([True, False, False, True], timestep=100)
        )
        self.assertEqual(adaptive.current_scale, adaptive.start_scale)
        self.assertEqual(adaptive.current_penalty, adaptive.initial_penalty)

    def test_adaptive_minimal_mask_ignores_cooldown_outcomes(self) -> None:
        adaptive = curriculum.AdaptiveMinimalMaskCurriculum(
            start_scale=0.50,
            scale_step=0.10,
            success_threshold=1.0,
            window_episodes=2,
            cooldown_timesteps=50,
        )
        adaptive.observe([True, True], timestep=100)
        self.assertEqual(adaptive.current_scale, 0.60)
        adaptive.observe([True, True], timestep=120)
        self.assertEqual(adaptive.recent_episode_count, 0)
        adaptive.observe([True, True], timestep=150)
        self.assertEqual(adaptive.current_scale, 0.70)


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


class KLAnchorTests(unittest.TestCase):
    class _ReferencePolicy:
        def __init__(self) -> None:
            self.parameter = torch.nn.Parameter(torch.ones(1))
            self.device = None
            self.training_mode = None

        def to(self, device):
            self.device = device
            return self

        def set_training_mode(self, mode: bool) -> None:
            self.training_mode = mode

        def parameters(self):
            return [self.parameter]

    class _TinyMaskedEnv(gym.Env):
        observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(constants.OBS_DIM,),
            dtype=np.float32,
        )
        action_space = spaces.MultiDiscrete([4, 10, 4, 10])

        def __init__(self) -> None:
            self.steps = 0

        @staticmethod
        def observation() -> np.ndarray:
            return base_observation().numpy()[0]

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            return self.observation(), {}

        def step(self, action):
            self.steps += 1
            reward = float(action[1] == 9 and action[3] == 9)
            return self.observation(), reward, self.steps >= 4, False, {}

    def test_set_kl_anchor_freezes_reference_policy(self) -> None:
        model = types.SimpleNamespace(device=torch.device("cpu"))
        reference = self._ReferencePolicy()
        anchored_ppo.KLAnchoredPPO.set_kl_anchor(model, reference, 0.1)
        self.assertIs(model.kl_reference_policy, reference)
        self.assertEqual(model.kl_anchor_coefficient, 0.1)
        self.assertEqual(reference.device, torch.device("cpu"))
        self.assertFalse(reference.training_mode)
        self.assertFalse(reference.parameter.requires_grad)
        self.assertIsNone(reference.parameter.grad)

    def test_set_kl_anchor_rejects_non_positive_coefficient(self) -> None:
        model = types.SimpleNamespace(device=torch.device("cpu"))
        with self.assertRaises(ValueError):
            anchored_ppo.KLAnchoredPPO.set_kl_anchor(
                model,
                self._ReferencePolicy(),
                0.0,
            )

    def test_stable_forward_kl_is_zero_for_identical_distributions(self) -> None:
        distribution = torch.distributions.Categorical(
            logits=torch.tensor([[3.0, -2.0, -1000.0]])
        )
        result = anchored_ppo.BCRegularizedPPO._stable_forward_kl(
            distribution,
            distribution,
        )
        self.assertTrue(torch.isfinite(result).all())
        self.assertAlmostEqual(result.item(), 0.0, places=7)

    def test_stable_forward_kl_survives_current_probability_underflow(self) -> None:
        reference = torch.distributions.Categorical(
            logits=torch.tensor([[0.0, 0.0, torch.finfo(torch.float32).min]])
        )
        current_logits = torch.tensor(
            [[0.0, -1000.0, torch.finfo(torch.float32).min]],
            requires_grad=True,
        )
        current = torch.distributions.Categorical(logits=current_logits)
        result = anchored_ppo.BCRegularizedPPO._stable_forward_kl(
            reference,
            current,
        ).mean()
        self.assertTrue(torch.isfinite(result))
        self.assertGreater(result.item(), 0.0)
        result.backward()
        self.assertTrue(torch.isfinite(current_logits.grad).all())
        self.assertGreater(abs(current_logits.grad[0, 1].item()), 0.0)

    def test_bc_replay_ppo_completes_minibatch_updates(self) -> None:
        model = anchored_ppo.BCRegularizedPPO(
            masked_policy.MaskedMultiDiscreteActorCriticPolicy,
            self._TinyMaskedEnv(),
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            learning_rate=1e-4,
            policy_kwargs={
                "mask_task_phase": "phase_1",
                "action_mask_mode": "minimal",
                "net_arch": {"pi": [16], "vf": [16]},
            },
            verbose=0,
        )
        expert_observations = np.repeat(
            self._TinyMaskedEnv.observation()[None, :],
            16,
            axis=0,
        )
        expert_actions = np.repeat(
            np.array([[3, 9, 3, 9]], dtype=np.int64),
            16,
            axis=0,
        )
        model.set_bc_replay(
            expert_observations,
            expert_actions,
            coefficient=0.1,
            batch_size=4,
        )
        model.learn(32)
        self.assertEqual(model.num_timesteps, 32)
        self.assertEqual(tuple(model.bc_replay_observations.shape), (16, 244))
        self.assertEqual(tuple(model.bc_replay_actions.shape), (16, 4))

    def test_bc_replay_stratifies_semantic_action_changes(self) -> None:
        model = types.SimpleNamespace(
            observation_space=spaces.Box(
                -np.inf,
                np.inf,
                shape=(constants.OBS_DIM,),
                dtype=np.float32,
            ),
            action_space=spaces.MultiDiscrete([4, 10, 4, 10]),
        )
        actions = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [2, 2, 0, 0],
            ],
            dtype=np.int64,
        )
        anchored_ppo.BCRegularizedPPO.set_bc_replay(
            model,
            np.zeros((len(actions), constants.OBS_DIM), dtype=np.float32),
            actions,
            coefficient=0.1,
            batch_size=10,
            decision_fraction=0.5,
        )
        self.assertEqual(model.bc_replay_decision_indices.tolist(), [0, 2, 4])
        self.assertEqual(model.bc_replay_steady_indices.tolist(), [1, 3])

        sampled = anchored_ppo.BCRegularizedPPO._sample_replay_indices(
            len(actions),
            10,
            model.bc_replay_decision_indices,
            model.bc_replay_steady_indices,
            model.bc_replay_decision_fraction,
        )
        sampled_decisions = sum(index in {0, 2, 4} for index in sampled.tolist())
        self.assertEqual(sampled_decisions, 5)

    def test_bc_replay_rejects_invalid_decision_fraction(self) -> None:
        model = types.SimpleNamespace(
            observation_space=spaces.Box(
                -np.inf,
                np.inf,
                shape=(constants.OBS_DIM,),
                dtype=np.float32,
            ),
            action_space=spaces.MultiDiscrete([4, 10, 4, 10]),
        )
        with self.assertRaises(ValueError):
            anchored_ppo.BCRegularizedPPO.set_bc_replay(
                model,
                np.zeros((2, constants.OBS_DIM), dtype=np.float32),
                np.zeros((2, 4), dtype=np.int64),
                coefficient=0.1,
                batch_size=4,
                decision_fraction=1.1,
            )

    def test_bc_replay_rejects_mismatched_dataset_lengths(self) -> None:
        model = types.SimpleNamespace(
            observation_space=spaces.Box(
                -np.inf,
                np.inf,
                shape=(constants.OBS_DIM,),
                dtype=np.float32,
            ),
            action_space=spaces.MultiDiscrete([4, 10, 4, 10]),
        )
        with self.assertRaises(ValueError):
            anchored_ppo.BCRegularizedPPO.set_bc_replay(
                model,
                np.zeros((2, constants.OBS_DIM), dtype=np.float32),
                np.zeros((1, 4), dtype=np.int64),
                coefficient=0.1,
                batch_size=4,
            )


class PhaseTwoTests(unittest.TestCase):
    def test_board_center_alias_preserves_action_dimension(self) -> None:
        self.assertEqual(len(constants.TARGET_MAP), 10)
        target_id = constants.target_id_for_phase(constants.BOARD_CENTER, "phase_2")
        self.assertEqual(target_id, constants.TARGET_TO_ID["peg_gray"])
        self.assertEqual(
            constants.target_name_for_phase(target_id, "phase_2"),
            constants.BOARD_CENTER,
        )
        self.assertEqual(
            constants.target_name_for_phase(target_id, "phase_1"),
            "peg_gray",
        )

    def test_phase_two_success_requires_all_rings_at_center(self) -> None:
        sm = types.SimpleNamespace(
            frozen_rings_mask={name: torch.tensor([True]) for name in constants.RING_NAMES},
            ring_support_peg={name: [constants.BOARD_CENTER] for name in constants.RING_NAMES},
        )
        result = success.task_success(
            sm, "phase_2", constants.RING_NAMES, 1, torch.device("cpu")
        )
        self.assertTrue(result.item())
        sm.ring_support_peg[constants.RING_NAMES[-1]][0] = constants.PEG_BLUE
        result = success.task_success(
            sm, "phase_2", constants.RING_NAMES, 1, torch.device("cpu")
        )
        self.assertFalse(result.item())


class PhaseThreeTests(unittest.TestCase):
    def test_initial_layout_is_four_ring_center_stack(self) -> None:
        order = list(constants.RING_NAMES)
        stacks = state_machine.SPARTANStateMachine.build_phase3_initial_stacks(
            "center4", order
        )
        self.assertEqual(stacks, {constants.BOARD_CENTER: order})

    def test_success_is_colour_invariant_one_per_coloured_peg(self) -> None:
        # Deliberately rotate ring colours relative to peg colours.
        supports = {
            ring_name: [constants.PEG_NAMES[(index + 1) % 4]]
            for index, ring_name in enumerate(constants.RING_NAMES)
        }
        sm = types.SimpleNamespace(
            frozen_rings_mask={
                name: torch.tensor([True]) for name in constants.RING_NAMES
            },
            ring_support_peg=supports,
        )
        result = success.task_success(
            sm, "phase_3", constants.RING_NAMES, 1, torch.device("cpu")
        )
        self.assertTrue(result.item())

    def test_duplicate_peg_fails_even_when_all_rings_are_coloured(self) -> None:
        supports = {
            constants.RING_NAMES[0]: [constants.PEG_RED],
            constants.RING_NAMES[1]: [constants.PEG_RED],
            constants.RING_NAMES[2]: [constants.PEG_GREEN],
            constants.RING_NAMES[3]: [constants.PEG_BLUE],
        }
        sm = types.SimpleNamespace(
            frozen_rings_mask={
                name: torch.tensor([True]) for name in constants.RING_NAMES
            },
            ring_support_peg=supports,
        )
        result = success.task_success(
            sm, "phase_3", constants.RING_NAMES, 1, torch.device("cpu")
        )
        self.assertFalse(result.item())


class PhaseFourTests(unittest.TestCase):
    def test_target_aliases_preserve_historical_action_dimension(self) -> None:
        self.assertEqual(len(constants.TARGET_MAP), 10)
        self.assertEqual(
            constants.target_name_for_phase(
                constants.OUTSIDE_RIGHT_TARGET_ID, "phase_4", "reach"
            ),
            constants.OUTSIDE_RIGHT,
        )
        self.assertEqual(
            constants.target_name_for_phase(
                constants.OUTSIDE_LEFT_TARGET_ID, "phase_4", "release"
            ),
            constants.OUTSIDE_LEFT,
        )
        self.assertEqual(
            constants.target_name_for_phase(
                constants.OUTSIDE_LEFT_TARGET_ID, "phase_4", "idle"
            ),
            "None",
        )

    def test_initial_layout_has_one_ring_per_coloured_peg(self) -> None:
        order = list(constants.RING_NAMES)
        stacks = state_machine.SPARTANStateMachine.build_phase4_initial_stacks(
            constants.PHASE4_DEFAULT_INITIAL_STATE, order
        )
        self.assertEqual(
            stacks,
            {peg: [ring] for peg, ring in zip(constants.PEG_NAMES, order)},
        )

    def test_success_requires_exactly_two_rings_per_side(self) -> None:
        supports = {
            constants.RING_NAMES[0]: [constants.OUTSIDE_RIGHT],
            constants.RING_NAMES[1]: [constants.OUTSIDE_RIGHT],
            constants.RING_NAMES[2]: [constants.OUTSIDE_LEFT],
            constants.RING_NAMES[3]: [constants.OUTSIDE_LEFT],
        }
        sm = types.SimpleNamespace(
            frozen_rings_mask={
                name: torch.tensor([True]) for name in constants.RING_NAMES
            },
            ring_support_peg=supports,
        )
        result = success.task_success(
            sm, "phase_4", constants.RING_NAMES, 1, torch.device("cpu")
        )
        self.assertTrue(result.item())
        sm.ring_support_peg[constants.RING_NAMES[-1]][0] = constants.OUTSIDE_RIGHT
        result = success.task_success(
            sm, "phase_4", constants.RING_NAMES, 1, torch.device("cpu")
        )
        self.assertFalse(result.item())


class TCCSuccessGateTests(unittest.TestCase):
    def test_requires_consecutive_below_threshold_observations(self) -> None:
        gate = phase_router.TCCSuccessGate(
            threshold=10.0, consecutive_steps=3, min_steps=0
        )
        self.assertFalse(gate.update(9.0, phase_step=1))
        self.assertFalse(gate.update(8.0, phase_step=2))
        self.assertTrue(gate.update(10.0, phase_step=3))

    def test_above_threshold_resets_streak(self) -> None:
        gate = phase_router.TCCSuccessGate(
            threshold=10.0, consecutive_steps=2, min_steps=0
        )
        self.assertFalse(gate.update(9.0, phase_step=1))
        self.assertFalse(gate.update(10.1, phase_step=2))
        self.assertFalse(gate.update(9.0, phase_step=3))
        self.assertTrue(gate.update(9.0, phase_step=4))

    def test_minimum_phase_steps_blocks_early_transition(self) -> None:
        gate = phase_router.TCCSuccessGate(
            threshold=10.0, consecutive_steps=2, min_steps=5
        )
        self.assertFalse(gate.update(1.0, phase_step=3))
        self.assertFalse(gate.update(1.0, phase_step=4))
        self.assertFalse(gate.update(1.0, phase_step=5))
        self.assertTrue(gate.update(1.0, phase_step=6))

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            phase_router.TCCSuccessGate(threshold=-1.0)
        with self.assertRaises(ValueError):
            phase_router.TCCSuccessGate(threshold=1.0, consecutive_steps=0)

    def test_ineligible_observation_resets_streak(self) -> None:
        gate = phase_router.TCCSuccessGate(
            threshold=1.0, consecutive_steps=2, min_steps=0
        )
        self.assertFalse(gate.update(0.5, 1, eligible=True))
        self.assertEqual(gate.streak, 1)
        self.assertFalse(gate.update(0.5, 2, eligible=False))
        self.assertEqual(gate.streak, 0)
        self.assertFalse(gate.update(0.5, 3, eligible=True))
        self.assertTrue(gate.update(0.5, 4, eligible=True))


class StateMachineTimeoutTests(unittest.TestCase):
    def test_phase1_initial_layouts_preserve_all_rings(self) -> None:
        order = list(constants.RING_NAMES)
        expected_counts = {
            "stack_green": (0, 0, 4),
            "red3_blue1": (3, 1, 0),
            "red1_blue3": (1, 3, 0),
            "red2_blue1_green1": (2, 1, 1),
            "red1_blue2_green1": (1, 2, 1),
        }

        for initial_state, counts in expected_counts.items():
            stacks = state_machine.SPARTANStateMachine.build_phase1_initial_stacks(
                initial_state,
                order,
            )
            flattened = [ring for rings in stacks.values() for ring in rings]
            self.assertEqual(flattened, order)
            self.assertEqual(len(stacks.get(constants.PEG_RED, [])), counts[0])
            self.assertEqual(len(stacks.get(constants.PEG_BLUE, [])), counts[1])
            self.assertEqual(len(stacks.get(constants.PEG_GREEN, [])), counts[2])

    def test_phase1_initial_layout_rejects_duplicate_ring_order(self) -> None:
        with self.assertRaises(ValueError):
            state_machine.SPARTANStateMachine.build_phase1_initial_stacks(
                "stack_green",
                [constants.RING_NAMES[0]] * 4,
            )

    def test_phase1_curriculum_states_resolve_to_concrete_layouts(self) -> None:
        recovery_states = {
            state_machine.SPARTANStateMachine.resolve_phase1_initial_state(
                "recovery_mixed"
            )
            for _ in range(100)
        }
        self.assertEqual(recovery_states, {"red3_blue1", "red1_blue3"})

        curriculum_states = {
            state_machine.SPARTANStateMachine.resolve_phase1_initial_state(
                "mixed_curriculum"
            )
            for _ in range(200)
        }
        self.assertIn("stack_green", curriculum_states)
        self.assertTrue(
            curriculum_states.issubset(
                {
                    "stack_green",
                    "red3_blue1",
                    "red1_blue3",
                    "red2_blue1_green1",
                    "red1_blue2_green1",
                }
            )
        )

    def test_phase2_recovery_layouts_preserve_all_rings(self) -> None:
        order = list(constants.RING_NAMES)
        expected_counts = {
            "red2_blue2": (2, 2, 0),
            "center2": (1, 1, 2),
        }
        for initial_state, counts in expected_counts.items():
            stacks = state_machine.SPARTANStateMachine.build_phase2_initial_stacks(
                initial_state,
                order,
            )
            flattened = [ring for rings in stacks.values() for ring in rings]
            self.assertEqual(flattened, order)
            self.assertEqual(len(stacks.get(constants.PEG_RED, [])), counts[0])
            self.assertEqual(len(stacks.get(constants.PEG_BLUE, [])), counts[1])
            self.assertEqual(len(stacks.get(constants.BOARD_CENTER, [])), counts[2])

        center3_sources = set()
        for _ in range(100):
            stacks = state_machine.SPARTANStateMachine.build_phase2_initial_stacks(
                "center3",
                order,
            )
            self.assertEqual(len(stacks[constants.BOARD_CENTER]), 3)
            source = next(
                peg
                for peg in (constants.PEG_RED, constants.PEG_BLUE)
                if peg in stacks
            )
            center3_sources.add(source)
        self.assertEqual(center3_sources, {constants.PEG_RED, constants.PEG_BLUE})

    def test_phase2_recovery_mixture_resolves_to_late_states(self) -> None:
        states = {
            state_machine.SPARTANStateMachine.resolve_phase2_initial_state(
                "recovery_mixed"
            )
            for _ in range(100)
        }
        self.assertEqual(states, {"center2", "center3"})

    def test_park_moves_to_entry_then_returns_idle(self) -> None:
        machine = state_machine.SPARTANStateMachine.__new__(
            state_machine.SPARTANStateMachine
        )
        machine.device = torch.device("cpu")
        machine.debug_mode = False
        machine._get_reach_entry_position = lambda subject, env_id: torch.ones(3)
        machine._tip_reached = lambda tip, dest: True
        machine._get_fixed_speed_cmd = lambda tip, dest: dest - tip

        active = ["park"]
        sub_state = ["MOVE_TO_REACH_ENTRY"]
        step_counter = [0]
        current = [{"verb": "park", "subject": "left_arm", "target": "None"}]
        act = torch.zeros((1, 7))

        machine._process_arm_logic(
            0,
            active,
            sub_state,
            step_counter,
            current,
            [None],
            [None],
            torch.zeros((1, 3)),
            torch.ones(1),
            [None],
            [False],
            torch.ones((1, 3)),
            act,
            torch.ones((1, 2)),
            "left_arm",
        )

        self.assertEqual(sub_state[0], "IDLE")
        self.assertIsNone(current[0])
        torch.testing.assert_close(act[0, :3], torch.zeros(3))

    def test_control_reset_preserves_ring_state_for_phase_transition(self) -> None:
        machine = state_machine.SPARTANStateMachine.__new__(
            state_machine.SPARTANStateMachine
        )
        machine.device = torch.device("cpu")
        machine.current_triplet_r = [{"verb": "release"}]
        machine.current_triplet_l = [{"verb": "idle"}]
        machine.sub_state_r = ["OPEN_GRIPPER"]
        machine.sub_state_l = ["IDLE"]
        machine.step_counter_r = [3]
        machine.step_counter_l = [0]
        machine.active_verb_r = ["release"]
        machine.active_verb_l = ["idle"]
        machine.attached_target_r = [None]
        machine.attached_target_l = [None]
        machine.last_target_r = [constants.PEG_GREEN]
        machine.last_target_l = [None]
        machine.release_target_r = [torch.ones(3)]
        machine.release_target_l = [None]
        machine.release_committed_r = [True]
        machine.release_committed_l = [False]
        machine.last_reach_failed_r = [True]
        machine.last_reach_failed_l = [True]
        machine.target_pos_r = torch.ones((1, 3))
        machine.target_pos_l = torch.ones((1, 3))
        machine.current_gripper_state_r = torch.tensor([-1.0])
        machine.current_gripper_state_l = torch.tensor([-1.0])

        machine.frozen_rings_mask = {
            name: torch.tensor([True]) for name in constants.RING_NAMES
        }
        machine.frozen_rings_pose = {
            name: torch.full((1, 7), float(index + 1))
            for index, name in enumerate(constants.RING_NAMES)
        }
        machine.ring_support_peg = {
            name: [constants.PEG_GREEN] for name in constants.RING_NAMES
        }
        machine.ring_stack_level = {
            name: [index] for index, name in enumerate(constants.RING_NAMES)
        }
        machine.peg_inventory = {
            constants.PEG_GREEN: torch.tensor([4], dtype=torch.long)
        }
        frozen_before = {
            name: value.clone() for name, value in machine.frozen_rings_mask.items()
        }
        poses_before = {
            name: value.clone() for name, value in machine.frozen_rings_pose.items()
        }
        support_before = {
            name: list(value) for name, value in machine.ring_support_peg.items()
        }

        machine.reset_control_state(0)

        self.assertEqual(machine.sub_state_r[0], "IDLE")
        self.assertEqual(machine.sub_state_l[0], "IDLE")
        self.assertIsNone(machine.current_triplet_r[0])
        self.assertIsNone(machine.current_triplet_l[0])
        self.assertEqual(machine.current_gripper_state_r[0].item(), 1.0)
        self.assertEqual(machine.current_gripper_state_l[0].item(), 1.0)
        self.assertFalse(machine.last_reach_failed_r[0])
        self.assertFalse(machine.last_reach_failed_l[0])
        for name in constants.RING_NAMES:
            torch.testing.assert_close(machine.frozen_rings_mask[name], frozen_before[name])
            torch.testing.assert_close(machine.frozen_rings_pose[name], poses_before[name])
            self.assertEqual(machine.ring_support_peg[name], support_before[name])
        self.assertEqual(machine.peg_inventory[constants.PEG_GREEN].item(), 4)

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
        machine.last_reach_failed_r = [False]
        machine.last_reach_failed_l = [False]

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
        self.assertTrue(machine.last_reach_failed_l[0])

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

    def test_idle_idle_is_blocked_while_task_has_work(self) -> None:
        obs = base_observation()
        idle = constants.VERB_TO_ID["idle"]
        none = constants.TARGET_TO_ID["None"]
        idle_action = [idle, none, idle, none]
        mask = self.mask(obs)
        self.assertFalse(mask[0, action_index(idle_action)])

        obs[:, 240:244] = 0
        mask = self.mask(obs)
        self.assertTrue(mask[0, action_index(idle_action)])

    def test_phase_two_allows_idle_idle_while_task_has_work(self) -> None:
        obs = base_observation()
        idle = constants.VERB_TO_ID["idle"]
        none = constants.TARGET_TO_ID["None"]
        mask = masked_policy.build_joint_action_mask(
            obs, self.action_table, "phase_2"
        )
        self.assertTrue(mask[0, action_index([idle, none, idle, none])])

    def test_minimal_mask_allows_idle_idle_in_phase_zero(self) -> None:
        obs = base_observation()
        idle = constants.VERB_TO_ID["idle"]
        none = constants.TARGET_TO_ID["None"]
        mask = masked_policy.build_joint_action_mask(
            obs, self.action_table, "phase_0", action_mask_mode="minimal"
        )
        self.assertTrue(mask[0, action_index([idle, none, idle, none])])

    def test_minimal_mask_allows_reaching_another_ring_while_holding(self) -> None:
        obs = base_observation()
        obs[:, 228] = 1  # left arm holds ring_red
        reach = constants.VERB_TO_ID["reach"]
        idle = constants.VERB_TO_ID["idle"]
        blue = constants.TARGET_TO_ID["ring_blue"]
        none = constants.TARGET_TO_ID["None"]
        guided = self.mask(obs)
        minimal = masked_policy.build_joint_action_mask(
            obs, self.action_table, "phase_0", action_mask_mode="minimal"
        )
        action = action_index([reach, blue, idle, none])
        self.assertFalse(guided[0, action])
        self.assertTrue(minimal[0, action])

    def test_minimal_mask_allows_empty_arm_to_reach_a_peg(self) -> None:
        obs = base_observation()
        reach = constants.VERB_TO_ID["reach"]
        idle = constants.VERB_TO_ID["idle"]
        yellow_peg = constants.TARGET_TO_ID["peg_yellow"]
        none = constants.TARGET_TO_ID["None"]
        minimal = masked_policy.build_joint_action_mask(
            obs, self.action_table, "phase_0", action_mask_mode="minimal"
        )
        self.assertTrue(
            minimal[0, action_index([reach, yellow_peg, idle, none])]
        )

    def test_goal_agnostic_mask_blocks_empty_peg_reach_but_not_destination(self) -> None:
        obs = base_observation()
        reach = constants.VERB_TO_ID["reach"]
        release = constants.VERB_TO_ID["release"]
        idle = constants.VERB_TO_ID["idle"]
        yellow_peg = constants.TARGET_TO_ID["peg_yellow"]
        none = constants.TARGET_TO_ID["None"]
        goal_agnostic = masked_policy.build_joint_action_mask(
            obs, self.action_table, "phase_0", action_mask_mode="goal_agnostic"
        )
        self.assertFalse(
            goal_agnostic[0, action_index([reach, yellow_peg, idle, none])]
        )

        obs[:, 228] = 1  # left arm holds ring_red
        obs[:, 124:152] = 0
        put_action_one_hot(obs, 124, [reach, yellow_peg, idle, none])
        goal_agnostic = masked_policy.build_joint_action_mask(
            obs, self.action_table, "phase_0", action_mask_mode="goal_agnostic"
        )
        self.assertTrue(
            goal_agnostic[0, action_index([release, yellow_peg, idle, none])]
        )

    def test_goal_agnostic_mask_retains_generic_carry_sequence(self) -> None:
        obs = base_observation()
        obs[:, 228] = 1  # left arm holds ring_red
        reach = constants.VERB_TO_ID["reach"]
        idle = constants.VERB_TO_ID["idle"]
        blue = constants.TARGET_TO_ID["ring_blue"]
        none = constants.TARGET_TO_ID["None"]
        goal_agnostic = masked_policy.build_joint_action_mask(
            obs, self.action_table, "phase_0", action_mask_mode="goal_agnostic"
        )
        self.assertFalse(
            goal_agnostic[0, action_index([reach, blue, idle, none])]
        )

    def test_minimal_mask_allows_non_goal_release_destination(self) -> None:
        obs = base_observation()
        reach = constants.VERB_TO_ID["reach"]
        release = constants.VERB_TO_ID["release"]
        idle = constants.VERB_TO_ID["idle"]
        yellow_peg = constants.TARGET_TO_ID["peg_yellow"]
        none = constants.TARGET_TO_ID["None"]
        obs[:, 228] = 1  # left arm holds ring_red
        obs[:, 124:152] = 0
        put_action_one_hot(obs, 124, [reach, yellow_peg, idle, none])
        guided = self.mask(obs)
        minimal = masked_policy.build_joint_action_mask(
            obs, self.action_table, "phase_0", action_mask_mode="minimal"
        )
        action = action_index([release, yellow_peg, idle, none])
        self.assertFalse(guided[0, action])
        self.assertTrue(minimal[0, action])

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

    def test_phase_two_release_uses_only_board_center_slot(self) -> None:
        obs = base_observation()
        reach = constants.VERB_TO_ID["reach"]
        release = constants.VERB_TO_ID["release"]
        idle = constants.VERB_TO_ID["idle"]
        none = constants.TARGET_TO_ID["None"]
        center = constants.BOARD_CENTER_TARGET_ID
        green = constants.TARGET_TO_ID["peg_green"]
        obs[:, 228] = 1  # left arm holds ring_red
        obs[:, 124:152] = 0
        put_action_one_hot(obs, 124, [reach, center, idle, none])
        mask = masked_policy.build_joint_action_mask(
            obs, self.action_table, "phase_2"
        )
        self.assertTrue(mask[0, action_index([release, center, idle, none])])
        self.assertFalse(mask[0, action_index([release, green, idle, none])])

    def test_phase_three_guided_release_accepts_all_coloured_pegs_not_gray(self) -> None:
        obs = base_observation()
        reach = constants.VERB_TO_ID["reach"]
        release = constants.VERB_TO_ID["release"]
        idle = constants.VERB_TO_ID["idle"]
        none = constants.TARGET_TO_ID["None"]
        obs[:, 228] = 1  # left arm holds ring_red

        for peg_name in constants.PEG_NAMES:
            peg_id = constants.TARGET_TO_ID[peg_name]
            obs[:, 124:152] = 0
            put_action_one_hot(obs, 124, [reach, peg_id, idle, none])
            mask = masked_policy.build_joint_action_mask(
                obs, self.action_table, "phase_3"
            )
            self.assertTrue(mask[0, action_index([release, peg_id, idle, none])])

        gray = constants.TARGET_TO_ID["peg_gray"]
        obs[:, 124:152] = 0
        put_action_one_hot(obs, 124, [reach, gray, idle, none])
        mask = masked_policy.build_joint_action_mask(
            obs, self.action_table, "phase_3"
        )
        self.assertFalse(mask[0, action_index([release, gray, idle, none])])

    def test_phase_three_masks_right_arm_red_destination_as_unreachable(self) -> None:
        obs = base_observation()
        reach = constants.VERB_TO_ID["reach"]
        idle = constants.VERB_TO_ID["idle"]
        none = constants.TARGET_TO_ID["None"]
        red_peg = constants.TARGET_TO_ID["peg_red"]
        mask = masked_policy.build_joint_action_mask(
            obs, self.action_table, "phase_3", action_mask_mode="minimal"
        )
        self.assertTrue(mask[0, action_index([reach, red_peg, idle, none])])
        self.assertFalse(mask[0, action_index([idle, none, reach, red_peg])])

    def test_phase_four_minimal_mask_exposes_only_reachable_side_per_arm(self) -> None:
        obs = base_observation()
        reach = constants.VERB_TO_ID["reach"]
        idle = constants.VERB_TO_ID["idle"]
        none = constants.TARGET_TO_ID["None"]
        outside_right = constants.OUTSIDE_RIGHT_TARGET_ID
        outside_left = constants.OUTSIDE_LEFT_TARGET_ID
        mask = masked_policy.build_joint_action_mask(
            obs, self.action_table, "phase_4", action_mask_mode="minimal"
        )
        self.assertTrue(
            mask[0, action_index([reach, outside_left, idle, none])]
        )
        self.assertFalse(
            mask[0, action_index([reach, outside_right, idle, none])]
        )
        self.assertTrue(
            mask[0, action_index([idle, none, reach, outside_right])]
        )
        self.assertFalse(
            mask[0, action_index([idle, none, reach, outside_left])]
        )

    def test_phase_two_masks_only_the_observed_infeasible_arm_stack_pair(self) -> None:
        obs = base_observation()
        reach = constants.VERB_TO_ID["reach"]
        idle = constants.VERB_TO_ID["idle"]
        none = constants.TARGET_TO_ID["None"]
        red_ring = constants.TARGET_TO_ID["ring_red"]
        blue_ring = constants.TARGET_TO_ID["ring_blue"]
        red_peg = torch.tensor([0.330, -0.025, 0.719])
        blue_peg = torch.tensor([0.360, -0.075, 0.719])
        obs[:, 176:179] = red_peg
        obs[:, 185:188] = blue_peg
        obs[:, 164:167] = red_peg
        obs[:, 173:176] = blue_peg
        mask = masked_policy.build_joint_action_mask(
            obs, self.action_table, "phase_2"
        )
        self.assertTrue(
            mask[0, action_index([reach, red_ring, idle, none])]
        )
        self.assertFalse(
            mask[0, action_index([idle, none, reach, red_ring])]
        )
        self.assertTrue(
            mask[0, action_index([reach, blue_ring, idle, none])]
        )
        self.assertTrue(
            mask[0, action_index([idle, none, reach, blue_ring])]
        )
        minimal_mask = masked_policy.build_joint_action_mask(
            obs, self.action_table, "phase_2", action_mask_mode="minimal"
        )
        self.assertFalse(
            minimal_mask[0, action_index([idle, none, reach, red_ring])]
        )
        self.assertTrue(
            minimal_mask[0, action_index([idle, none, reach, blue_ring])]
        )

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

    def test_soft_mask_penalty_reduces_but_does_not_remove_new_actions(self) -> None:
        obs = base_observation()
        logits = torch.zeros((1, 28))
        guided = masked_policy.build_joint_action_mask(
            obs, self.action_table, "phase_0", action_mask_mode="guided"
        )
        minimal = masked_policy.build_joint_action_mask(
            obs, self.action_table, "phase_0", action_mask_mode="minimal"
        )
        newly_exposed = minimal & ~guided
        self.assertTrue(newly_exposed.any())

        distribution = masked_policy.MaskedJointMultiCategoricalDistribution(
            [4, 10, 4, 10],
            task_phase="phase_0",
            action_mask_mode="minimal",
            soft_mask_penalty=8.0,
        )
        distribution.proba_distribution(logits, obs)
        penalized = distribution.joint_distribution.probs[0, newly_exposed[0]].sum()
        self.assertGreater(float(penalized), 0.0)

        distribution.set_soft_mask_penalty(0.0)
        distribution.proba_distribution(logits, obs)
        unpenalized = distribution.joint_distribution.probs[0, newly_exposed[0]].sum()
        self.assertLess(float(penalized), float(unpenalized))
        self.assertLess(float(penalized), 0.01)

    def test_soft_mask_penalty_rejects_negative_values(self) -> None:
        distribution = masked_policy.MaskedJointMultiCategoricalDistribution(
            [4, 10, 4, 10], task_phase="phase_0"
        )
        with self.assertRaises(ValueError):
            distribution.set_soft_mask_penalty(-1.0)


if __name__ == "__main__":
    unittest.main()
