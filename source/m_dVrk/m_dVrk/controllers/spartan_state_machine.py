"""SPARTAN hierarchical controller (state machine) for the m_dVrk task.

This module contains the ``SPARTANStateMachine`` class, which interprets
high-level HRL triplets (verb, subject, target) and generates low-level
joint-space commands for the dVRK PSM arms inside Isaac Sim.

The state machine is the sole owner of:
  - arm sub-states (IDLE, MOVE_TO_REACH_ENTRY, APPROACH_ABOVE, DESCEND,
    SETTLE, CLOSE_GRIPPER, LIFT_UP, OPEN_GRIPPER)
  - ring attachment and frozen-placement bookkeeping
  - peg inventory counts
  - per-environment reset logic (phase 0 and phase 1)
"""

from __future__ import annotations

import random

import torch
import isaaclab.utils.math as math_utils

from m_dVrk.controllers.ring_sync import (
    get_scene_entity_positions_w,
    get_scene_entity_position_w,
    _get_scene_entity_position_w,
)
from m_dVrk.hrl.constants import RING_NAMES, ALL_PEG_NAMES, PEG_GREEN, PEG_RED, PEG_BLUE


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

ARM_IDLE_STATE = "IDLE"

INITIAL_PHASE_BY_VERB: dict[str, str] = {
    "reach":   "MOVE_TO_REACH_ENTRY",
    "grasp":   "CLOSE_GRIPPER",
    "release": "OPEN_GRIPPER",
}

VALID_PHASES_BY_VERB: dict[str, set[str]] = {
    "reach":   {"MOVE_TO_REACH_ENTRY", "APPROACH_ABOVE", "DESCEND", "SETTLE"},
    "grasp":   {"CLOSE_GRIPPER", "LIFT_UP"},
    "release": {"OPEN_GRIPPER"},
}


# ---------------------------------------------------------------------------
# SPARTANStateMachine
# ---------------------------------------------------------------------------

class SPARTANStateMachine:
    """Vectorised high-level controller for ``num_envs`` parallel dVRK setups.

    Each environment has two PSM arms (left and right). The machine accepts
    *triplet* commands of the form ``{verb, subject, target}`` and advances
    through a sequence of motion phases to execute them.

    Physical constants (Z heights, tolerances, speeds) are class attributes
    so they can be overridden for ablations without touching the logic.
    """

    Z_TABLE: float = 0.717
    SAFE_Z_OFFSET: float = 0.015
    GRASP_Z_OFFSET: float = 0.005
    GRASP_ATTACH_DISTANCE: float = 0.015
    GRASP_CLOSE_MIN_STEPS: int = 10
    GRASP_TIMEOUT_STEPS: int = 220
    RELEASE_PEG_CAPTURE_DISTANCE: float = 0.015
    BOARD_RELEASE_Z_OFFSET: float = 0.0
    LINEAR_SPEED: float = 0.02
    REACH_ENTRY_X: float = 0.375
    REACH_ENTRY_Y: float = -0.05
    REACH_ENTRY_LATERAL_OFFSET: float = 0.03
    REACH_ENTRY_Z: float = 0.76
    TOLERANCE_XY: float = 0.006
    TOLERANCE_Z: float = 0.006
    RING_THICKNESS: float = 0.001

    def __init__(self, env, debug_mode: bool = False) -> None:
        self.debug_mode = debug_mode
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs

        # --- Arm state ---
        self.current_triplet_r: list = [None] * self.num_envs
        self.current_triplet_l: list = [None] * self.num_envs
        self.sub_state_r: list[str] = [ARM_IDLE_STATE] * self.num_envs
        self.sub_state_l: list[str] = [ARM_IDLE_STATE] * self.num_envs
        self.step_counter_r: list[int] = [0] * self.num_envs
        self.step_counter_l: list[int] = [0] * self.num_envs
        self.active_verb_r: list[str] = ["idle"] * self.num_envs
        self.active_verb_l: list[str] = ["idle"] * self.num_envs

        self.target_pos_r = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_pos_l = torch.zeros((self.num_envs, 3), device=self.device)

        self.current_gripper_state_r = torch.ones(self.num_envs, device=self.device)
        self.current_gripper_state_l = torch.ones(self.num_envs, device=self.device)

        self.attached_target_r: list = [None] * self.num_envs
        self.attached_target_l: list = [None] * self.num_envs
        self.last_target_r: list = [None] * self.num_envs
        self.last_target_l: list = [None] * self.num_envs
        self.release_target_r: list = [None] * self.num_envs
        self.release_target_l: list = [None] * self.num_envs
        self.release_committed_r: list[bool] = [False] * self.num_envs
        self.release_committed_l: list[bool] = [False] * self.num_envs

        # --- Ring / peg bookkeeping ---
        self.peg_inventory: dict[str, torch.Tensor] = {
            peg: torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            for peg in ALL_PEG_NAMES
        }
        self.frozen_rings_mask: dict[str, torch.Tensor] = {
            ring: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for ring in RING_NAMES
        }
        self.frozen_rings_pose: dict[str, torch.Tensor] = {
            ring: torch.zeros((self.num_envs, 7), device=self.device)
            for ring in RING_NAMES
        }
        self.ring_support_peg: dict[str, list] = {
            ring: [None] * self.num_envs for ring in RING_NAMES
        }
        self.ring_stack_level: dict[str, torch.Tensor] = {
            ring: torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
            for ring in RING_NAMES
        }

        # --- Robot tip indices ---
        robot_r = env.scene["robot_right"]
        robot_l = env.scene["robot_left"]
        self._tip_idx_r = robot_r.find_bodies("psm_tool_tip_link")[0][0]
        self._tip_idx_l = robot_l.find_bodies("psm_tool_tip_link")[0][0]
        self.target_quat_r = self._get_tip_quat_in_root(robot_r, self._tip_idx_r).clone()
        self.target_quat_l = self._get_tip_quat_in_root(robot_l, self._tip_idx_l).clone()

        # Public aliases are provided as class-level properties below.

        # --- Cached peg positions ---
        self._peg_name_to_index: dict[str, int] = {
            name: idx for idx, name in enumerate(ALL_PEG_NAMES)
        }
        self._cached_peg_positions_w = self._build_cached_peg_positions_w()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_cached_peg_positions_w(self) -> torch.Tensor:
        cached_positions = [
            get_scene_entity_positions_w(self.env, peg_name)
            for peg_name in ALL_PEG_NAMES
        ]
        return torch.stack(cached_positions, dim=1)

    @property
    def tip_idx_r(self) -> int:
        return self._tip_idx_r

    @property
    def tip_idx_l(self) -> int:
        return self._tip_idx_l

    def get_tip_idx_r(self) -> int:
        return int(self._tip_idx_r)

    def get_tip_idx_l(self) -> int:
        return int(self._tip_idx_l)

    def _get_cached_peg_position_w(
        self, peg_name: str, env_id: int
    ) -> torch.Tensor | None:
        peg_index = self._peg_name_to_index.get(peg_name)
        if peg_index is None:
            return None
        return self._cached_peg_positions_w[env_id, peg_index]

    def _set_phase(self, sub_state, step_counter, env_id: int, phase: str) -> None:
        if self.debug_mode:
            print(
                f"[DEBUG SM] Env {env_id} - Transition to phase: {phase} "
                f"(took {step_counter[env_id]} steps)"
            )
        sub_state[env_id] = phase
        step_counter[env_id] = 0

    def _finish_command(
        self,
        active_verb,
        sub_state,
        step_counter,
        env_id: int,
        current_triplet=None,
    ) -> None:
        if self.debug_mode:
            print(
                f"[DEBUG SM] Env {env_id} - Finished command {active_verb[env_id]}. "
                f"Going to IDLE. (took {step_counter[env_id]} steps)"
            )
        active_verb[env_id] = "idle"
        sub_state[env_id] = ARM_IDLE_STATE
        step_counter[env_id] = 0
        if current_triplet is not None:
            current_triplet[env_id] = None

    def _get_tip_quat_in_root(self, robot, tip_idx) -> torch.Tensor:
        _, tip_quat_b = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            robot.data.body_pos_w[:, tip_idx],
            robot.data.body_quat_w[:, tip_idx],
        )
        return tip_quat_b

    def _tip_reached(self, tip_pos: torch.Tensor, dest: torch.Tensor) -> bool:
        return (
            torch.norm(dest[0:2] - tip_pos[0:2]).item() < self.TOLERANCE_XY
            and torch.abs(dest[2] - tip_pos[2]).item() < self.TOLERANCE_Z
        )

    def _get_fixed_speed_cmd(
        self, current_pos: torch.Tensor, dest: torch.Tensor
    ) -> torch.Tensor:
        delta = dest - current_pos
        distance = torch.norm(delta)
        if distance < 1e-8:
            return torch.zeros(3, device=self.device)
        if distance <= self.LINEAR_SPEED:
            return delta
        return (delta / distance) * self.LINEAR_SPEED

    def _get_reach_entry_position(self, subject: str, env_id: int) -> torch.Tensor:
        origin = self.env.scene.env_origins[env_id]
        side_sign = -1.0 if subject == "right_arm" else 1.0
        local_entry = torch.tensor(
            [
                self.REACH_ENTRY_X,
                self.REACH_ENTRY_Y + side_sign * self.REACH_ENTRY_LATERAL_OFFSET,
                self.REACH_ENTRY_Z,
            ],
            device=self.device,
        )
        return origin + local_entry

    def _resolve_release_target(
        self, target_name: str | None, last_target: str | None
    ) -> str | None:
        if isinstance(target_name, str) and target_name.startswith("peg_"):
            return target_name
        if isinstance(last_target, str) and last_target.startswith("peg_"):
            return last_target
        return None

    def _is_peg_target(self, target_name: str | None) -> bool:
        return isinstance(target_name, str) and target_name.startswith("peg_")

    def _get_reach_settle_offset(self, target_name: str) -> float:
        return self.SAFE_Z_OFFSET if self._is_peg_target(target_name) else self.GRASP_Z_OFFSET

    def _find_release_peg(
        self,
        release_pos_w: torch.Tensor,
        env_id: int,
        preferred_peg: str | None = None,
    ) -> str | None:
        if preferred_peg is not None:
            peg_pos_w = self._get_cached_peg_position_w(preferred_peg, env_id)
            if peg_pos_w is None:
                peg_pos_w = _get_scene_entity_position_w(self.env, preferred_peg, env_id)
            xy_distance = torch.norm(release_pos_w[0:2] - peg_pos_w[0:2]).item()
            if self.debug_mode:
                print(
                    f"[DEBUG SM] Env {env_id} - Release target: {preferred_peg} "
                    f"- XY dist: {xy_distance:.4f}m (tol: {self.RELEASE_PEG_CAPTURE_DISTANCE}m)"
                )
            if xy_distance <= self.RELEASE_PEG_CAPTURE_DISTANCE:
                return preferred_peg

        best_peg: str | None = None
        best_xy_distance: float | None = None
        for peg_name in self.peg_inventory:
            if peg_name == preferred_peg:
                continue
            peg_pos_w = self._get_cached_peg_position_w(peg_name, env_id)
            if peg_pos_w is None:
                peg_pos_w = _get_scene_entity_position_w(self.env, peg_name, env_id)
            xy_distance = torch.norm(release_pos_w[0:2] - peg_pos_w[0:2]).item()
            if xy_distance > self.RELEASE_PEG_CAPTURE_DISTANCE:
                continue
            if best_xy_distance is None or xy_distance < best_xy_distance:
                best_peg = peg_name
                best_xy_distance = xy_distance

        return best_peg

    def _clear_ring_support(self, ring_name: str, env_id: int) -> None:
        self.ring_support_peg[ring_name][env_id] = None
        self.ring_stack_level[ring_name][env_id] = -1

    def _count_rings_on_peg(self, env_id: int, peg_name: str) -> int:
        count = 0
        for ring_name in self.frozen_rings_mask:
            if (self.frozen_rings_mask[ring_name][env_id]
                    and self.ring_support_peg[ring_name][env_id] == peg_name):
                count += 1
        return count

    def _sync_peg_inventory(self, env_id: int, peg_name: str) -> None:
        self.peg_inventory[peg_name][env_id] = self._count_rings_on_peg(env_id, peg_name)

    def _is_top_ring_on_peg(self, ring_name: str, env_id: int) -> bool:
        peg_name = self.ring_support_peg[ring_name][env_id]
        if peg_name is None:
            return True
        ring_level = int(self.ring_stack_level[ring_name][env_id].item())
        top_level: int | None = None
        for candidate_name in self.ring_stack_level:
            if (self.frozen_rings_mask[candidate_name][env_id]
                    and self.ring_support_peg[candidate_name][env_id] == peg_name):
                candidate_level = int(self.ring_stack_level[candidate_name][env_id].item())
                if top_level is None or candidate_level > top_level:
                    top_level = candidate_level
        return top_level is None or ring_level == top_level

    def _detach_ring_from_peg(self, ring_name: str, env_id: int) -> None:
        peg_name = self.ring_support_peg[ring_name][env_id]
        if peg_name is None:
            self._clear_ring_support(ring_name, env_id)
            return
        removed_level = int(self.ring_stack_level[ring_name][env_id].item())
        self._clear_ring_support(ring_name, env_id)
        for candidate_name in self.ring_stack_level:
            if self.ring_support_peg[candidate_name][env_id] != peg_name:
                continue
            candidate_level = int(self.ring_stack_level[candidate_name][env_id].item())
            if candidate_level > removed_level:
                self.ring_stack_level[candidate_name][env_id] = candidate_level - 1
        self._sync_peg_inventory(env_id, peg_name)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def ring_count_on_peg(self, env_id: int, peg_name: str) -> int:
        """Return the number of frozen rings on *peg_name* for *env_id*."""
        return self._count_rings_on_peg(env_id, peg_name)

    def get_top_rings_on_peg(
        self, env_id: int, peg_name: str, num_rings: int = 2
    ) -> list[str]:
        """Return the top *num_rings* ring names on *peg_name*, highest first."""
        rings_on_peg = []
        for ring_name in RING_NAMES:
            if (self.ring_support_peg[ring_name][env_id] == peg_name
                    and self.frozen_rings_mask[ring_name][env_id]):
                rings_on_peg.append(
                    (ring_name, int(self.ring_stack_level[ring_name][env_id].item()))
                )
        rings_on_peg.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in rings_on_peg[:num_rings]]

    def arm_idle(self, env_id: int, arm: str) -> bool:
        """Return ``True`` if *arm* is idle (no active triplet, sub-state IDLE)."""
        if arm == "right_arm":
            return (
                self.sub_state_r[env_id] == ARM_IDLE_STATE
                and self.current_triplet_r[env_id] is None
            )
        return (
            self.sub_state_l[env_id] == ARM_IDLE_STATE
            and self.current_triplet_l[env_id] is None
        )

    def all_idle(self, env_id: int) -> bool:
        """Return ``True`` if both arms are idle for *env_id*."""
        return self.arm_idle(env_id, "right_arm") and self.arm_idle(env_id, "left_arm")

    def set_new_triplet(
        self, verb: str, subject: str, target: str, env_id: int
    ) -> None:
        """Accept a new HRL command for *subject* in environment *env_id*.

        If the arm is already executing the same command, this is a no-op.
        If the verb is ``"idle"``, the arm is immediately transitioned to IDLE.
        """
        new_cmd = {"verb": verb, "subject": subject, "target": target}

        if verb == "grasp":
            if target not in RING_NAMES:
                return
            last_target = (
                self.last_target_r[env_id]
                if subject == "right_arm"
                else self.last_target_l[env_id]
            )
            if last_target != target:
                return

        if verb == "release":
            if target == "None" or not target.startswith("peg_"):
                return
            attached_target = (
                self.attached_target_r[env_id]
                if subject == "right_arm"
                else self.attached_target_l[env_id]
            )
            last_target = (
                self.last_target_r[env_id]
                if subject == "right_arm"
                else self.last_target_l[env_id]
            )
            if attached_target is None or last_target != target:
                return

        if subject == "right_arm":
            if (
                self.current_triplet_r[env_id] == new_cmd
                and self.active_verb_r[env_id] == verb
                and self.sub_state_r[env_id] != ARM_IDLE_STATE
            ):
                return

            self.current_triplet_r[env_id] = new_cmd
            self.active_verb_r[env_id] = verb
            self.target_pos_r[env_id] = 0.0
            self.release_target_r[env_id] = None
            self.release_committed_r[env_id] = False

            if verb == "idle":
                self._finish_command(
                    self.active_verb_r, self.sub_state_r,
                    self.step_counter_r, env_id, self.current_triplet_r,
                )
                return

            self._set_phase(self.sub_state_r, self.step_counter_r, env_id, INITIAL_PHASE_BY_VERB[verb])
            if verb in ("reach", "grasp"):
                self.target_pos_r[env_id] = self.get_target_coordinates(target, subject, env_id)
            elif verb == "release":
                self.release_target_r[env_id] = self._resolve_release_target(target, self.last_target_r[env_id])
                self.target_pos_r[env_id] = self.get_target_coordinates(target, subject, env_id)

        else:  # left_arm
            if (
                self.current_triplet_l[env_id] == new_cmd
                and self.active_verb_l[env_id] == verb
                and self.sub_state_l[env_id] != ARM_IDLE_STATE
            ):
                return

            self.current_triplet_l[env_id] = new_cmd
            self.active_verb_l[env_id] = verb
            self.target_pos_l[env_id] = 0.0
            self.release_target_l[env_id] = None
            self.release_committed_l[env_id] = False

            if verb == "idle":
                self._finish_command(
                    self.active_verb_l, self.sub_state_l,
                    self.step_counter_l, env_id, self.current_triplet_l,
                )
                return

            self._set_phase(self.sub_state_l, self.step_counter_l, env_id, INITIAL_PHASE_BY_VERB[verb])
            if verb in ("reach", "grasp"):
                self.target_pos_l[env_id] = self.get_target_coordinates(target, subject, env_id)
            elif verb == "release":
                self.release_target_l[env_id] = self._resolve_release_target(target, self.last_target_l[env_id])
                self.target_pos_l[env_id] = self.get_target_coordinates(target, subject, env_id)

    def get_target_coordinates(
        self, target_name: str, subject: str, env_id: int
    ) -> torch.Tensor:
        """Resolve *target_name* to a world-space 3D position for *env_id*."""
        origin = self.env.scene.env_origins[env_id]

        if target_name == "None" or target_name is None:
            return self._get_reach_entry_position(subject, env_id)

        sign_arm = 1.0 if subject == "right_arm" else -1.0
        offset = torch.tensor([-0.005 * sign_arm, 0.0, 0.0], device=self.device)

        cached_peg_pos = self._get_cached_peg_position_w(target_name, env_id)
        if cached_peg_pos is not None:
            return cached_peg_pos + offset

        if target_name in self.env.scene.keys():
            try:
                return _get_scene_entity_position_w(self.env, target_name, env_id) + offset
            except RuntimeError as exc:
                print(f"[ENV {env_id}] Warning: {exc}")

        print(
            f"[ENV {env_id}] Warning: cannot resolve position of '{target_name}' "
            f"for '{subject}'. Using fallback."
        )
        return origin + torch.tensor([0.5, 0.0, 0.5], device=self.device)

    def reset_env(self, env_id: int, phase: str = "phase_0") -> None:
        """Reset the state machine for environment *env_id*.

        For ``phase_1``, all 4 rings are pre-frozen on ``peg_green`` in a
        random stacking order (matching the evaluation initial condition).
        """
        self.current_triplet_r[env_id] = None
        self.current_triplet_l[env_id] = None
        self.sub_state_r[env_id] = ARM_IDLE_STATE
        self.sub_state_l[env_id] = ARM_IDLE_STATE
        self.step_counter_r[env_id] = 0
        self.step_counter_l[env_id] = 0
        self.active_verb_r[env_id] = "idle"
        self.active_verb_l[env_id] = "idle"
        self.attached_target_r[env_id] = None
        self.attached_target_l[env_id] = None
        self.last_target_r[env_id] = None
        self.last_target_l[env_id] = None
        self.release_target_r[env_id] = None
        self.release_target_l[env_id] = None
        self.release_committed_r[env_id] = False
        self.release_committed_l[env_id] = False
        self.target_pos_r[env_id] = torch.zeros(3, device=self.device)
        self.target_pos_l[env_id] = torch.zeros(3, device=self.device)
        self.current_gripper_state_r[env_id] = 1.0
        self.current_gripper_state_l[env_id] = 1.0

        for peg in self.peg_inventory:
            self.peg_inventory[peg][env_id] = 0
        for ring in self.frozen_rings_mask:
            self.frozen_rings_mask[ring][env_id] = False
            self.frozen_rings_pose[ring][env_id] = 0.0
            self._clear_ring_support(ring, env_id)

        if phase == "phase_1":
            rings_to_stack = RING_NAMES.copy()
            random.shuffle(rings_to_stack)
            peg_pos_w = self._get_cached_peg_position_w(PEG_GREEN, env_id)
            if peg_pos_w is None:
                peg_pos_w = _get_scene_entity_position_w(self.env, PEG_GREEN, env_id)

            for idx, ring_name in enumerate(rings_to_stack):
                release_pos = peg_pos_w.clone()
                release_pos[2] = (
                    self.Z_TABLE
                    + self.env.scene.env_origins[env_id, 2]
                    + idx * self.RING_THICKNESS
                )
                self.frozen_rings_mask[ring_name][env_id] = True
                self.frozen_rings_pose[ring_name][env_id, 0:3] = release_pos
                self.frozen_rings_pose[ring_name][env_id, 3:7] = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], device=self.device
                )
                self.ring_support_peg[ring_name][env_id] = PEG_GREEN
                self.ring_stack_level[ring_name][env_id] = idx

            self._sync_peg_inventory(env_id, PEG_GREEN)

        print(f"[ENV {env_id}] Reset complete ({phase}).")

    def green_peg_ring_count(self, env_id: int) -> int:
        """Convenience: count frozen rings on the green peg for *env_id*."""
        return self._count_rings_on_peg(env_id, PEG_GREEN)

    def successful(self, env_id: int, phase: str = "phase_0", target_peg: str = PEG_GREEN) -> bool:
        """Check success for a single environment (used by scripted collector)."""
        if phase == "phase_1":
            red_count = self._count_rings_on_peg(env_id, PEG_RED)
            blue_count = self._count_rings_on_peg(env_id, PEG_BLUE)
            all_frozen = all(
                self.frozen_rings_mask[ring_name][env_id] for ring_name in RING_NAMES
            )
            return all_frozen and red_count == 2 and blue_count == 2
        else:
            return all(
                self.frozen_rings_mask[ring_name][env_id]
                and self.ring_support_peg[ring_name][env_id] == target_peg
                for ring_name in RING_NAMES
            )

    # -----------------------------------------------------------------------
    # Low-level arm logic
    # -----------------------------------------------------------------------

    def _process_arm_logic(
        self, i, active_verb, sub_state, step_counter, current_triplet,
        last_target, attached_target, target_pos, curr_grip_state,
        release_target, release_committed, tip, act, grip, subject,
    ) -> None:
        """Advance the state machine for one arm in one environment."""
        if sub_state[i] == ARM_IDLE_STATE or current_triplet[i] is None:
            act[i, 0:3] = 0.0
            return

        verb = active_verb[i]
        phase = sub_state[i]
        valid_phases = VALID_PHASES_BY_VERB.get(verb)

        if valid_phases is None:
            self._finish_command(active_verb, sub_state, step_counter, i, current_triplet)
            act[i, 0:3] = 0.0
            return

        if phase not in valid_phases:
            self._set_phase(sub_state, step_counter, i, INITIAL_PHASE_BY_VERB[verb])
            phase = sub_state[i]

        if verb == "reach":
            target_name = current_triplet[i]["target"]
            settle_offset = self._get_reach_settle_offset(target_name)
            if phase == "MOVE_TO_REACH_ENTRY":
                dest = self._get_reach_entry_position(subject, i)
                act[i, 0:3] = self._get_fixed_speed_cmd(tip[i], dest)
                if self._tip_reached(tip[i], dest) or step_counter[i] > 400:
                    self._set_phase(sub_state, step_counter, i, "APPROACH_ABOVE")
            elif phase == "APPROACH_ABOVE":
                target_pos[i] = self.get_target_coordinates(target_name, subject, i)
                dest = target_pos[i] + torch.tensor([0.0, 0.0, self.SAFE_Z_OFFSET], device=self.device)
                act[i, 0:3] = self._get_fixed_speed_cmd(tip[i], dest)
                if self._tip_reached(tip[i], dest) or step_counter[i] > 400:
                    if self._is_peg_target(target_name):
                        self._set_phase(sub_state, step_counter, i, "SETTLE")
                    else:
                        self._set_phase(sub_state, step_counter, i, "DESCEND")
            elif phase == "DESCEND":
                target_pos[i] = self.get_target_coordinates(target_name, subject, i)
                dest = target_pos[i] + torch.tensor([0.0, 0.0, self.GRASP_Z_OFFSET], device=self.device)
                act[i, 0:3] = self._get_fixed_speed_cmd(tip[i], dest)
                if self._tip_reached(tip[i], dest):
                    self._set_phase(sub_state, step_counter, i, "SETTLE")
            elif phase == "SETTLE":
                target_pos[i] = self.get_target_coordinates(target_name, subject, i)
                dest = target_pos[i] + torch.tensor([0.0, 0.0, settle_offset], device=self.device)
                act[i, 0:3] = self._get_fixed_speed_cmd(tip[i], dest)
                if step_counter[i] > 15:
                    last_target[i] = target_name
                    self._finish_command(active_verb, sub_state, step_counter, i, current_triplet)

        elif verb == "grasp":
            if phase == "CLOSE_GRIPPER":
                curr_grip_state[i] = -1.0
                grip[i] = -1.0
                obj = current_triplet[i]["target"]
                grasp_target = self.get_target_coordinates(obj, subject, i)
                target_pos[i] = grasp_target
                dest = grasp_target + torch.tensor([0.0, 0.0, self.GRASP_Z_OFFSET], device=self.device)
                act[i, 0:3] = self._get_fixed_speed_cmd(tip[i], dest)

                if (
                    obj in self.frozen_rings_mask
                    and step_counter[i] >= self.GRASP_CLOSE_MIN_STEPS
                    and torch.norm(grasp_target - tip[i]) < self.GRASP_ATTACH_DISTANCE
                    and self._is_top_ring_on_peg(obj, i)
                ):
                    self._detach_ring_from_peg(obj, i)
                    self.frozen_rings_mask[obj][i] = False
                    attached_target[i] = obj
                    last_target[i] = obj
                    self._set_phase(sub_state, step_counter, i, "LIFT_UP")
                elif step_counter[i] > self.GRASP_TIMEOUT_STEPS:
                    curr_grip_state[i] = 1.0
                    grip[i] = 1.0
                    self._finish_command(active_verb, sub_state, step_counter, i, current_triplet)

            elif phase == "LIFT_UP":
                dest = target_pos[i] + torch.tensor([0.0, 0.0, self.SAFE_Z_OFFSET], device=self.device)
                act[i, 0:3] = self._get_fixed_speed_cmd(tip[i], dest)
                if self._tip_reached(tip[i], dest) or step_counter[i] > 250:
                    self._finish_command(active_verb, sub_state, step_counter, i, current_triplet)

        elif verb == "release":
            if phase == "OPEN_GRIPPER":
                curr_grip_state[i] = 1.0
                grip[i] = 1.0

                target_name = current_triplet[i]["target"]
                settle_offset = self._get_reach_settle_offset(target_name)
                dest = target_pos[i] + torch.tensor([0.0, 0.0, settle_offset], device=self.device)
                act[i, 0:3] = self._get_fixed_speed_cmd(tip[i], dest)

                if step_counter[i] == 10 and not release_committed[i]:
                    obj = attached_target[i]
                    if obj and obj != "None":
                        curr_pos = self.env.scene[obj].data.root_pos_w[i].clone()
                        peg = self._find_release_peg(curr_pos, i, release_target[i])
                        release_pos = curr_pos.clone()

                        if peg:
                            peg_pos = _get_scene_entity_position_w(self.env, peg, i).clone()
                            num_rings = self._count_rings_on_peg(i, peg)
                            release_pos[0:2] = peg_pos[0:2]
                            release_pos[2] = (
                                self.Z_TABLE
                                + self.env.scene.env_origins[i, 2]
                                + num_rings * self.RING_THICKNESS
                            )
                            self.ring_support_peg[obj][i] = peg
                            self.ring_stack_level[obj][i] = num_rings
                            self.peg_inventory[peg][i] = num_rings + 1
                        else:
                            release_pos[2] = (
                                self.Z_TABLE
                                + self.env.scene.env_origins[i, 2]
                                + self.BOARD_RELEASE_Z_OFFSET
                            )
                            self._clear_ring_support(obj, i)

                        self.frozen_rings_mask[obj][i] = True
                        self.frozen_rings_pose[obj][i, 0:3] = release_pos
                        self.frozen_rings_pose[obj][i, 3:7] = torch.tensor(
                            [1.0, 0.0, 0.0, 0.0], device=self.device
                        )
                    attached_target[i] = None
                    release_committed[i] = True

                if step_counter[i] > 20:
                    self._finish_command(active_verb, sub_state, step_counter, i, current_triplet)

        step_counter[i] += 1

    def get_action(self) -> torch.Tensor:
        """Compute the continuous joint-space action for all environments.

        Returns:
            Float tensor of shape ``(num_envs, 18)`` containing concatenated
            right-arm and left-arm actions (position + orientation + gripper).
        """
        act_r = torch.zeros((self.num_envs, 7), device=self.device)
        act_l = torch.zeros((self.num_envs, 7), device=self.device)

        grip_r = self.current_gripper_state_r.unsqueeze(1).repeat(1, 2)
        grip_l = self.current_gripper_state_l.unsqueeze(1).repeat(1, 2)

        robot_r = self.env.scene["robot_right"]
        robot_l = self.env.scene["robot_left"]
        tip_r = robot_r.data.body_pos_w[:, self._tip_idx_r]
        tip_l = robot_l.data.body_pos_w[:, self._tip_idx_l]

        for i in range(self.num_envs):
            self._process_arm_logic(
                i,
                self.active_verb_r, self.sub_state_r, self.step_counter_r,
                self.current_triplet_r, self.last_target_r, self.attached_target_r,
                self.target_pos_r, self.current_gripper_state_r,
                self.release_target_r, self.release_committed_r,
                tip_r, act_r, grip_r, "right_arm",
            )
            self._process_arm_logic(
                i,
                self.active_verb_l, self.sub_state_l, self.step_counter_l,
                self.current_triplet_l, self.last_target_l, self.attached_target_l,
                self.target_pos_l, self.current_gripper_state_l,
                self.release_target_l, self.release_committed_l,
                tip_l, act_l, grip_l, "left_arm",
            )

        desired_pos_r_w = tip_r + act_r[:, 0:3]
        desired_pos_l_w = tip_l + act_l[:, 0:3]

        desired_pos_r_b, _ = math_utils.subtract_frame_transforms(
            robot_r.data.root_pos_w, robot_r.data.root_quat_w, desired_pos_r_w, None,
        )
        desired_pos_l_b, _ = math_utils.subtract_frame_transforms(
            robot_l.data.root_pos_w, robot_l.data.root_quat_w, desired_pos_l_w, None,
        )

        act_r[:, 0:3] = desired_pos_r_b
        act_r[:, 3:7] = self.target_quat_r
        act_l[:, 0:3] = desired_pos_l_b
        act_l[:, 3:7] = self.target_quat_l

        return torch.cat([act_r, grip_r, act_l, grip_l], dim=-1)
