# """Shared dVRK parallel environment control primitives.

# This module intentionally stays free of AppLauncher side effects so that
# scripted utilities can reuse the same state machine logic without importing
# training entrypoints that bootstrap Isaac Sim at import time.
# """

# from __future__ import annotations

# import torch

# import isaaclab.utils.math as math_utils


# RING_NAMES = ["ring_red", "ring_yellow", "ring_green", "ring_blue"]
# PEG_NAMES = ["peg_red", "peg_yellow", "peg_green", "peg_blue", "peg_gray", "peg_gray1"]
# ARM_IDLE_STATE = "IDLE"
# INITIAL_PHASE_BY_VERB = {
#     "reach": "MOVE_TO_REACH_ENTRY",
#     "grasp": "CLOSE_GRIPPER",
#     "release": "OPEN_GRIPPER",
# }
# VALID_PHASES_BY_VERB = {
#     "reach": {"MOVE_TO_REACH_ENTRY", "APPROACH_ABOVE", "DESCEND", "SETTLE"},
#     "grasp": {"CLOSE_GRIPPER", "LIFT_UP"},
#     "release": {"OPEN_GRIPPER"},
# }


# def get_scene_entity_positions_w(env, entity_name: str) -> torch.Tensor:
#     """Resolve a scene entity to one world position per env."""
#     entity = env.scene[entity_name]
#     origins = env.scene.env_origins

#     if hasattr(entity, "data") and hasattr(entity.data, "root_pos_w"):
#         pos_w = entity.data.root_pos_w
#         if pos_w.shape[0] == env.num_envs:
#             return pos_w.clone()
#         if pos_w.shape[0] == 1:
#             local_pos = pos_w[0].clone() - origins[0]
#             return origins + local_pos

#     if hasattr(entity, "get_world_poses"):
#         view_count = getattr(entity, "count", None)
#         if view_count == env.num_envs:
#             try:
#                 pos_w, _ = entity.get_world_poses(indices=list(range(env.num_envs)))
#                 if pos_w.shape[0] == env.num_envs:
#                     return pos_w.clone()
#             except Exception:
#                 pass

#         pos_w, _ = entity.get_world_poses()
#         if pos_w.shape[0] == env.num_envs:
#             return pos_w.clone()
#         if pos_w.shape[0] == 1:
#             local_pos = pos_w[0].clone() - origins[0]
#             return origins + local_pos

#     raise RuntimeError(
#         f"Cannot resolve scene entity '{entity_name}' to {env.num_envs} world positions."
#     )


# def get_scene_entity_position_w(env, entity_name: str, env_id: int) -> torch.Tensor:
#     return get_scene_entity_positions_w(env, entity_name)[env_id]


# def _get_scene_entity_positions_w(env, entity_name: str) -> torch.Tensor:
#     return get_scene_entity_positions_w(env, entity_name)


# def _get_scene_entity_position_w(env, entity_name: str, env_id: int) -> torch.Tensor:
#     return get_scene_entity_position_w(env, entity_name, env_id)


# class SPARTANStateMachine:
#     Z_TABLE = 0.717
#     SAFE_Z_OFFSET = 0.03
#     GRASP_Z_OFFSET = 0.005
#     GRASP_ATTACH_DISTANCE = 0.06
#     GRASP_CLOSE_MIN_STEPS = 10
#     GRASP_TIMEOUT_STEPS = 220
#     RELEASE_PEG_CAPTURE_DISTANCE = 0.03
#     BOARD_RELEASE_Z_OFFSET = 0.0
#     LINEAR_SPEED = 0.02
#     REACH_ENTRY_X = 0.375
#     REACH_ENTRY_Y = -0.05
#     REACH_ENTRY_LATERAL_OFFSET = 0.03
#     REACH_ENTRY_Z = 0.76
#     TOLERANCE_XY = 0.006
#     TOLERANCE_Z = 0.006
#     RING_THICKNESS = 0.001

#     def __init__(self, env):
#         self.env = env
#         self.device = env.device
#         self.num_envs = env.num_envs

#         self.current_triplet_r = [None] * self.num_envs
#         self.current_triplet_l = [None] * self.num_envs
#         self.sub_state_r = [ARM_IDLE_STATE] * self.num_envs
#         self.sub_state_l = [ARM_IDLE_STATE] * self.num_envs
#         self.step_counter_r = [0] * self.num_envs
#         self.step_counter_l = [0] * self.num_envs
#         self.active_verb_r = ["idle"] * self.num_envs
#         self.active_verb_l = ["idle"] * self.num_envs

#         self.target_pos_r = torch.zeros((self.num_envs, 3), device=self.device)
#         self.target_pos_l = torch.zeros((self.num_envs, 3), device=self.device)

#         self.current_gripper_state_r = torch.ones(self.num_envs, device=self.device)
#         self.current_gripper_state_l = torch.ones(self.num_envs, device=self.device)

#         self.attached_target_r = [None] * self.num_envs
#         self.attached_target_l = [None] * self.num_envs
#         self.last_target_r = [None] * self.num_envs
#         self.last_target_l = [None] * self.num_envs
#         self.release_target_r = [None] * self.num_envs
#         self.release_target_l = [None] * self.num_envs
#         self.release_committed_r = [False] * self.num_envs
#         self.release_committed_l = [False] * self.num_envs

#         self.peg_inventory = {
#             peg: torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
#             for peg in PEG_NAMES
#         }
#         self.frozen_rings_mask = {
#             ring: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
#             for ring in RING_NAMES
#         }
#         self.frozen_rings_pose = {
#             ring: torch.zeros((self.num_envs, 7), device=self.device)
#             for ring in RING_NAMES
#         }
#         self.ring_support_peg = {
#             ring: [None] * self.num_envs
#             for ring in RING_NAMES
#         }
#         self.ring_stack_level = {
#             ring: torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
#             for ring in RING_NAMES
#         }

#         robot_r = env.scene["robot_right"]
#         robot_l = env.scene["robot_left"]
#         self._tip_idx_r = robot_r.find_bodies("psm_tool_tip_link")[0][0]
#         self._tip_idx_l = robot_l.find_bodies("psm_tool_tip_link")[0][0]
#         self.target_quat_r = self._get_tip_quat_in_root(robot_r, self._tip_idx_r).clone()
#         self.target_quat_l = self._get_tip_quat_in_root(robot_l, self._tip_idx_l).clone()

#         self._peg_name_to_index = {name: idx for idx, name in enumerate(PEG_NAMES)}
#         self._cached_peg_positions_w = self._build_cached_peg_positions_w()

#     def _build_cached_peg_positions_w(self) -> torch.Tensor:
#         cached_positions = []
#         for peg_name in PEG_NAMES:
#             cached_positions.append(get_scene_entity_positions_w(self.env, peg_name))
#         return torch.stack(cached_positions, dim=1)

#     def _get_cached_peg_position_w(self, peg_name: str, env_id: int) -> torch.Tensor | None:
#         peg_index = self._peg_name_to_index.get(peg_name)
#         if peg_index is None:
#             return None
#         return self._cached_peg_positions_w[env_id, peg_index]

#     def _set_phase(self, sub_state, step_counter, env_id: int, phase: str):
#         sub_state[env_id] = phase
#         step_counter[env_id] = 0

#     def _finish_command(self, active_verb, sub_state, step_counter, env_id: int, current_triplet=None):
#         active_verb[env_id] = "idle"
#         sub_state[env_id] = ARM_IDLE_STATE
#         step_counter[env_id] = 0
#         if current_triplet is not None:
#             current_triplet[env_id] = None

#     def _get_tip_quat_in_root(self, robot, tip_idx):
#         _, tip_quat_b = math_utils.subtract_frame_transforms(
#             robot.data.root_pos_w,
#             robot.data.root_quat_w,
#             robot.data.body_pos_w[:, tip_idx],
#             robot.data.body_quat_w[:, tip_idx],
#         )
#         return tip_quat_b

#     def _tip_reached(self, tip_pos, dest) -> bool:
#         return (
#             torch.norm(dest[0:2] - tip_pos[0:2]) < self.TOLERANCE_XY
#             and torch.abs(dest[2] - tip_pos[2]) < self.TOLERANCE_Z
#         )

#     def _get_fixed_speed_cmd(self, current_pos, dest):
#         delta = dest - current_pos
#         distance = torch.norm(delta)

#         if distance < 1e-8:
#             return torch.zeros(3, device=self.device)
#         if distance <= self.LINEAR_SPEED:
#             return delta

#         return (delta / distance) * self.LINEAR_SPEED

#     def _get_reach_entry_position(self, subject: str, env_id: int):
#         origin = self.env.scene.env_origins[env_id]
#         side_sign = -1.0 if subject == "right_arm" else 1.0
#         local_entry = torch.tensor(
#             [
#                 self.REACH_ENTRY_X,
#                 self.REACH_ENTRY_Y + side_sign * self.REACH_ENTRY_LATERAL_OFFSET,
#                 self.REACH_ENTRY_Z,
#             ],
#             device=self.device,
#         )
#         return origin + local_entry

#     def _resolve_release_target(self, target_name: str | None, last_target: str | None):
#         if isinstance(target_name, str) and target_name.startswith("peg_"):
#             return target_name
#         if isinstance(last_target, str) and last_target.startswith("peg_"):
#             return last_target
#         return None

#     def _is_peg_target(self, target_name: str | None) -> bool:
#         return isinstance(target_name, str) and target_name.startswith("peg_")

#     def _get_reach_settle_offset(self, target_name: str) -> float:
#         if self._is_peg_target(target_name):
#             return self.SAFE_Z_OFFSET
#         return self.GRASP_Z_OFFSET

#     def _find_release_peg(self, release_pos_w, env_id: int, preferred_peg: str | None = None):
#         candidate_pegs = []
#         if preferred_peg is not None:
#             candidate_pegs.append(preferred_peg)

#         for peg_name in self.peg_inventory:
#             if peg_name != preferred_peg:
#                 candidate_pegs.append(peg_name)

#         best_peg = None
#         best_xy_distance = None
#         for peg_name in candidate_pegs:
#             peg_pos_w = self._get_cached_peg_position_w(peg_name, env_id)
#             if peg_pos_w is None:
#                 peg_pos_w = _get_scene_entity_position_w(self.env, peg_name, env_id)
#             xy_distance = torch.norm(release_pos_w[0:2] - peg_pos_w[0:2]).item()
#             if xy_distance > self.RELEASE_PEG_CAPTURE_DISTANCE:
#                 continue

#             if best_xy_distance is None or xy_distance < best_xy_distance:
#                 best_peg = peg_name
#                 best_xy_distance = xy_distance

#         return best_peg

#     def _clear_ring_support(self, ring_name: str, env_id: int):
#         self.ring_support_peg[ring_name][env_id] = None
#         self.ring_stack_level[ring_name][env_id] = -1

#     def _count_rings_on_peg(self, env_id: int, peg_name: str) -> int:
#         count = 0
#         for ring_name in self.frozen_rings_mask:
#             if self.frozen_rings_mask[ring_name][env_id] and self.ring_support_peg[ring_name][env_id] == peg_name:
#                 count += 1
#         return count

#     def ring_count_on_peg(self, env_id: int, peg_name: str) -> int:
#         return self._count_rings_on_peg(env_id, peg_name)

#     def _sync_peg_inventory(self, env_id: int, peg_name: str):
#         self.peg_inventory[peg_name][env_id] = self._count_rings_on_peg(env_id, peg_name)

#     def _is_top_ring_on_peg(self, ring_name: str, env_id: int) -> bool:
#         peg_name = self.ring_support_peg[ring_name][env_id]
#         if peg_name is None:
#             return True

#         ring_level = int(self.ring_stack_level[ring_name][env_id].item())
#         top_level = None
#         for candidate_name in self.ring_stack_level:
#             if self.frozen_rings_mask[candidate_name][env_id] and self.ring_support_peg[candidate_name][env_id] == peg_name:
#                 candidate_level = int(self.ring_stack_level[candidate_name][env_id].item())
#                 if top_level is None or candidate_level > top_level:
#                     top_level = candidate_level

#         return top_level is None or ring_level == top_level

#     def _detach_ring_from_peg(self, ring_name: str, env_id: int):
#         peg_name = self.ring_support_peg[ring_name][env_id]
#         if peg_name is None:
#             self._clear_ring_support(ring_name, env_id)
#             return

#         removed_level = int(self.ring_stack_level[ring_name][env_id].item())
#         self._clear_ring_support(ring_name, env_id)

#         for candidate_name in self.ring_stack_level:
#             if self.ring_support_peg[candidate_name][env_id] != peg_name:
#                 continue

#             candidate_level = int(self.ring_stack_level[candidate_name][env_id].item())
#             if candidate_level > removed_level:
#                 self.ring_stack_level[candidate_name][env_id] = candidate_level - 1

#         self._sync_peg_inventory(env_id, peg_name)

#     def reset_env(self, env_id: int):
#         self.current_triplet_r[env_id] = None
#         self.current_triplet_l[env_id] = None

#         self.sub_state_r[env_id] = ARM_IDLE_STATE
#         self.sub_state_l[env_id] = ARM_IDLE_STATE

#         self.step_counter_r[env_id] = 0
#         self.step_counter_l[env_id] = 0
#         self.active_verb_r[env_id] = "idle"
#         self.active_verb_l[env_id] = "idle"

#         self.attached_target_r[env_id] = None
#         self.attached_target_l[env_id] = None

#         self.last_target_r[env_id] = None
#         self.last_target_l[env_id] = None
#         self.release_target_r[env_id] = None
#         self.release_target_l[env_id] = None
#         self.release_committed_r[env_id] = False
#         self.release_committed_l[env_id] = False

#         self.target_pos_r[env_id] = torch.zeros(3, device=self.device)
#         self.target_pos_l[env_id] = torch.zeros(3, device=self.device)
#         self.current_gripper_state_r[env_id] = 1.0
#         self.current_gripper_state_l[env_id] = 1.0

#         for peg in self.peg_inventory:
#             self.peg_inventory[peg][env_id] = 0
#         for ring in self.frozen_rings_mask:
#             self.frozen_rings_mask[ring][env_id] = False
#             self.frozen_rings_pose[ring][env_id] = 0.0
#             self._clear_ring_support(ring, env_id)

#         print(f"[ENV {env_id}] Reset completo. Stato macchina a stati reimpostato.")

#     def arm_idle(self, env_id: int, arm: str) -> bool:
#         if arm == "right_arm":
#             return self.sub_state_r[env_id] == ARM_IDLE_STATE and self.current_triplet_r[env_id] is None
#         return self.sub_state_l[env_id] == ARM_IDLE_STATE and self.current_triplet_l[env_id] is None

#     def all_idle(self, env_id: int) -> bool:
#         return self.arm_idle(env_id, "right_arm") and self.arm_idle(env_id, "left_arm")

#     def set_new_triplet(self, verb: str, subject: str, target: str, env_id: int):
#         new_cmd = {"verb": verb, "subject": subject, "target": target}
#         if subject == "right_arm":
#             if self.current_triplet_r[env_id] == new_cmd and self.active_verb_r[env_id] == verb and self.sub_state_r[env_id] != ARM_IDLE_STATE:
#                 return

#             self.current_triplet_r[env_id] = new_cmd
#             self.active_verb_r[env_id] = verb
#             self.target_pos_r[env_id] = 0.0
#             self.release_target_r[env_id] = None
#             self.release_committed_r[env_id] = False

#             if verb == "idle":
#                 self._finish_command(self.active_verb_r, self.sub_state_r, self.step_counter_r, env_id, self.current_triplet_r)
#                 return

#             self._set_phase(self.sub_state_r, self.step_counter_r, env_id, INITIAL_PHASE_BY_VERB[verb])

#             if verb in ("reach", "grasp"):
#                 self.target_pos_r[env_id] = self.get_target_coordinates(target, subject, env_id)
#             elif verb == "release":
#                 self.release_target_r[env_id] = self._resolve_release_target(target, self.last_target_r[env_id])
#         else:
#             if self.current_triplet_l[env_id] == new_cmd and self.active_verb_l[env_id] == verb and self.sub_state_l[env_id] != ARM_IDLE_STATE:
#                 return

#             self.current_triplet_l[env_id] = new_cmd
#             self.active_verb_l[env_id] = verb
#             self.target_pos_l[env_id] = 0.0
#             self.release_target_l[env_id] = None
#             self.release_committed_l[env_id] = False

#             if verb == "idle":
#                 self._finish_command(self.active_verb_l, self.sub_state_l, self.step_counter_l, env_id, self.current_triplet_l)
#                 return

#             self._set_phase(self.sub_state_l, self.step_counter_l, env_id, INITIAL_PHASE_BY_VERB[verb])

#             if verb in ("reach", "grasp"):
#                 self.target_pos_l[env_id] = self.get_target_coordinates(target, subject, env_id)
#             elif verb == "release":
#                 self.release_target_l[env_id] = self._resolve_release_target(target, self.last_target_l[env_id])

#     def get_target_coordinates(self, target_name: str, subject: str, env_id: int):
#         origin = self.env.scene.env_origins[env_id]

#         if target_name == "None" or target_name is None:
#             return self._get_reach_entry_position(subject, env_id)

#         sign_arm = 1.0 if subject == "right_arm" else -1.0
#         offset = torch.tensor([-0.005 * sign_arm, 0.0, 0.0], device=self.device)

#         cached_peg_pos = self._get_cached_peg_position_w(target_name, env_id)
#         if cached_peg_pos is not None:
#             return cached_peg_pos + offset

#         if target_name in self.env.scene.keys():
#             try:
#                 return _get_scene_entity_position_w(self.env, target_name, env_id) + offset
#             except RuntimeError as exc:
#                 print(f"[ENV {env_id}] Warning: {exc}")

#         print(
#             f"[ENV {env_id}] Warning: I cannot solve the position of target "
#             f"'{target_name}' for subject '{subject}'. Using fallback coordinates."
#         )
#         local_fallback = torch.tensor([0.5, 0.0, 0.5], device=self.device)
#         return origin + local_fallback

#     def _process_arm_logic(
#         self,
#         i,
#         active_verb,
#         sub_state,
#         step_counter,
#         current_triplet,
#         last_target,
#         attached_target,
#         target_pos,
#         curr_grip_state,
#         release_target,
#         release_committed,
#         tip,
#         act,
#         grip,
#         subject,
#     ):
#         if sub_state[i] == ARM_IDLE_STATE or current_triplet[i] is None:
#             act[i, 0:3] = 0.0
#             return

#         verb = active_verb[i]
#         phase = sub_state[i]
#         valid_phases = VALID_PHASES_BY_VERB.get(verb)

#         if valid_phases is None:
#             self._finish_command(active_verb, sub_state, step_counter, i, current_triplet)
#             act[i, 0:3] = 0.0
#             return

#         if phase not in valid_phases:
#             self._set_phase(sub_state, step_counter, i, INITIAL_PHASE_BY_VERB[verb])
#             phase = sub_state[i]

#         if verb == "reach":
#             target_name = current_triplet[i]["target"]
#             settle_offset = self._get_reach_settle_offset(target_name)
#             if phase == "MOVE_TO_REACH_ENTRY":
#                 dest = self._get_reach_entry_position(subject, i)
#                 act[i, 0:3] = self._get_fixed_speed_cmd(tip[i], dest)
#                 if self._tip_reached(tip[i], dest) or step_counter[i] > 150:
#                     self._set_phase(sub_state, step_counter, i, "APPROACH_ABOVE")
#             elif phase == "APPROACH_ABOVE":
#                 target_pos[i] = self.get_target_coordinates(target_name, subject, i)
#                 dest = target_pos[i] + torch.tensor([0.0, 0.0, self.SAFE_Z_OFFSET], device=self.device)
#                 act[i, 0:3] = self._get_fixed_speed_cmd(tip[i], dest)
#                 if self._tip_reached(tip[i], dest) or step_counter[i] > 150:
#                     if self._is_peg_target(target_name):
#                         self._set_phase(sub_state, step_counter, i, "SETTLE")
#                     else:
#                         self._set_phase(sub_state, step_counter, i, "DESCEND")
#             elif phase == "DESCEND":
#                 target_pos[i] = self.get_target_coordinates(target_name, subject, i)
#                 dest = target_pos[i] + torch.tensor([0.0, 0.0, self.GRASP_Z_OFFSET], device=self.device)
#                 act[i, 0:3] = self._get_fixed_speed_cmd(tip[i], dest)
#                 if self._tip_reached(tip[i], dest):
#                     self._set_phase(sub_state, step_counter, i, "SETTLE")
#             elif phase == "SETTLE":
#                 target_pos[i] = self.get_target_coordinates(target_name, subject, i)
#                 dest = target_pos[i] + torch.tensor([0.0, 0.0, settle_offset], device=self.device)
#                 act[i, 0:3] = self._get_fixed_speed_cmd(tip[i], dest)
#                 if step_counter[i] > 15:
#                     last_target[i] = target_name
#                     self._finish_command(active_verb, sub_state, step_counter, i, current_triplet)

#         elif verb == "grasp":
#             if phase == "CLOSE_GRIPPER":
#                 curr_grip_state[i] = -1.0
#                 grip[i] = -1.0
#                 obj = current_triplet[i]["target"]
#                 grasp_target = self.get_target_coordinates(obj, subject, i)
#                 target_pos[i] = grasp_target
#                 dest = grasp_target + torch.tensor([0.0, 0.0, self.GRASP_Z_OFFSET], device=self.device)
#                 act[i, 0:3] = self._get_fixed_speed_cmd(tip[i], dest)

#                 if (
#                     obj in self.frozen_rings_mask
#                     and step_counter[i] >= self.GRASP_CLOSE_MIN_STEPS
#                     and torch.norm(grasp_target - tip[i]) < self.GRASP_ATTACH_DISTANCE
#                     and self._is_top_ring_on_peg(obj, i)
#                 ):
#                     self._detach_ring_from_peg(obj, i)
#                     self.frozen_rings_mask[obj][i] = False
#                     attached_target[i] = obj
#                     last_target[i] = obj
#                     self._set_phase(sub_state, step_counter, i, "LIFT_UP")
#                 elif step_counter[i] > self.GRASP_TIMEOUT_STEPS:
#                     curr_grip_state[i] = 1.0
#                     grip[i] = 1.0
#                     self._finish_command(active_verb, sub_state, step_counter, i, current_triplet)
#             elif phase == "LIFT_UP":
#                 dest = target_pos[i] + torch.tensor([0.0, 0.0, self.SAFE_Z_OFFSET], device=self.device)
#                 act[i, 0:3] = self._get_fixed_speed_cmd(tip[i], dest)
#                 if self._tip_reached(tip[i], dest) or step_counter[i] > 250:
#                     self._finish_command(active_verb, sub_state, step_counter, i, current_triplet)

#         elif verb == "release":
#             if phase == "OPEN_GRIPPER":
#                 curr_grip_state[i] = 1.0
#                 grip[i] = 1.0
#                 act[i, 0:3] = 0.0

#                 if step_counter[i] == 10 and not release_committed[i]:
#                     obj = attached_target[i]
#                     if obj and obj != "None":
#                         curr_pos = self.env.scene[obj].data.root_pos_w[i].clone()
#                         peg = self._find_release_peg(curr_pos, i, release_target[i])
#                         release_pos = curr_pos.clone()

#                         if peg:
#                             peg_pos = _get_scene_entity_position_w(self.env, peg, i).clone()
#                             num_rings = self._count_rings_on_peg(i, peg)
#                             release_pos[0:2] = peg_pos[0:2]
#                             release_pos[2] = self.Z_TABLE + self.env.scene.env_origins[i, 2] + (num_rings * self.RING_THICKNESS)
#                             self.ring_support_peg[obj][i] = peg
#                             self.ring_stack_level[obj][i] = num_rings
#                             self.peg_inventory[peg][i] = num_rings + 1
#                         else:
#                             release_pos[2] = self.Z_TABLE + self.env.scene.env_origins[i, 2] + self.BOARD_RELEASE_Z_OFFSET
#                             self._clear_ring_support(obj, i)

#                         self.frozen_rings_mask[obj][i] = True
#                         self.frozen_rings_pose[obj][i, 0:3] = release_pos
#                         self.frozen_rings_pose[obj][i, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
#                     attached_target[i] = None
#                     release_committed[i] = True

#                 if step_counter[i] > 20:
#                     self._finish_command(active_verb, sub_state, step_counter, i, current_triplet)

#         step_counter[i] += 1

#     def get_action(self):
#         act_r = torch.zeros((self.num_envs, 7), device=self.device)
#         act_l = torch.zeros((self.num_envs, 7), device=self.device)

#         grip_r = self.current_gripper_state_r.unsqueeze(1).repeat(1, 2)
#         grip_l = self.current_gripper_state_l.unsqueeze(1).repeat(1, 2)
#         robot_r = self.env.scene["robot_right"]
#         robot_l = self.env.scene["robot_left"]
#         tip_r = robot_r.data.body_pos_w[:, self._tip_idx_r]
#         tip_l = robot_l.data.body_pos_w[:, self._tip_idx_l]

#         for i in range(self.num_envs):
#             self._process_arm_logic(
#                 i,
#                 self.active_verb_r,
#                 self.sub_state_r,
#                 self.step_counter_r,
#                 self.current_triplet_r,
#                 self.last_target_r,
#                 self.attached_target_r,
#                 self.target_pos_r,
#                 self.current_gripper_state_r,
#                 self.release_target_r,
#                 self.release_committed_r,
#                 tip_r,
#                 act_r,
#                 grip_r,
#                 "right_arm",
#             )
#             self._process_arm_logic(
#                 i,
#                 self.active_verb_l,
#                 self.sub_state_l,
#                 self.step_counter_l,
#                 self.current_triplet_l,
#                 self.last_target_l,
#                 self.attached_target_l,
#                 self.target_pos_l,
#                 self.current_gripper_state_l,
#                 self.release_target_l,
#                 self.release_committed_l,
#                 tip_l,
#                 act_l,
#                 grip_l,
#                 "left_arm",
#             )

#         desired_pos_r_w = tip_r + act_r[:, 0:3]
#         desired_pos_l_w = tip_l + act_l[:, 0:3]

#         desired_pos_r_b, _ = math_utils.subtract_frame_transforms(
#             robot_r.data.root_pos_w,
#             robot_r.data.root_quat_w,
#             desired_pos_r_w,
#             None,
#         )
#         desired_pos_l_b, _ = math_utils.subtract_frame_transforms(
#             robot_l.data.root_pos_w,
#             robot_l.data.root_quat_w,
#             desired_pos_l_w,
#             None,
#         )

#         act_r[:, 0:3] = desired_pos_r_b
#         act_r[:, 3:7] = self.target_quat_r
#         act_l[:, 0:3] = desired_pos_l_b
#         act_l[:, 3:7] = self.target_quat_l

#         return torch.cat([act_r, grip_r, act_l, grip_l], dim=-1)

#     def green_peg_ring_count(self, env_id: int) -> int:
#         return self._count_rings_on_peg(env_id, "peg_green")

#     def successful(self, env_id: int, target_peg: str) -> bool:
#         return all(
#             self.frozen_rings_mask[ring_name][env_id]
#             and self.ring_support_peg[ring_name][env_id] == target_peg
#             for ring_name in RING_NAMES
#         )


# def sync_attached_and_frozen_rings(env, sm: SPARTANStateMachine, active_env_mask: torch.Tensor):
#     robot_r = env.scene["robot_right"]
#     robot_l = env.scene["robot_left"]
#     tip_r = robot_r.data.body_pos_w[:, sm._tip_idx_r]
#     tip_l = robot_l.data.body_pos_w[:, sm._tip_idx_l]
#     snap_offset_r = torch.tensor([-0.005, 0.0, 0.0], device=env.device)
#     snap_offset_l = torch.tensor([+0.005, 0.0, 0.0], device=env.device)
#     snap_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

#     for ring_name in RING_NAMES:
#         ring = env.scene[ring_name]
#         new_s = ring.data.root_state_w.clone()
#         mask_r = torch.tensor(
#             [sm.attached_target_r[i] == ring_name for i in range(env.num_envs)],
#             dtype=torch.bool,
#             device=env.device,
#         ) & active_env_mask
#         mask_l = torch.tensor(
#             [sm.attached_target_l[i] == ring_name for i in range(env.num_envs)],
#             dtype=torch.bool,
#             device=env.device,
#         ) & active_env_mask
#         mask_frozen = sm.frozen_rings_mask[ring_name] & active_env_mask
#         write_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

#         if mask_r.any():
#             new_s[mask_r, 0:3] = tip_r[mask_r] + snap_offset_r
#             new_s[mask_r, 3:7] = snap_quat
#             new_s[mask_r, 7:13] = 0.0
#             write_mask |= mask_r
#         if mask_l.any():
#             new_s[mask_l, 0:3] = tip_l[mask_l] + snap_offset_l
#             new_s[mask_l, 3:7] = snap_quat
#             new_s[mask_l, 7:13] = 0.0
#             write_mask |= mask_l
#         if mask_frozen.any():
#             new_s[mask_frozen, 0:7] = sm.frozen_rings_pose[ring_name][mask_frozen]
#             new_s[mask_frozen, 7:13] = 0.0
#             write_mask |= mask_frozen
#         if write_mask.any():
#             env_ids_to_write = write_mask.nonzero(as_tuple=False).flatten()
#             ring.write_root_state_to_sim(new_s[env_ids_to_write], env_ids=env_ids_to_write)


"""Shared dVRK parallel environment control primitives.

This module intentionally stays free of AppLauncher side effects so that
scripted utilities can reuse the same state machine logic without importing
training entrypoints that bootstrap Isaac Sim at import time.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils


RING_NAMES = ["ring_red", "ring_yellow", "ring_green", "ring_blue"]
PEG_NAMES = ["peg_red", "peg_yellow", "peg_green", "peg_gray", "peg_gray1", "peg_blue"]
ARM_IDLE_STATE = "IDLE"
INITIAL_PHASE_BY_VERB = {
    "reach": "MOVE_TO_REACH_ENTRY",
    "grasp": "CLOSE_GRIPPER",
    "release": "OPEN_GRIPPER",
}
VALID_PHASES_BY_VERB = {
    "reach": {"MOVE_TO_REACH_ENTRY", "APPROACH_ABOVE", "DESCEND", "SETTLE"},
    "grasp": {"CLOSE_GRIPPER", "LIFT_UP"},
    "release": {"OPEN_GRIPPER"},
}


def get_scene_entity_positions_w(env, entity_name: str) -> torch.Tensor:
    """Resolve a scene entity to one world position per env."""
    entity = env.scene[entity_name]
    origins = env.scene.env_origins

    if hasattr(entity, "data") and hasattr(entity.data, "root_pos_w"):
        pos_w = entity.data.root_pos_w
        if pos_w.shape[0] == env.num_envs:
            return pos_w.clone()
        if pos_w.shape[0] == 1:
            local_pos = pos_w[0].clone() - origins[0]
            return origins + local_pos

    if hasattr(entity, "get_world_poses"):
        view_count = getattr(entity, "count", None)
        if view_count == env.num_envs:
            try:
                pos_w, _ = entity.get_world_poses(indices=list(range(env.num_envs)))
                if pos_w.shape[0] == env.num_envs:
                    return pos_w.clone()
            except Exception:
                pass

        pos_w, _ = entity.get_world_poses()
        if pos_w.shape[0] == env.num_envs:
            return pos_w.clone()
        if pos_w.shape[0] == 1:
            local_pos = pos_w[0].clone() - origins[0]
            return origins + local_pos

    raise RuntimeError(
        f"Cannot resolve scene entity '{entity_name}' to {env.num_envs} world positions."
    )


def get_scene_entity_position_w(env, entity_name: str, env_id: int) -> torch.Tensor:
    return get_scene_entity_positions_w(env, entity_name)[env_id]


def _get_scene_entity_positions_w(env, entity_name: str) -> torch.Tensor:
    return get_scene_entity_positions_w(env, entity_name)


def _get_scene_entity_position_w(env, entity_name: str, env_id: int) -> torch.Tensor:
    return get_scene_entity_position_w(env, entity_name, env_id)


class SPARTANStateMachine:
    Z_TABLE = 0.717
    SAFE_Z_OFFSET = 0.015
    GRASP_Z_OFFSET = 0.005
    GRASP_ATTACH_DISTANCE = 0.06
    GRASP_CLOSE_MIN_STEPS = 10
    GRASP_TIMEOUT_STEPS = 220
    RELEASE_PEG_CAPTURE_DISTANCE = 0.03
    BOARD_RELEASE_Z_OFFSET = 0.0
    LINEAR_SPEED = 0.02
    REACH_ENTRY_X = 0.375
    REACH_ENTRY_Y = -0.05
    REACH_ENTRY_LATERAL_OFFSET = 0.03
    REACH_ENTRY_Z = 0.76
    TOLERANCE_XY = 0.006
    TOLERANCE_Z = 0.006
    RING_THICKNESS = 0.001

    def __init__(self, env, debug_mode=False):
        self.debug_mode = debug_mode
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs

        self.current_triplet_r = [None] * self.num_envs
        self.current_triplet_l = [None] * self.num_envs
        self.sub_state_r = [ARM_IDLE_STATE] * self.num_envs
        self.sub_state_l = [ARM_IDLE_STATE] * self.num_envs
        self.step_counter_r = [0] * self.num_envs
        self.step_counter_l = [0] * self.num_envs
        self.active_verb_r = ["idle"] * self.num_envs
        self.active_verb_l = ["idle"] * self.num_envs

        self.target_pos_r = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_pos_l = torch.zeros((self.num_envs, 3), device=self.device)

        self.current_gripper_state_r = torch.ones(self.num_envs, device=self.device)
        self.current_gripper_state_l = torch.ones(self.num_envs, device=self.device)

        self.attached_target_r = [None] * self.num_envs
        self.attached_target_l = [None] * self.num_envs
        self.last_target_r = [None] * self.num_envs
        self.last_target_l = [None] * self.num_envs
        self.release_target_r = [None] * self.num_envs
        self.release_target_l = [None] * self.num_envs
        self.release_committed_r = [False] * self.num_envs
        self.release_committed_l = [False] * self.num_envs

        self.peg_inventory = {
            peg: torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            for peg in PEG_NAMES
        }
        self.frozen_rings_mask = {
            ring: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for ring in RING_NAMES
        }
        self.frozen_rings_pose = {
            ring: torch.zeros((self.num_envs, 7), device=self.device)
            for ring in RING_NAMES
        }
        self.ring_support_peg = {
            ring: [None] * self.num_envs
            for ring in RING_NAMES
        }
        self.ring_stack_level = {
            ring: torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
            for ring in RING_NAMES
        }

        robot_r = env.scene["robot_right"]
        robot_l = env.scene["robot_left"]
        self._tip_idx_r = robot_r.find_bodies("psm_tool_tip_link")[0][0]
        self._tip_idx_l = robot_l.find_bodies("psm_tool_tip_link")[0][0]
        self.target_quat_r = self._get_tip_quat_in_root(robot_r, self._tip_idx_r).clone()
        self.target_quat_l = self._get_tip_quat_in_root(robot_l, self._tip_idx_l).clone()

        self._peg_name_to_index = {name: idx for idx, name in enumerate(PEG_NAMES)}
        self._cached_peg_positions_w = self._build_cached_peg_positions_w()

    def _build_cached_peg_positions_w(self) -> torch.Tensor:
        cached_positions = []
        for peg_name in PEG_NAMES:
            cached_positions.append(get_scene_entity_positions_w(self.env, peg_name))
        return torch.stack(cached_positions, dim=1)

    def _get_cached_peg_position_w(self, peg_name: str, env_id: int) -> torch.Tensor | None:
        peg_index = self._peg_name_to_index.get(peg_name)
        if peg_index is None:
            return None
        return self._cached_peg_positions_w[env_id, peg_index]

    def _set_phase(self, sub_state, step_counter, env_id: int, phase: str):
        if getattr(self, "debug_mode", False):
            print(f"[DEBUG SM] Env {env_id} - Transition to phase: {phase} (took {step_counter[env_id]} steps)")
        sub_state[env_id] = phase
        step_counter[env_id] = 0

    def _finish_command(self, active_verb, sub_state, step_counter, env_id: int, current_triplet=None):
        if getattr(self, "debug_mode", False):
            print(f"[DEBUG SM] Env {env_id} - Finished command {active_verb[env_id]}. Going to IDLE. (took {step_counter[env_id]} steps)")
        active_verb[env_id] = "idle"
        sub_state[env_id] = ARM_IDLE_STATE
        step_counter[env_id] = 0
        if current_triplet is not None:
            current_triplet[env_id] = None

    def _get_tip_quat_in_root(self, robot, tip_idx):
        _, tip_quat_b = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            robot.data.body_pos_w[:, tip_idx],
            robot.data.body_quat_w[:, tip_idx],
        )
        return tip_quat_b

    def _tip_reached(self, tip_pos, dest) -> bool:
        return (
            torch.norm(dest[0:2] - tip_pos[0:2]) < self.TOLERANCE_XY
            and torch.abs(dest[2] - tip_pos[2]) < self.TOLERANCE_Z
        )

    def _get_fixed_speed_cmd(self, current_pos, dest):
        delta = dest - current_pos
        distance = torch.norm(delta)

        if distance < 1e-8:
            return torch.zeros(3, device=self.device)
        if distance <= self.LINEAR_SPEED:
            return delta

        return (delta / distance) * self.LINEAR_SPEED

    def _get_reach_entry_position(self, subject: str, env_id: int):
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

    def _resolve_release_target(self, target_name: str | None, last_target: str | None):
        if isinstance(target_name, str) and target_name.startswith("peg_"):
            return target_name
        if isinstance(last_target, str) and last_target.startswith("peg_"):
            return last_target
        return None

    def _is_peg_target(self, target_name: str | None) -> bool:
        return isinstance(target_name, str) and target_name.startswith("peg_")

    def _get_reach_settle_offset(self, target_name: str) -> float:
        if self._is_peg_target(target_name):
            return self.SAFE_Z_OFFSET
        return self.GRASP_Z_OFFSET

    def _find_release_peg(self, release_pos_w, env_id: int, preferred_peg: str | None = None):
        if preferred_peg is not None:
            peg_pos_w = self._get_cached_peg_position_w(preferred_peg, env_id)
            if peg_pos_w is None:
                peg_pos_w = _get_scene_entity_position_w(self.env, preferred_peg, env_id)
            xy_distance = torch.norm(release_pos_w[0:2] - peg_pos_w[0:2]).item()
            
            if getattr(self, "debug_mode", False):
                print(f"[DEBUG SM] Env {env_id} - Release target: {preferred_peg} - Distanza XY reale: {xy_distance:.4f}m (Tolleranza: {self.RELEASE_PEG_CAPTURE_DISTANCE}m)")
            
            if xy_distance <= self.RELEASE_PEG_CAPTURE_DISTANCE:
                return preferred_peg

        candidate_pegs = []
        for peg_name in self.peg_inventory:
            if peg_name != preferred_peg:
                candidate_pegs.append(peg_name)

        best_peg = None
        best_xy_distance = None
        for peg_name in candidate_pegs:
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

    def _clear_ring_support(self, ring_name: str, env_id: int):
        self.ring_support_peg[ring_name][env_id] = None
        self.ring_stack_level[ring_name][env_id] = -1

    def _count_rings_on_peg(self, env_id: int, peg_name: str) -> int:
        count = 0
        for ring_name in self.frozen_rings_mask:
            if self.frozen_rings_mask[ring_name][env_id] and self.ring_support_peg[ring_name][env_id] == peg_name:
                count += 1
        return count

    def ring_count_on_peg(self, env_id: int, peg_name: str) -> int:
        return self._count_rings_on_peg(env_id, peg_name)

    def _sync_peg_inventory(self, env_id: int, peg_name: str):
        self.peg_inventory[peg_name][env_id] = self._count_rings_on_peg(env_id, peg_name)

    def _is_top_ring_on_peg(self, ring_name: str, env_id: int) -> bool:
        peg_name = self.ring_support_peg[ring_name][env_id]
        if peg_name is None:
            return True

        ring_level = int(self.ring_stack_level[ring_name][env_id].item())
        top_level = None
        for candidate_name in self.ring_stack_level:
            if self.frozen_rings_mask[candidate_name][env_id] and self.ring_support_peg[candidate_name][env_id] == peg_name:
                candidate_level = int(self.ring_stack_level[candidate_name][env_id].item())
                if top_level is None or candidate_level > top_level:
                    top_level = candidate_level

        return top_level is None or ring_level == top_level

    def _detach_ring_from_peg(self, ring_name: str, env_id: int):
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

    def reset_env(self, env_id: int):
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

        print(f"[ENV {env_id}] Reset completo. Stato macchina a stati reimpostato.")

    def arm_idle(self, env_id: int, arm: str) -> bool:
        if arm == "right_arm":
            return self.sub_state_r[env_id] == ARM_IDLE_STATE and self.current_triplet_r[env_id] is None
        return self.sub_state_l[env_id] == ARM_IDLE_STATE and self.current_triplet_l[env_id] is None

    def all_idle(self, env_id: int) -> bool:
        return self.arm_idle(env_id, "right_arm") and self.arm_idle(env_id, "left_arm")

    def set_new_triplet(self, verb: str, subject: str, target: str, env_id: int):
        new_cmd = {"verb": verb, "subject": subject, "target": target}
        if subject == "right_arm":
            if self.current_triplet_r[env_id] == new_cmd and self.active_verb_r[env_id] == verb and self.sub_state_r[env_id] != ARM_IDLE_STATE:
                return

            self.current_triplet_r[env_id] = new_cmd
            self.active_verb_r[env_id] = verb
            self.target_pos_r[env_id] = 0.0
            self.release_target_r[env_id] = None
            self.release_committed_r[env_id] = False

            if verb == "idle":
                self._finish_command(self.active_verb_r, self.sub_state_r, self.step_counter_r, env_id, self.current_triplet_r)
                return

            self._set_phase(self.sub_state_r, self.step_counter_r, env_id, INITIAL_PHASE_BY_VERB[verb])

            if verb in ("reach", "grasp"):
                self.target_pos_r[env_id] = self.get_target_coordinates(target, subject, env_id)
            elif verb == "release":
                self.release_target_r[env_id] = self._resolve_release_target(target, self.last_target_r[env_id])
                self.target_pos_r[env_id] = self.get_target_coordinates(target, subject, env_id)
        else:
            if self.current_triplet_l[env_id] == new_cmd and self.active_verb_l[env_id] == verb and self.sub_state_l[env_id] != ARM_IDLE_STATE:
                return

            self.current_triplet_l[env_id] = new_cmd
            self.active_verb_l[env_id] = verb
            self.target_pos_l[env_id] = 0.0
            self.release_target_l[env_id] = None
            self.release_committed_l[env_id] = False

            if verb == "idle":
                self._finish_command(self.active_verb_l, self.sub_state_l, self.step_counter_l, env_id, self.current_triplet_l)
                return

            self._set_phase(self.sub_state_l, self.step_counter_l, env_id, INITIAL_PHASE_BY_VERB[verb])

            if verb in ("reach", "grasp"):
                self.target_pos_l[env_id] = self.get_target_coordinates(target, subject, env_id)
            elif verb == "release":
                self.release_target_l[env_id] = self._resolve_release_target(target, self.last_target_l[env_id])
                self.target_pos_l[env_id] = self.get_target_coordinates(target, subject, env_id)

    def get_target_coordinates(self, target_name: str, subject: str, env_id: int):
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
            f"[ENV {env_id}] Warning: I cannot solve the position of target "
            f"'{target_name}' for subject '{subject}'. Using fallback coordinates."
        )
        local_fallback = torch.tensor([0.5, 0.0, 0.5], device=self.device)
        return origin + local_fallback

    def _process_arm_logic(
        self,
        i,
        active_verb,
        sub_state,
        step_counter,
        current_triplet,
        last_target,
        attached_target,
        target_pos,
        curr_grip_state,
        release_target,
        release_committed,
        tip,
        act,
        grip,
        subject,
    ):
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
                            release_pos[2] = self.Z_TABLE + self.env.scene.env_origins[i, 2] + (num_rings * self.RING_THICKNESS)
                            self.ring_support_peg[obj][i] = peg
                            self.ring_stack_level[obj][i] = num_rings
                            self.peg_inventory[peg][i] = num_rings + 1
                        else:
                            release_pos[2] = self.Z_TABLE + self.env.scene.env_origins[i, 2] + self.BOARD_RELEASE_Z_OFFSET
                            self._clear_ring_support(obj, i)

                        self.frozen_rings_mask[obj][i] = True
                        self.frozen_rings_pose[obj][i, 0:3] = release_pos
                        self.frozen_rings_pose[obj][i, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
                    attached_target[i] = None
                    release_committed[i] = True

                if step_counter[i] > 20:
                    self._finish_command(active_verb, sub_state, step_counter, i, current_triplet)

        step_counter[i] += 1

    def get_action(self):
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
                self.active_verb_r,
                self.sub_state_r,
                self.step_counter_r,
                self.current_triplet_r,
                self.last_target_r,
                self.attached_target_r,
                self.target_pos_r,
                self.current_gripper_state_r,
                self.release_target_r,
                self.release_committed_r,
                tip_r,
                act_r,
                grip_r,
                "right_arm",
            )
            self._process_arm_logic(
                i,
                self.active_verb_l,
                self.sub_state_l,
                self.step_counter_l,
                self.current_triplet_l,
                self.last_target_l,
                self.attached_target_l,
                self.target_pos_l,
                self.current_gripper_state_l,
                self.release_target_l,
                self.release_committed_l,
                tip_l,
                act_l,
                grip_l,
                "left_arm",
            )

        desired_pos_r_w = tip_r + act_r[:, 0:3]
        desired_pos_l_w = tip_l + act_l[:, 0:3]

        desired_pos_r_b, _ = math_utils.subtract_frame_transforms(
            robot_r.data.root_pos_w,
            robot_r.data.root_quat_w,
            desired_pos_r_w,
            None,
        )
        desired_pos_l_b, _ = math_utils.subtract_frame_transforms(
            robot_l.data.root_pos_w,
            robot_l.data.root_quat_w,
            desired_pos_l_w,
            None,
        )

        act_r[:, 0:3] = desired_pos_r_b
        act_r[:, 3:7] = self.target_quat_r
        act_l[:, 0:3] = desired_pos_l_b
        act_l[:, 3:7] = self.target_quat_l

        return torch.cat([act_r, grip_r, act_l, grip_l], dim=-1)

    def green_peg_ring_count(self, env_id: int) -> int:
        return self._count_rings_on_peg(env_id, "peg_green")

    def successful(self, env_id: int, target_peg: str) -> bool:
        return all(
            self.frozen_rings_mask[ring_name][env_id]
            and self.ring_support_peg[ring_name][env_id] == target_peg
            for ring_name in RING_NAMES
        )


def sync_attached_and_frozen_rings(env, sm: SPARTANStateMachine, active_env_mask: torch.Tensor):
    robot_r = env.scene["robot_right"]
    robot_l = env.scene["robot_left"]
    tip_r = robot_r.data.body_pos_w[:, sm._tip_idx_r]
    tip_l = robot_l.data.body_pos_w[:, sm._tip_idx_l]
    snap_offset_r = torch.tensor([-0.005, 0.0, 0.0], device=env.device)
    snap_offset_l = torch.tensor([+0.005, 0.0, 0.0], device=env.device)
    snap_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

    for ring_name in RING_NAMES:
        ring = env.scene[ring_name]
        new_s = ring.data.root_state_w.clone()
        mask_r = torch.tensor(
            [sm.attached_target_r[i] == ring_name for i in range(env.num_envs)],
            dtype=torch.bool,
            device=env.device,
        ) & active_env_mask
        mask_l = torch.tensor(
            [sm.attached_target_l[i] == ring_name for i in range(env.num_envs)],
            dtype=torch.bool,
            device=env.device,
        ) & active_env_mask
        mask_frozen = sm.frozen_rings_mask[ring_name] & active_env_mask
        write_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        if mask_r.any():
            new_s[mask_r, 0:3] = tip_r[mask_r] + snap_offset_r
            new_s[mask_r, 3:7] = snap_quat
            new_s[mask_r, 7:13] = 0.0
            write_mask |= mask_r
        if mask_l.any():
            new_s[mask_l, 0:3] = tip_l[mask_l] + snap_offset_l
            new_s[mask_l, 3:7] = snap_quat
            new_s[mask_l, 7:13] = 0.0
            write_mask |= mask_l
        if mask_frozen.any():
            new_s[mask_frozen, 0:7] = sm.frozen_rings_pose[ring_name][mask_frozen]
            new_s[mask_frozen, 7:13] = 0.0
            write_mask |= mask_frozen
        if write_mask.any():
            env_ids_to_write = write_mask.nonzero(as_tuple=False).flatten()
            ring.write_root_state_to_sim(new_s[env_ids_to_write], env_ids=env_ids_to_write)