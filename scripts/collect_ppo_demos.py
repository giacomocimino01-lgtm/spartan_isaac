"""Collect expert demonstrations (observations and actions) for PPO Behavior Cloning.

This script runs the SPARTANStateMachine to solve the Peg-and-Ring task, wraps the environment
in DVRKVisionHRLWrapper (the same wrapper used in parallel_run.py), and records successful
(observation, action) pairs to a .npz file.

Example usage:
    PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 nohup ${IsaacLab_PATH}/isaaclab.sh -p scripts/collect_ppo_demos.py --num_envs 8 --num_episodes 50 --headless --save_videos > outlog.log 2>&1 &

"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import torchvision.io as io
torch.cuda.empty_cache()

from app_launcher_utils import pin_process_to_requested_cuda_device, resolve_tcc_checkpoint
from isaaclab.app import AppLauncher

# 1. PARSE ARGUMENTS
parser = argparse.ArgumentParser(description="Collect expert demonstrations for PPO Behavior Cloning.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel Isaac environments.")
parser.add_argument("--num_episodes", type=int, default=25, help="Number of successful episodes to collect.")
parser.add_argument(
    "--output_path",
    type=str,
    default="/home/aiprah/Documents/m_dVrk/dataset_supervisionato_randomized.npz",
    help="Output path for the collected dataset .npz file.",
)
parser.add_argument(
    "--save_videos",
    action="store_true",
    help="Save video frames of successful episodes as PNGs in a subfolder.",
)
parser.add_argument(
    "--randomize_rings",
    action="store_true",
    help="Use the environment random ring reset instead of the fixed deterministic layout.",
)
parser.add_argument(
    "--ring_order",
    choices=("random", "fixed"),
    default="random",
    help="Ring order inside each scripted video.",
)
parser.add_argument(
    "--arm_mode",
    choices=("right", "left", "random", "alternate"),
    default="random",
    help="Arm selection for each ring.",
)
parser.add_argument("--target_peg", type=str, default="peg_green", help="Peg where rings are placed.")
parser.add_argument(
    "--num_rings",
    type=int,
    default=4,
    choices=(1, 2, 3, 4),
    help="Number of rings to place in each episode.",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--max_steps_per_episode", type=int, default=6500, help="Discard episode after this many steps.")
parser.add_argument(
    "--task_phase",
    type=str,
    choices=["phase_0", "phase_1"],
    default="phase_0",
    help="Phase for the environment task.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The camera sensor requires Isaac rendering to be enabled.
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True

requested_device = getattr(args_cli, "device", None)
args_cli.device = pin_process_to_requested_cuda_device(requested_device)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from isaaclab.envs import ManagerBasedRLEnv
from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg
from parallel_env import RING_NAMES, SPARTANStateMachine, get_scene_entity_positions_w, sync_attached_and_frozen_rings

# 2. DEFINE MAPS AND WRAPPER (IDENTICAL TO parallel_run.py)
VERB_MAP = {0: "reach", 1: "grasp", 2: "release", 3: "idle"}
TARGET_MAP = {
    0: "ring_red", 1: "ring_yellow", 2: "ring_green", 3: "ring_blue",
    4: "peg_red", 5: "peg_yellow", 6: "peg_green", 7: "peg_blue",
    8: "peg_gray", 9: "None"
}

VERB_TO_ID = {v: k for k, v in VERB_MAP.items()}
TARGET_TO_ID = {v: k for k, v in TARGET_MAP.items()}
RING_TARGETS = {target for target in TARGET_MAP.values() if target.startswith("ring_")}
PEG_TARGETS = {target for target in TARGET_MAP.values() if target.startswith("peg_")}

IDLE_ACTION = torch.tensor(
    [
        VERB_TO_ID["idle"], TARGET_TO_ID["None"],  # left
        VERB_TO_ID["idle"], TARGET_TO_ID["None"],  # right
    ],
    dtype=torch.long,
)

class XIRLResnet18(nn.Module):
    def __init__(self, embedding_size=32):
        super().__init__()
        resnet = models.resnet18(weights=None)
        num_ftrs = resnet.fc.in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.encoder = nn.Linear(num_ftrs, embedding_size)

    def forward(self, x):
        feats = self.backbone(x)          # (B, 512, 1, 1)
        feats = torch.flatten(feats, 1)   # (B, 512)
        embs = self.encoder(feats)        # (B, 32)
        return embs

class DVRKVisionHRLWrapper(VecEnv):
    def __init__(self, isaac_env, state_machine, tcc_model=None, task_phase="phase_0"):
        self.env = isaac_env
        self.sm = state_machine
        self.tcc_model = tcc_model
        self.task_phase = task_phase
        
        self.num_envs = isaac_env.num_envs
        self.stack_size = 3                 
        self.emb_dim = 32
        
        act_space = spaces.MultiDiscrete([len(VERB_MAP), len(TARGET_MAP), len(VERB_MAP), len(TARGET_MAP)])
        
        self.aux_dim = 62
        self.geom_dim = 86
        self.obs_dim = self.stack_size * self.emb_dim + self.aux_dim + self.geom_dim

        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.render_mode = None
        super().__init__(self.num_envs, obs_space, act_space)

        self.prev_actions = IDLE_ACTION.to(self.env.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.last_override_flags = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.env.device)
        self.emb_buffer = torch.zeros((self.num_envs, self.stack_size, self.emb_dim), dtype=torch.float32, device=self.env.device)
        self.preprocess = T.Compose([T.Resize((112, 112), antialias=True)])
        self.current_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.env.device)
        self.goal_embedding = None
        self.actions = None

        _robot_r = self.env.scene["robot_right"]
        _robot_l = self.env.scene["robot_left"]
        self._tip_idx_r = _robot_r.find_bodies("psm_tool_tip_link")[0][0]
        self._tip_idx_l = _robot_l.find_bodies("psm_tool_tip_link")[0][0]

        self._ring_names = ["ring_red", "ring_yellow", "ring_green", "ring_blue"]
        self._peg_names = ["peg_red", "peg_yellow", "peg_green", "peg_blue"]
        self._cached_peg_pos_local = self._cache_static_peg_positions_local()
        self.success_frames = [None] * self.num_envs
        # Number of idle steps to run after task completion before declaring success.
        # This allows the rendering pipeline to propagate ring positions written via
        # write_root_state_to_sim to the visual/camera buffer.
        self.SETTLE_STEPS = 5
        self.settle_steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.env.device)

    def _cache_static_peg_positions_local(self):
        origins = self.env.scene.env_origins
        peg_pos_local = []
        for name in self._peg_names:
            pos_w = get_scene_entity_positions_w(self.env, name)
            peg_pos_local.append(pos_w - origins)
        return torch.stack(peg_pos_local, dim=1)

    def _peg_ring_count(self, peg_name):
        ring_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.env.device)
        for name in self._ring_names:
            on_peg = torch.tensor([self.sm.ring_support_peg[name][i] == peg_name for i in range(self.num_envs)], dtype=torch.bool, device=self.env.device)
            ring_count += (self.sm.frozen_rings_mask[name] & on_peg).long()
        return ring_count

    def _task_success(self):
        if self.task_phase == "phase_1":
            # For Phase 1: 2 rings on red, 2 rings on blue, all 4 rings frozen
            red_count = self._peg_ring_count("peg_red")
            blue_count = self._peg_ring_count("peg_blue")
            
            # Check if all 4 rings are frozen
            all_frozen = torch.ones(self.num_envs, dtype=torch.bool, device=self.env.device)
            for name in self._ring_names:
                all_frozen &= self.sm.frozen_rings_mask[name]
                
            return (red_count == 2) & (blue_count == 2) & all_frozen
        else:
            # Phase 0: 4 rings on green
            return self._peg_ring_count("peg_green") == 4

    def _actions_to_onehot(self, actions_tensor):
        """
        actions_tensor shape: (num_envs, 4)
        order: [verb_l, target_l, verb_r, target_r]
        returns: 28 dims
        """
        actions_tensor = actions_tensor.long()

        verb_l = F.one_hot(actions_tensor[:, 0], num_classes=len(VERB_MAP)).float()
        tgt_l = F.one_hot(actions_tensor[:, 1], num_classes=len(TARGET_MAP)).float()
        verb_r = F.one_hot(actions_tensor[:, 2], num_classes=len(VERB_MAP)).float()
        tgt_r = F.one_hot(actions_tensor[:, 3], num_classes=len(TARGET_MAP)).float()

        return torch.cat([verb_l, tgt_l, verb_r, tgt_r], dim=1)

    def _current_triplets_to_action_tensor(self):
        """
        Encode what SPARTAN is currently executing.
        If an arm is IDLE, encode idle->None.
        Order: [verb_l, target_l, verb_r, target_r]
        """
        device = self.env.device
        out = IDLE_ACTION.to(device).unsqueeze(0).repeat(self.num_envs, 1)

        for i in range(self.num_envs):
            if self.sm.sub_state_l[i] != "IDLE" and self.sm.current_triplet_l[i] is not None:
                cmd_l = self.sm.current_triplet_l[i]
                out[i, 0] = VERB_TO_ID.get(cmd_l["verb"], VERB_TO_ID["idle"])
                out[i, 1] = TARGET_TO_ID.get(cmd_l["target"], TARGET_TO_ID["None"])

            if self.sm.sub_state_r[i] != "IDLE" and self.sm.current_triplet_r[i] is not None:
                cmd_r = self.sm.current_triplet_r[i]
                out[i, 2] = VERB_TO_ID.get(cmd_r["verb"], VERB_TO_ID["idle"])
                out[i, 3] = TARGET_TO_ID.get(cmd_r["target"], TARGET_TO_ID["None"])

        return out

    def _get_manipulable_rings_mask(self) -> torch.Tensor:
        """Restituisce una maschera (num_envs, 4) con 1.0 se l'anello è manipolabile, 0.0 altrimenti."""
        device = self.env.device
        manipulable_mask = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=device)

        for env_i in range(self.num_envs):
            for ring_idx, ring_name in enumerate(self._ring_names):
                # 1. Se è già impugnato dal braccio sinistro o destro, non occorre fare re-grasp
                if (self.sm.attached_target_l[env_i] == ring_name
                    or self.sm.attached_target_r[env_i] == ring_name):
                    continue

                # 2. Verifichiamo la posizione dell'anello rispetto ai pioli
                current_peg = self.sm.ring_support_peg.get(ring_name, [None] * self.num_envs)[env_i]

                if current_peg is not None and current_peg != "None":
                    # Un ring frozen su un peg finale è già piazzato. In phase1, però,
                    # i ring partono frozen su peg_green e devono essere presi se top.
                    placed_on_final_peg = (
                        (self.task_phase == "phase_0" and current_peg == "peg_green")
                        or (self.task_phase == "phase_1" and current_peg in {"peg_red", "peg_blue"})
                    )
                    if self.sm.frozen_rings_mask[ring_name][env_i] and placed_on_final_peg:
                        continue

                    # L'anello è su un piolo: è manipolabile SOLO se è in cima.
                    top_rings = self.sm.get_top_rings_on_peg(env_i, current_peg, num_rings=1)
                    top_ring = top_rings[0] if top_rings else None
                    if top_ring == ring_name:
                        manipulable_mask[env_i, ring_idx] = 1.0
                else:
                    if self.sm.frozen_rings_mask[ring_name][env_i]:
                        continue
                    # L'anello NON è su un piolo (è libero nel workspace/tavolo)
                    ring_pos_w = get_scene_entity_positions_w(self.env, ring_name)[env_i]
                    origin_w = self.env.scene.env_origins[env_i]
                    rel_pos = ring_pos_w - origin_w  # Posizione locale [x, y, z]

                    in_workspace = (rel_pos[2] > -0.1) and (torch.norm(rel_pos[:2]) < 0.45)
                    if in_workspace:
                        manipulable_mask[env_i, ring_idx] = 1.0

        return manipulable_mask
    def _get_aux_obs(self):
        """
        62 dims:
            current active triplets      28
            previous requested triplets  28
            busy flags                    2
            override-busy flags           2
            gripper states                2
        """
        device = self.env.device

        current_triplet_actions = self._current_triplets_to_action_tensor()
        current_triplet_oh = self._actions_to_onehot(current_triplet_actions)

        prev_action_oh = self._actions_to_onehot(self.prev_actions)

        busy_l = torch.tensor(
            [s != "IDLE" for s in self.sm.sub_state_l],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)

        busy_r = torch.tensor(
            [s != "IDLE" for s in self.sm.sub_state_r],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)

        busy_flags = torch.cat([busy_l, busy_r], dim=1)

        gripper_states = torch.stack(
            [
                self.sm.current_gripper_state_l,
                self.sm.current_gripper_state_r,
            ],
            dim=1,
        )

        aux = torch.cat(
            [
                current_triplet_oh,        # 28
                prev_action_oh,            # 28
                busy_flags,                # 2
                self.last_override_flags,  # 2
                gripper_states,            # 2
            ],
            dim=1,
        )
        return aux
    
    def _get_geom_obs(self):
        device = self.env.device
        origins = self.env.scene.env_origins
        robot_r = self.env.scene["robot_right"]
        robot_l = self.env.scene["robot_left"]

        tip_r_w = robot_r.data.body_pos_w[:, self._tip_idx_r]
        tip_l_w = robot_l.data.body_pos_w[:, self._tip_idx_l]
        tip_r = tip_r_w - origins
        tip_l = tip_l_w - origins

        ring_pos_local = []
        for name in self._ring_names:
            pos_w = get_scene_entity_positions_w(self.env, name)
            ring_pos_local.append(pos_w - origins)

        ring_pos_local = torch.stack(ring_pos_local, dim=1)
        peg_pos_local = self._cached_peg_pos_local

        rel_tip_r_to_rings = ring_pos_local - tip_r.unsqueeze(1)
        rel_tip_l_to_rings = ring_pos_local - tip_l.unsqueeze(1)
        rel_ring_to_matching_peg = peg_pos_local - ring_pos_local

        attached_r = torch.zeros((self.num_envs, 4), device=device)
        attached_l = torch.zeros((self.num_envs, 4), device=device)
        for j, ring_name in enumerate(self._ring_names):
            for i in range(self.num_envs):
                attached_r[i, j] = 1.0 if self.sm.attached_target_r[i] == ring_name else 0.0
                attached_l[i, j] = 1.0 if self.sm.attached_target_l[i] == ring_name else 0.0

        frozen = torch.stack([self.sm.frozen_rings_mask[name].float() for name in self._ring_names], dim=1)
        inventory = torch.stack([self.sm.peg_inventory[name].float() for name in self._peg_names], dim=1) / 4.0

        manipulable_mask = self._get_manipulable_rings_mask()

        geom = torch.cat(
            [
                tip_r, tip_l,
                ring_pos_local.reshape(self.num_envs, -1),
                peg_pos_local.reshape(self.num_envs, -1),
                rel_tip_r_to_rings.reshape(self.num_envs, -1),
                rel_tip_l_to_rings.reshape(self.num_envs, -1),
                rel_ring_to_matching_peg.reshape(self.num_envs, -1),
                attached_r, attached_l, frozen, inventory,
                manipulable_mask,
            ],
            dim=1,
        )
        return geom

    def _get_batched_embeddings(self):
        rgb_data = self.env.scene.sensors["camera"].data.output["rgb"]
        model_device = next(self.tcc_model.parameters()).device
        raw_img_batch = rgb_data[:, :, :, :3].permute(0, 3, 1, 2).float().to(model_device) / 255.0
        img_batch = self.preprocess(raw_img_batch)
        with torch.no_grad():
            emb = self.tcc_model(img_batch)
        return emb.to(self.env.device)

    def reset(self):
        self.env.reset()
        self.current_step[:] = 0
        for i in range(self.num_envs):
            self.sm.reset_env(i, phase=self.task_phase)
        sync_attached_and_frozen_rings(self.env, self.sm, torch.ones(self.num_envs, dtype=torch.bool, device=self.env.device))
        self.prev_actions[:] = IDLE_ACTION.to(self.env.device).unsqueeze(0)
        self.last_override_flags.zero_()
        if self.tcc_model is not None:
            init_emb = self._get_batched_embeddings()
            self.emb_buffer[:] = init_emb.unsqueeze(1).repeat(1, self.stack_size, 1)
        else:
            self.emb_buffer.zero_()
        return self._get_obs()

    def step_async(self, actions):
        self.actions = actions

    def _sanitize_arm_command(self, env_i: int, verb: str, target: str, manipulable_mask: torch.Tensor):
        if verb == "idle":
            return "idle", "None", False
        if target == "None":
            return "idle", "None", True

        if verb == "grasp":
            if target not in RING_TARGETS:
                return "idle", "None", True

            current_peg = self.sm.ring_support_peg.get(target, [None] * self.num_envs)[env_i]
            if current_peg is not None and current_peg != "None":
                top_rings = self.sm.get_top_rings_on_peg(env_i, current_peg, num_rings=1)
                top_ring = top_rings[0] if top_rings else None
                if top_ring != target:
                    return "idle", "None", True

        if verb == "release":
            return verb, target, False
        return verb, target, False

    def step_wait(self):
        actions = self.actions
        sanitized_actions = torch.as_tensor(actions, dtype=torch.long, device=self.env.device).clone()
        override_flags = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.env.device)
        invalid_command_counts = torch.zeros(self.num_envs, dtype=torch.float32, device=self.env.device)
        manipulable_masks = self._get_manipulable_rings_mask()

        for i in range(self.num_envs):
            was_busy_l = self.sm.sub_state_l[i] != "IDLE"
            was_busy_r = self.sm.sub_state_r[i] != "IDLE"
            old_cmd_l = self.sm.current_triplet_l[i]
            old_cmd_r = self.sm.current_triplet_r[i]

            verb_l, tgt_l = VERB_MAP[int(actions[i, 0])], TARGET_MAP[int(actions[i, 1])]
            verb_r, tgt_r = VERB_MAP[int(actions[i, 2])], TARGET_MAP[int(actions[i, 3])]

            verb_l, tgt_l, invalid_l = self._sanitize_arm_command(i, verb_l, tgt_l, manipulable_masks)
            verb_r, tgt_r, invalid_r = self._sanitize_arm_command(i, verb_r, tgt_r, manipulable_masks)

            sanitized_actions[i, 0] = VERB_TO_ID[verb_l]
            sanitized_actions[i, 1] = TARGET_TO_ID[tgt_l]
            sanitized_actions[i, 2] = VERB_TO_ID[verb_r]
            sanitized_actions[i, 3] = TARGET_TO_ID[tgt_r]

            new_cmd_l = {"verb": verb_l, "subject": "left_arm", "target": tgt_l}
            new_cmd_r = {"verb": verb_r, "subject": "right_arm", "target": tgt_r}

            override_flags[i, 0] = 1.0 if was_busy_l and old_cmd_l != new_cmd_l else 0.0
            override_flags[i, 1] = 1.0 if was_busy_r and old_cmd_r != new_cmd_r else 0.0
            invalid_command_counts[i] = float(invalid_l and not was_busy_l) + float(invalid_r and not was_busy_r)

            if was_busy_l and old_cmd_l is not None:
                sanitized_actions[i, 0] = VERB_TO_ID.get(old_cmd_l["verb"], VERB_TO_ID["idle"])
                sanitized_actions[i, 1] = TARGET_TO_ID.get(old_cmd_l["target"], TARGET_TO_ID["None"])
            if was_busy_r and old_cmd_r is not None:
                sanitized_actions[i, 2] = VERB_TO_ID.get(old_cmd_r["verb"], VERB_TO_ID["idle"])
                sanitized_actions[i, 3] = TARGET_TO_ID.get(old_cmd_r["target"], TARGET_TO_ID["None"])

            if not was_busy_l:
                self.sm.set_new_triplet(verb_l, "left_arm", tgt_l, i)
            if not was_busy_r:
                self.sm.set_new_triplet(verb_r, "right_arm", tgt_r, i)

        self.prev_actions = sanitized_actions
        self.last_override_flags = override_flags
        
        robot_r = self.env.scene["robot_right"]
        robot_l = self.env.scene["robot_left"]
        _snap_offset_r = torch.tensor([-0.005, 0.0, 0.0], device=self.env.device)
        _snap_offset_l = torch.tensor([+0.005, 0.0, 0.0], device=self.env.device)

        continuous_actions = self.sm.get_action()
        _, _, terminated_il, truncated_il, _ = self.env.step(continuous_actions)
        terminated_il = terminated_il.to(device=self.env.device, dtype=torch.bool)
        truncated_il = truncated_il.to(device=self.env.device, dtype=torch.bool)
        isaac_done = terminated_il | truncated_il
        active_env_mask = ~isaac_done

        tip_r = robot_r.data.body_pos_w[:, self._tip_idx_r]  
        tip_l = robot_l.data.body_pos_w[:, self._tip_idx_l]  
        snap_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.env.device)

        for ring_name in ["ring_red", "ring_yellow", "ring_green", "ring_blue"]:
            ring = self.env.scene[ring_name]
            new_s = ring.data.root_state_w.clone()

            mask_r = torch.tensor([self.sm.attached_target_r[i] == ring_name for i in range(self.num_envs)], dtype=torch.bool, device=self.env.device) & active_env_mask
            mask_l = torch.tensor([self.sm.attached_target_l[i] == ring_name for i in range(self.num_envs)], dtype=torch.bool, device=self.env.device) & active_env_mask
            mask_frozen = self.sm.frozen_rings_mask[ring_name] & active_env_mask

            write_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.device)
            if mask_r.any():
                new_s[mask_r, 0:3] = tip_r[mask_r] + _snap_offset_r
                new_s[mask_r, 3:7] = snap_quat
                new_s[mask_r, 7:13] = 0.0
                write_mask |= mask_r
            if mask_l.any():
                new_s[mask_l, 0:3] = tip_l[mask_l] + _snap_offset_l
                new_s[mask_l, 3:7] = snap_quat
                new_s[mask_l, 7:13] = 0.0
                write_mask |= mask_l
            if mask_frozen.any():
                new_s[mask_frozen, 0:7] = self.sm.frozen_rings_pose[ring_name][mask_frozen]
                new_s[mask_frozen, 7:13] = 0.0 
                write_mask |= mask_frozen 

            if write_mask.any():
                env_ids_to_write = write_mask.nonzero(as_tuple=False).flatten()
                ring.write_root_state_to_sim(new_s[env_ids_to_write], env_ids=env_ids_to_write)

        new_embs = self._get_batched_embeddings() 
        self.emb_buffer = torch.roll(self.emb_buffer, shifts=-1, dims=1)
        self.emb_buffer[:, -1, :] = new_embs
        self.current_step += 1
        
        rew_distance_raw = torch.norm(new_embs - self.goal_embedding, dim=1) ** 2 if self.goal_embedding is not None else torch.zeros(self.num_envs, device=self.env.device)
        rewards = -rew_distance_raw * 1e-3
        rewards -= invalid_command_counts * 1.0

        green_peg_ring_count = self._peg_ring_count("peg_green")
        rings_complete = self._task_success()
        # Start settle countdown when all rings are on the target peg
        just_completed = rings_complete & (self.settle_steps_remaining == 0)
        self.settle_steps_remaining[just_completed] = self.SETTLE_STEPS
        # Decrement active countdowns
        settling = self.settle_steps_remaining > 0
        self.settle_steps_remaining[settling] -= 1
        # Prefer the settled visual state, but do not turn a logically complete
        # task into a failure just because Isaac time limit fired first.
        settled_success = rings_complete & (self.settle_steps_remaining == 0)
        task_success = settled_success | (rings_complete & isaac_done)
        dones = isaac_done | task_success
        dones_numpy = dones.cpu().numpy()

        terminal_obs = self._get_obs()
        infos = [{} for _ in range(self.num_envs)]
        terminated_np = terminated_il.cpu().numpy()
        truncated_np = truncated_il.cpu().numpy()

        for idx in dones.nonzero(as_tuple=False).flatten().tolist():
            infos[idx]["terminal_observation"] = terminal_obs[idx].copy()
            infos[idx]["TimeLimit.truncated"] = bool(truncated_np[idx] and not terminated_np[idx])
            infos[idx]["is_success"] = bool(task_success[idx].item())
            infos[idx]["rings_complete"] = bool(rings_complete[idx].item())
            infos[idx]["terminal_distance"] = float(rew_distance_raw[idx].item())
            infos[idx]["green_peg_ring_count"] = int(green_peg_ring_count[idx].item())
            if self.task_phase == "phase_1":
                infos[idx]["red_peg_ring_count"] = int(self._peg_ring_count("peg_red")[idx].item())
                infos[idx]["blue_peg_ring_count"] = int(self._peg_ring_count("peg_blue")[idx].item())

        manual_reset = dones & (~isaac_done)
        if manual_reset.any():
            reset_ids = manual_reset.nonzero(as_tuple=False).flatten()
            self.env.reset(env_ids=reset_ids)

        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).flatten().tolist()
            for idx in done_ids:
                self.sm.reset_env(idx, phase=self.task_phase)
                self.current_step[idx] = 0
                self.settle_steps_remaining[idx] = 0
                self.prev_actions[idx] = IDLE_ACTION.to(self.env.device)
                self.last_override_flags[idx] = 0.0

            # Sync newly reset environments
            sync_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.device)
            for idx in done_ids:
                sync_mask[idx] = True
            sync_attached_and_frozen_rings(self.env, self.sm, sync_mask)

            reset_embs = self._get_batched_embeddings()
            for idx in done_ids:
                self.emb_buffer[idx] = reset_embs[idx].unsqueeze(0).repeat(self.stack_size, 1)
        
        obs_after_reset = self._get_obs()
        return obs_after_reset, rewards.cpu().numpy(), dones_numpy, infos

    def _get_obs(self):
        emb_obs = self.emb_buffer.view(self.num_envs, -1)
        aux_obs = self._get_aux_obs()
        geom_obs = self._get_geom_obs()
        obs = torch.cat([emb_obs, aux_obs, geom_obs], dim=1)
        return obs.cpu().numpy().astype(np.float32)
    
    def get_attr(self, attr_name, indices=None):
        n = self.num_envs if indices is None else (1 if isinstance(indices, int) else len(indices))
        return [getattr(self, attr_name, None)] * n
    def set_attr(self, attr_name, value, indices=None): pass
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): 
        n = self.num_envs if indices is None else (1 if isinstance(indices, int) else len(indices))
        return [None] * n
    def env_is_wrapped(self, wrapper_class, indices=None): 
        n = self.num_envs if indices is None else (1 if isinstance(indices, int) else len(indices))
        return [False] * n
    def close(self): pass

# 3. HELPER CLASSES FOR DATA COLLECTION
class EpisodeSlot:
    def __init__(self):
        self.commands: list[tuple[str, str, str]] = []
        self.command_idx = 0
        self.step_idx = 0
        self.done = True
        self.targeted_rings: list[str] = []
        self.obs_buffer: list[np.ndarray] = []
        self.actions_buffer: list[np.ndarray] = []
        self.rgb_buffer: list[torch.Tensor] = []

def choose_arm(mode: str, ring_idx: int) -> str:
    if mode == "right":
        return "right_arm"
    if mode == "left":
        return "left_arm"
    if mode == "alternate":
        return "right_arm" if ring_idx % 2 == 0 else "left_arm"
    return random.choice(["right_arm", "left_arm"])

def build_commands_for_env(sm: SPARTANStateMachine, env_id: int, task_phase: str, ring_order: str, arm_mode: str, target_peg: str, num_rings: int = 4) -> tuple[list[tuple[str, str, str]], list[str]]:
    if task_phase == "phase_1":
        rings = sm.get_top_rings_on_peg(env_id, "peg_green", num_rings=4)
        dest_pegs = ["peg_red", "peg_red", "peg_blue", "peg_blue"]
        random.shuffle(dest_pegs)
        
        commands: list[tuple[str, str, str]] = []
        for ring_idx, ring_name in enumerate(rings):
            arm = choose_arm(arm_mode, ring_idx)
            dest_peg = dest_pegs[ring_idx]
            commands.extend(
                [
                    ("reach", arm, ring_name),
                    ("grasp", arm, ring_name),
                    ("reach", arm, dest_peg),
                    ("release", arm, dest_peg),
                ]
            )
        return commands, rings
    else:
        rings = RING_NAMES.copy()
        if ring_order == "random":
            random.shuffle(rings)
        rings = rings[:num_rings]
        commands: list[tuple[str, str, str]] = []
        for ring_idx, ring_name in enumerate(rings):
            arm = choose_arm(arm_mode, ring_idx)
            commands.extend(
                [
                    ("reach", arm, ring_name),
                    ("grasp", arm, ring_name),
                    ("reach", arm, target_peg),
                    ("release", arm, target_peg),
                ]
            )
        return commands, rings

def start_episode(env, sm: SPARTANStateMachine, slots: list[EpisodeSlot], env_id: int):
    slot = slots[env_id]
    
    # Reset state machine first to establish the stacked rings positions/order
    sm.reset_env(env_id, phase=args_cli.task_phase)
    
    # Sync the initial state of this environment immediately to place rings in simulator
    sync_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    sync_mask[env_id] = True
    sync_attached_and_frozen_rings(env, sm, sync_mask)

    slot.commands, slot.targeted_rings = build_commands_for_env(
        sm, env_id, args_cli.task_phase, args_cli.ring_order, args_cli.arm_mode, args_cli.target_peg, args_cli.num_rings
    )
    slot.command_idx = 0
    slot.step_idx = 0
    slot.done = False
    slot.obs_buffer.clear()
    slot.actions_buffer.clear()
    slot.rgb_buffer.clear()

def get_triplet_action_for_env(sm: SPARTANStateMachine, slot: EpisodeSlot, env_id: int) -> tuple[int, int, int, int]:
    # Default to idle
    verb_l, tgt_l = VERB_TO_ID["idle"], TARGET_TO_ID["None"]
    verb_r, tgt_r = VERB_TO_ID["idle"], TARGET_TO_ID["None"]

    # If the state machine is ready for a new command, issue it
    if sm.all_idle(env_id) and slot.command_idx < len(slot.commands):
        verb, arm, target = slot.commands[slot.command_idx]
        slot.command_idx += 1
        print(f"[collector] Env {env_id} Command {slot.command_idx}/{len(slot.commands)}: {arm} {verb}->{target}")
        
        if arm == "left_arm":
            verb_l = VERB_TO_ID[verb]
            tgt_l = TARGET_TO_ID[target]
        else:
            verb_r = VERB_TO_ID[verb]
            tgt_r = TARGET_TO_ID[target]

    return verb_l, tgt_l, verb_r, tgt_r

# 4. MAIN DATA COLLECTION LOOP
def main():
    if args_cli.task_phase == "phase_1":
        args_cli.num_rings = 4
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    env_cfg = MDvrkEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.num_rerenders_on_reset = 2
    env_cfg.events.reset_rings.params["randomize"] = bool(args_cli.randomize_rings)

    print(f"[collector] Starting environment on device {env_cfg.sim.device}...")
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    sm = SPARTANStateMachine(isaac_env)

    # Load TCC Model weights to ensure valid observation embedding stacks
    tcc = XIRLResnet18(embedding_size=32).to(isaac_env.device)
    try:
        experiment_dir = f"/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_{args_cli.task_phase.replace('_', '')}_tcc"
        tcc_ckpt_path = resolve_tcc_checkpoint(experiment_dir)
        print(f"[collector] Loading XIRL TCC weights from: {tcc_ckpt_path}")
        ckpt = torch.load(
            tcc_ckpt_path,
            map_location=isaac_env.device,
        )
        sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
        clean_sd = {} 
        for k, v in sd.items():
            if k.startswith("module."): k = k[len("module."):]
            if k.startswith("model."): k = k[len("model."):]
            clean_sd[k] = v
        tcc.load_state_dict(clean_sd, strict=True)
        print("[collector] XIRL TCC weights loaded successfully.")
    except Exception as e:
        print(f"[collector] WARNING: Failed to load XIRL weights: {e}. Observation embeddings might be noisy.")

    tcc.eval()

    rl_env = DVRKVisionHRLWrapper(isaac_env, sm, tcc, task_phase=args_cli.task_phase)
    
    # Initialize Episode slots
    slots = [EpisodeSlot() for _ in range(args_cli.num_envs)]
    
    successful_episodes_count = 0
    all_observations = []
    all_actions = []

    # Reset environment to get initial observations
    obs = rl_env.reset()

    # Start initial episodes
    for env_id in range(args_cli.num_envs):
        start_episode(isaac_env, sm, slots, env_id)

    print(f"[collector] Starting collection of {args_cli.num_episodes} successful episodes...")

    while simulation_app.is_running() and successful_episodes_count < args_cli.num_episodes:
        # Determine high-level commands to step the environment
        step_actions = np.zeros((args_cli.num_envs, 4), dtype=np.int64)
        for env_id in range(args_cli.num_envs):
            slot = slots[env_id]
            if not slot.done:
                verb_l, tgt_l, verb_r, tgt_r = get_triplet_action_for_env(sm, slot, env_id)
                step_actions[env_id] = [verb_l, tgt_l, verb_r, tgt_r]

        # Save current observations and actions to buffer BEFORE stepping the environment.
        # This aligns the state observed at time t with the action executed at time t.
        # Note: the wrapper will modify continuous commands inside step_wait, but it keeps track
        # of the active high-level discrete actions in `prev_actions` which is what we need to record.
        for env_id in range(args_cli.num_envs):
            slot = slots[env_id]
            if not slot.done:
                slot.obs_buffer.append(obs[env_id].copy())
                if args_cli.save_videos:
                    rgb_frame = rl_env.env.scene.sensors["camera"].data.output["rgb"][env_id, :, :, :3].cpu()
                    slot.rgb_buffer.append(rgb_frame)

        # Step the environment
        next_obs, rewards, dones, infos = rl_env.step(step_actions)

        # Retrieve the actual actions that were executed by the wrapper (handling arm busy overrides)
        executed_actions = rl_env.prev_actions.cpu().numpy()

        for env_id in range(args_cli.num_envs):
            slot = slots[env_id]
            if not slot.done:
                slot.actions_buffer.append(executed_actions[env_id].copy())
                slot.step_idx += 1

                # Check if episode ended
                if dones[env_id]:
                    is_success = infos[env_id].get("is_success", False)
                    # Verify task logical completion
                    sm_success = sm.successful(env_id, phase=args_cli.task_phase)
                    
                    if is_success or sm_success:
                        successful_episodes_count += 1
                        all_observations.extend(slot.obs_buffer)
                        all_actions.extend(slot.actions_buffer)
                        print(
                            f"[collector] Episode SUCCESS! Saved {len(slot.obs_buffer)} steps. "
                            f"Progress: {successful_episodes_count}/{args_cli.num_episodes}"
                        )
                        if args_cli.save_videos:

                            video_dir = Path("/home/aiprah/Documents/m_dVrk/video_raccolti") / f"episodio_{successful_episodes_count:04d}"
                            video_dir.mkdir(parents=True, exist_ok=True)
                            for frame_idx, img in enumerate(slot.rgb_buffer):
                                if img.dtype != torch.uint8:
                                    img = (img.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
                                img = img.permute(2, 0, 1).contiguous()
                                io.write_png(img, str(video_dir / f"{frame_idx:06d}.png"))
                            print(f"[collector] Video frames saved to {video_dir}")
                    else:
                        print(f"[collector] Episode FAILED (steps: {slot.step_idx}). Discarding buffer.")

                    # Start a new episode in this environment
                    if successful_episodes_count < args_cli.num_episodes:
                        start_episode(isaac_env, sm, slots, env_id)
                    else:
                        slot.done = True
                
                # Check for max step timeout
                elif slot.step_idx >= args_cli.max_steps_per_episode:
                    print(f"[collector] Episode TIMEOUT in env {env_id}. Discarding buffer.")
                    start_episode(isaac_env, sm, slots, env_id)

        obs = next_obs

    # Save collected dataset to disk
    if len(all_observations) > 0:
        os.makedirs(os.path.dirname(args_cli.output_path), exist_ok=True)
        np.savez(
            args_cli.output_path,
            obs=np.array(all_observations, dtype=np.float32),
            actions=np.array(all_actions, dtype=np.int64),
        )
        print(f"[collector] Saved dataset successfully to {args_cli.output_path}")
        print(f"[collector] Total transitions collected: {len(all_observations)}")
    else:
        print("[collector] ERROR: No transitions were collected.")

    rl_env.close()
    isaac_env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
