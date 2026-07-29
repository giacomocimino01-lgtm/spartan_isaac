import argparse
import threading
import os
import glob
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import torchvision.io as io
import cv2

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

# 1. INITIAL SETUP OF ISAAC SIM
from isaaclab.app import AppLauncher
from app_launcher_utils import resolve_tcc_checkpoint

parser = argparse.ArgumentParser(description="Inference evaluation of RL agent.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (default 1 for evaluation).")
parser.add_argument("--checkpoint", type=str, default="/home/aiprah/Documents/m_dVrk/modelli_salvati_sim/dvrk_ppo_best_terminal_distance.zip", help="Path to the .zip model file to load.")
parser.add_argument("--tcc_checkpoint", type=str, default="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase0_tcc/checkpoints/4001.ckpt", help="Path to the TCC representation weights.")
parser.add_argument("--dataset_path", type=str, default="/mnt/data/aiprah/data/sim_dataset_xirl_extra/train/phase_0/", help="Path to Xirl demonstrations dataset for computing mean goal embedding.")
parser.add_argument(
    "--randomize_rings",
    action="store_true",
    help="Randomize ring reset positions. By default, rings reset to a fixed deterministic layout.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The camera sensor requires Isaac rendering to be enabled.
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils
from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg 

from parallel_env import SPARTANStateMachine, get_scene_entity_positions_w

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

PHYSICS_STEPS_PER_RL_STEP = 400
INVALID_COMMAND_PENALTY = 1.0
REWARD_DEBUG_DUMP_INTERVAL = 500
REWARD_DEBUG_DUMP_ENV_IDS = (0,)
REWARD_DEBUG_DUMP_DIR = "reward_debug_frames"
REWARD_DEBUG_DUMP_MAX_PER_ENV = 400

# ==========================================
# CLASS XIRL: CLONE OF THE PRETRAINED TCC NETWORK
# ==========================================
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

# ==========================================
# CLASS 1.5: THE WRAPPER RL
# ==========================================
class DVRKVisionHRLWrapper(VecEnv):
    def __init__(self, isaac_env, state_machine, tcc_model=None):
        self.env = isaac_env
        self.sm = state_machine
        self.tcc_model = tcc_model
        
        self.num_envs = isaac_env.num_envs
        self.stack_size = 3                 
        self.emb_dim = 32
        
        # ACTION SPACE: [verb_l, tgt_l, verb_r, tgt_r]
        act_space = spaces.MultiDiscrete([len(VERB_MAP), len(TARGET_MAP), len(VERB_MAP), len(TARGET_MAP)])
        
        # OBSERVATION SPACE: [emb_stack (stack_size*emb_dim), aux_info (62)]
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

        self.reward_debug_dump_interval = max(int(REWARD_DEBUG_DUMP_INTERVAL), 0)
        self.reward_debug_dump_dir = REWARD_DEBUG_DUMP_DIR
        self.reward_debug_dump_max_per_env = max(int(REWARD_DEBUG_DUMP_MAX_PER_ENV), 0)
        self.reward_debug_dump_env_ids = self._sanitize_reward_dump_env_ids(REWARD_DEBUG_DUMP_ENV_IDS)
        self.reward_debug_dump_counts = [0] * self.num_envs
        self.reward_debug_last_dump_step = [-1] * self.num_envs

        if self.reward_debug_dump_interval > 0 and self.reward_debug_dump_env_ids and self.reward_debug_dump_max_per_env > 0:
            os.makedirs(self.reward_debug_dump_dir, exist_ok=True)
            print(
                f"[RewardDebug] Active dump: every={self.reward_debug_dump_interval} steps, "
                f"envs={self.reward_debug_dump_env_ids}, dir={self.reward_debug_dump_dir}"
            )

        # Number of idle steps to run after task completion before declaring success.
        # This allows the rendering pipeline to propagate ring positions written via
        # write_root_state_to_sim to the visual/camera buffer.
        self.SETTLE_STEPS = 5
        self.settle_steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.env.device)

        # We need to keep stored the last action to feed it into the state machine logic
        self.prev_actions = IDLE_ACTION.to(self.env.device).unsqueeze(0).repeat(self.num_envs, 1)

        # [left_overrode_busy, right_overrode_busy]
        self.last_override_flags = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.float32,
            device=self.env.device,
        )
        
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

    def _cache_static_peg_positions_local(self):
        origins = self.env.scene.env_origins
        peg_pos_local = []
        for name in self._peg_names:
            pos_w = get_scene_entity_positions_w(self.env, name)
            peg_pos_local.append(pos_w - origins)
        return torch.stack(peg_pos_local, dim=1)

    def _sanitize_reward_dump_env_ids(self, env_ids_spec):
        env_ids = []
        for raw_env_id in env_ids_spec:
            try:
                env_id = int(raw_env_id)
            except ValueError:
                print(f"[RewardDebug] Ignoring invalid env id: {raw_env_id}")
                continue

            if 0 <= env_id < self.num_envs:
                env_ids.append(env_id)
            else:
                print(f"[RewardDebug] Ignoring out-of-range env id: {env_id}")

        return sorted(set(env_ids))

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

        assert aux.shape[1] == self.aux_dim, f"Expected {self.aux_dim}, got {aux.shape[1]}"
        return aux

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
    def _get_geom_obs(self):
        """
        Privileged low-dimensional geometry observations.

        Returned dimensions:
            tip local positions                  6
            ring local positions                12
            peg local positions                 12
            right tip -> rings                  12
            left tip -> rings                   12
            ring -> matching peg                12
            attached flags                       8
            frozen flags                         4
            peg inventory                        4
            manipulable flags                    4
        Total:                                  86
        """
        device = self.env.device
        origins = self.env.scene.env_origins  # (N, 3)

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

        ring_pos_local = torch.stack(ring_pos_local, dim=1)  # (N, 4, 3)
        peg_pos_local = self._cached_peg_pos_local

        # Relative vectors from each tip to each ring.
        rel_tip_r_to_rings = ring_pos_local - tip_r.unsqueeze(1)  # (N, 4, 3)
        rel_tip_l_to_rings = ring_pos_local - tip_l.unsqueeze(1)  # (N, 4, 3)

        # Matching pairs: red ring -> red peg, yellow -> yellow, etc.
        rel_ring_to_matching_peg = peg_pos_local - ring_pos_local  # (N, 4, 3)

        # Attached flags.
        attached_r = torch.zeros((self.num_envs, 4), device=device)
        attached_l = torch.zeros((self.num_envs, 4), device=device)

        for j, ring_name in enumerate(self._ring_names):
            for i in range(self.num_envs):
                attached_r[i, j] = 1.0 if self.sm.attached_target_r[i] == ring_name else 0.0
                attached_l[i, j] = 1.0 if self.sm.attached_target_l[i] == ring_name else 0.0

        # Frozen/placed flags.
        frozen = torch.stack(
            [self.sm.frozen_rings_mask[name].float() for name in self._ring_names],
            dim=1,
        )  # (N, 4)

        # Peg inventory, normalized a bit.
        inventory = torch.stack(
            [self.sm.peg_inventory[name].float() for name in self._peg_names],
            dim=1,
        ) / 4.0

        geom = torch.cat(
            [
                tip_r,
                tip_l,
                ring_pos_local.reshape(self.num_envs, -1),
                peg_pos_local.reshape(self.num_envs, -1),
                rel_tip_r_to_rings.reshape(self.num_envs, -1),
                rel_tip_l_to_rings.reshape(self.num_envs, -1),
                rel_ring_to_matching_peg.reshape(self.num_envs, -1),
                attached_r,
                attached_l,
                frozen,
                inventory,
                manipulable_mask,
            ],
            dim=1,
        )

        assert geom.shape[1] == 86, f"Expected geom obs dim 86, got {geom.shape[1]}"
        return geom

    def _green_peg_ring_count(self):
        counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.env.device)
        for ring_name in ["ring_red", "ring_yellow", "ring_green", "ring_blue"]:
            on_green = torch.tensor(
                [self.sm.ring_support_peg[ring_name][i] == "peg_green" for i in range(self.num_envs)],
                dtype=torch.bool,
                device=self.env.device,
            )
            counts += (self.sm.frozen_rings_mask[ring_name] & on_green).long()
        return counts

    def _task_success(self):
        return self._green_peg_ring_count() == 4

    def _write_debug_png(self, img_tensor, out_path):
        img_u8 = torch.clamp(img_tensor.detach().cpu(), 0.0, 1.0)
        img_u8 = (img_u8 * 255.0).round().to(torch.uint8)
        io.write_png(img_u8, out_path)

    def _maybe_dump_reward_images(self, raw_img_batch, model_img_batch):
        if self.reward_debug_dump_interval <= 0:
            return
        if not self.reward_debug_dump_env_ids:
            return
        if self.reward_debug_dump_max_per_env <= 0:
            return

        for env_id in self.reward_debug_dump_env_ids:
            if self.reward_debug_dump_counts[env_id] >= self.reward_debug_dump_max_per_env:
                continue

            step = int(self.current_step[env_id].item())
            if step <= 0 or step % self.reward_debug_dump_interval != 0:
                continue
            if self.reward_debug_last_dump_step[env_id] == step:
                continue

            dump_idx = self.reward_debug_dump_counts[env_id]
            base_path = os.path.join(
                self.reward_debug_dump_dir,
                f"env_{env_id:03d}_dump_{dump_idx:03d}_step_{step:07d}",
            )
            self._write_debug_png(raw_img_batch[env_id], f"{base_path}_reward_raw.png")
            self._write_debug_png(model_img_batch[env_id], f"{base_path}_reward_input.png")

            self.reward_debug_dump_counts[env_id] += 1
            self.reward_debug_last_dump_step[env_id] = step

    def _get_batched_embeddings(self, dump_debug=False):
        rgb_data = self.env.scene.sensors["camera"].data.output["rgb"]

        model_device = next(self.tcc_model.parameters()).device

        raw_img_batch = (
            rgb_data[:, :, :, :3]
            .permute(0, 3, 1, 2)
            .float()
            .to(model_device)
            / 255.0
        )

        img_batch = self.preprocess(raw_img_batch)

        if dump_debug:
            self._maybe_dump_reward_images(raw_img_batch, img_batch)

        with torch.no_grad():
            emb = self.tcc_model(img_batch)

        return emb.to(self.env.device) 

    def reset(self):
        self.env.reset()
        self.current_step[:] = 0

        for i in range(self.num_envs):
            self.sm.reset_env(i)

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

    def _sanitize_arm_command(self, verb: str, target: str):
        if verb == "idle":
            return "idle", "None", False

        if verb == "grasp" and target not in RING_TARGETS:
            return "idle", "None", True

        if verb == "release":
            return verb, target, False

        if target == "None":
            return "idle", "None", True

        return verb, target, False

    def step_wait(self):
        actions = self.actions

        sanitized_actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.env.device,
        ).clone()

        override_flags = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.float32,
            device=self.env.device,
        )
        invalid_command_counts = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.env.device,
        )

        for i in range(self.num_envs):
            was_busy_l = self.sm.sub_state_l[i] != "IDLE"
            was_busy_r = self.sm.sub_state_r[i] != "IDLE"

            old_cmd_l = self.sm.current_triplet_l[i]
            old_cmd_r = self.sm.current_triplet_r[i]

            verb_l, tgt_l = VERB_MAP[int(actions[i, 0])], TARGET_MAP[int(actions[i, 1])]
            verb_r, tgt_r = VERB_MAP[int(actions[i, 2])], TARGET_MAP[int(actions[i, 3])]

            verb_l, tgt_l, invalid_l = self._sanitize_arm_command(verb_l, tgt_l)
            verb_r, tgt_r, invalid_r = self._sanitize_arm_command(verb_r,tgt_r)

            sanitized_actions[i, 0] = VERB_TO_ID[verb_l]
            sanitized_actions[i, 1] = TARGET_TO_ID[tgt_l]
            sanitized_actions[i, 2] = VERB_TO_ID[verb_r]
            sanitized_actions[i, 3] = TARGET_TO_ID[tgt_r]

            new_cmd_l = {"verb": verb_l, "subject": "left_arm", "target": tgt_l}
            new_cmd_r = {"verb": verb_r, "subject": "right_arm", "target": tgt_r}

            # This is only diagnostic context for the policy.
            # If an arm is busy, the new request is ignored until that arm is idle.
            override_flags[i, 0] = 1.0 if was_busy_l and old_cmd_l != new_cmd_l else 0.0
            override_flags[i, 1] = 1.0 if was_busy_r and old_cmd_r != new_cmd_r else 0.0

            # Penalize invalid commands only if that arm was actually free.
            # If the arm was busy, the command was ignored, so it should not receive
            # an invalid-command penalty.
            invalid_command_counts[i] = (
                float(invalid_l and not was_busy_l)
                + float(invalid_r and not was_busy_r)
            )

            # If an arm is busy, keep prev_actions aligned with the command actually
            # being executed, not the ignored newly sampled command.
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

            # IsaacLab already resets Isaac-done environments inside env.step().
            # Do not overwrite those fresh reset poses with stale pre-reset
            # attached/frozen ring state before the state machine is reset below.
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

        new_embs = self._get_batched_embeddings(dump_debug=True) 
        self.emb_buffer = torch.roll(self.emb_buffer, shifts=-1, dims=1)
        self.emb_buffer[:, -1, :] = new_embs
        
        step0 = self.current_step[0].item()
        if step0 % 10 == 0:
            cmd_r = self.sm.current_triplet_r[0]
            cmd_l = self.sm.current_triplet_l[0]
            exec_r = continuous_actions[0, 0:3].detach().cpu().tolist()
            grip_r = continuous_actions[0, 7:9].detach().cpu().tolist()
            exec_l = continuous_actions[0, 9:12].detach().cpu().tolist()
            grip_l = continuous_actions[0, 16:18].detach().cpu().tolist()
            tip_r0 = tip_r[0].detach().cpu().tolist()
            tip_l0 = tip_l[0].detach().cpu().tolist()
            print(
                f"[ENV0 Step {step0}] "
                f"R: {cmd_r['verb'] if cmd_r else 'idle'}->{cmd_r['target'] if cmd_r else 'None'} ({self.sm.sub_state_r[0]}) | "
                f"L: {cmd_l['verb'] if cmd_l else 'idle'}->{cmd_l['target'] if cmd_l else 'None'} ({self.sm.sub_state_l[0]}) | "
                f"exec_r={exec_r} grip_r={grip_r} tip_r={tip_r0} "
                f"exec_l={exec_l} grip_l={grip_l} tip_l={tip_l0}"
            )

        self.current_step += 1
        
        # --- Reward ---
        rew_distance_raw = torch.norm(new_embs - self.goal_embedding, dim=1) ** 2
        rewards = -rew_distance_raw * 1e-3
        rewards -= invalid_command_counts * INVALID_COMMAND_PENALTY

        green_peg_ring_count = self._green_peg_ring_count()
        rings_complete = green_peg_ring_count == 4
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
        # ------------------------------

        # --------------------------------------------------
        # Done flags
        # --------------------------------------------------
        dones = isaac_done | task_success
        dones_numpy = dones.cpu().numpy()

        # IMPORTANT:
        # Save terminal obs BEFORE resetting envs / buffers / state machine.
        terminal_obs = self._get_obs()

        infos = [{} for _ in range(self.num_envs)]

        for idx, invalid_count in enumerate(invalid_command_counts.cpu().tolist()):
            if invalid_count:
                infos[idx]["invalid_command_count"] = int(invalid_count)

        terminated_np = terminated_il.cpu().numpy()
        truncated_np = truncated_il.cpu().numpy()

        for idx in dones.nonzero(as_tuple=False).flatten().tolist():
            infos[idx]["terminal_observation"] = terminal_obs[idx].copy()
            infos[idx]["TimeLimit.truncated"] = bool(truncated_np[idx] and not terminated_np[idx])
            infos[idx]["is_success"] = bool(task_success[idx].item())
            infos[idx]["rings_complete"] = bool(rings_complete[idx].item())
            infos[idx]["terminal_distance"] = float(rew_distance_raw[idx].item())
            infos[idx]["green_peg_ring_count"] = int(green_peg_ring_count[idx].item())

        # --------------------------------------------------
        # Manual reset for non-Isaac terminal conditions
        # Example: logical task success, handled outside IsaacLab terminations.
        # Isaac-done envs may already be handled internally by IsaacLab.
        # --------------------------------------------------
        manual_reset = dones & (~isaac_done)

        if manual_reset.any():
            reset_ids = manual_reset.nonzero(as_tuple=False).flatten()
            self.env.reset(env_ids=reset_ids)

        # --------------------------------------------------
        # Reset wrapper/state-machine state
        # --------------------------------------------------
        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).flatten().tolist()

            for idx in done_ids:
                print(
                    f"[ENV {idx}] Episode done. "
                    f"Distance={rew_distance_raw[idx].item():.4f}. Resetting environment."
                )

                self.sm.reset_env(idx)
                self.current_step[idx] = 0
                self.settle_steps_remaining[idx] = 0

                self.prev_actions[idx] = IDLE_ACTION.to(self.env.device)
                self.last_override_flags[idx] = 0.0

            # Refill embedding stack AFTER reset, not with zeros.
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

        assert obs.shape[1] == self.obs_dim, f"Expected {self.obs_dim}, got {obs.shape[1]}"
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

def compute_average_goal_embedding(tcc_model, preprocess, dataset_path, device):
    embeddings = []
    video_dirs = sorted([os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    print(f"[INFO] Computing mean goal from {len(video_dirs)} videos in {dataset_path}...")
    for v_dir in video_dirs:
        frames = sorted(glob.glob(os.path.join(v_dir, "*.jpg")) + glob.glob(os.path.join(v_dir, "*.png")))
        if not frames: continue
        last_frame_path = frames[-1]
        try:
            print(f"[DEBUG] Processing {last_frame_path} for goal embedding...")
            img = io.read_image(last_frame_path)[:3].unsqueeze(0).float().to(device) / 255.0
            with torch.no_grad():
                emb = tcc_model(preprocess(img)).squeeze()
            embeddings.append(emb)
        except Exception as e:
            print(f"[Warning] Cannot process {last_frame_path}: {e}")
    if not embeddings: raise ValueError("No valid frames found in the dataset to compute the goal!")
    stacked_embs = torch.stack(embeddings)
    mean_goal_embedding = torch.mean(stacked_embs, dim=0)
    print(f"[INFO] Mean goal computed successfully from {len(embeddings)} final frames!")
    return mean_goal_embedding

# ==========================================
# EVALUATION MAIN
# ==========================================
def main_eval():
    env_cfg = MDvrkEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.num_rerenders_on_reset = 2
    env_cfg.wait_for_textures = False
    env_cfg.seed = 42
    env_cfg.events.reset_rings.params["randomize"] = bool(args_cli.randomize_rings)
    print(
        "[INFO] Ring reset mode: "
        f"{'randomized' if args_cli.randomize_rings else 'fixed deterministic'}"
    )
    print(f"[INFO] Simulation device: {env_cfg.sim.device}")
    
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    sm = SPARTANStateMachine(isaac_env)
    tcc = XIRLResnet18(32).to(isaac_env.device)
    
    try:
        resolved_tcc_ckpt = resolve_tcc_checkpoint(args_cli.tcc_checkpoint)
        print(f"[INFO] Loading TCC weights from: {resolved_tcc_ckpt}")
        ckpt = torch.load(resolved_tcc_ckpt, map_location=isaac_env.device)
        if "model" in ckpt:
            sd = ckpt["model"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt

        clean_sd = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("model."):
                k = k[len("model."):]
            clean_sd[k] = v

        load_result = tcc.load_state_dict(clean_sd, strict=True)
        print("[INFO] Weights loaded!")
    except Exception as e: 
        print(f"[ERR] Weights: {e}")

    tcc.eval()
    rl_env = DVRKVisionHRLWrapper(isaac_env, sm, tcc)
    
    # Compute goal embedding on the raw unwrapped environment first
    try:
        proc = T.Compose([T.Resize((112, 112), antialias=True)])
        rl_env.goal_embedding = compute_average_goal_embedding(tcc, proc, args_cli.dataset_path, isaac_env.device)
        print("[INFO] Goal ready (computed as dataset mean)!")
    except Exception as e: 
        print(f"[ERR] Error computing mean goal from dataset: {e}")
        rl_env.goal_embedding = torch.zeros(32, device=isaac_env.device)

    # Try to find the matching VecNormalize file and wrap the environment
    checkpoint_path = args_cli.checkpoint
    base_no_ext, _ = os.path.splitext(checkpoint_path)
    
    vec_normalize_path = None
    possible_paths = [
        f"{base_no_ext}_vecnormalize.pkl",
    ]
    # Check for step checkpoints (e.g. dvrk_ppo_1234_steps.zip)
    dir_name = os.path.dirname(checkpoint_path)
    base_name = os.path.basename(checkpoint_path)
    if base_name.startswith("dvrk_ppo_") and "_steps" in base_name:
        steps_part = base_name.replace("dvrk_ppo_", "")
        possible_paths.append(os.path.join(dir_name, f"dvrk_ppo_vecnormalize_{steps_part.replace('.zip', '.pkl')}"))
    
    possible_paths.extend([
        os.path.join(os.path.dirname(dir_name), "sb3_log_sim/vecnormalize.pkl"),
            "/home/aiprah/Documents/m_dVrk/random_sb3_log_sim_phase_1/vecnormalize.pkl",
        "/home/aiprah/Documents/m_dVrk/sb3_log_sim/vecnormalize.pkl"
    ])
    
    for path in possible_paths:
        if os.path.exists(path):
            vec_normalize_path = path
            break
            
    if vec_normalize_path is not None:
        print(f"[INFO] Loading VecNormalize statistics from: {vec_normalize_path}")
        rl_env = VecNormalize.load(vec_normalize_path, rl_env)
        rl_env.training = False
        rl_env.norm_reward = False
    else:
        print("[WARNING] No VecNormalize statistics found! Observations will not be normalized.")

    print(f"[INFO] Loading model from: {args_cli.checkpoint}")
    model = PPO.load(args_cli.checkpoint, env=rl_env, device="cuda")
    camera_sensor = rl_env.unwrapped.env.scene["camera"]
    video_frames = []
    print("[INFO] Starting evaluation...")
    obs = rl_env.reset()
    
    steps = 0
    while simulation_app.is_running():
        # The RL agent computes actions deterministically
        action, _states = model.predict(obs, deterministic=True)
        # Executes the step in the environment
        obs, rewards, dones, infos = rl_env.step(action)
        
        # Get active triplets for env 0
        cmd_l = rl_env.sm.current_triplet_l[0]
        cmd_r = rl_env.sm.current_triplet_r[0]
        str_l = f"{cmd_l['verb']} -> {cmd_l['target']}" if cmd_l else "idle -> None"
        str_r = f"{cmd_r['verb']} -> {cmd_r['target']}" if cmd_r else "idle -> None"
        print(f"[Step {steps:03d}] Left Arm: {str_l:<20} | Right Arm: {str_r:<20}")

        # Retrieve virtual camera frame (H, W, 4) or (H, W, 3)
        rgb_data = camera_sensor.data.output["rgb"][0]
        
        # Convert to a writable 3-channel numpy array on CPU for drawing
        frame_np = rgb_data.cpu().numpy()[:, :, :3].copy()
        
        # Draw a semi-transparent HUD background banner at the top
        overlay = frame_np.copy()
        cv2.rectangle(overlay, (0, 0), (frame_np.shape[1], 75), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame_np, 0.6, 0, frame_np)
        
        # Draw Left Arm (Green) and Right Arm (Cyan) triplets
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_np, f"LEFT ARM:  {str_l.upper()}", (15, 30), font, 0.6, (50, 255, 50), 2, cv2.LINE_AA)
        cv2.putText(frame_np, f"RIGHT ARM: {str_r.upper()}", (15, 55), font, 0.6, (50, 200, 255), 2, cv2.LINE_AA)
        
        # Draw Step Counter (White) and Reward (Yellow-Gold)
        cv2.putText(frame_np, f"STEP: {steps:03d}", (frame_np.shape[1] - 150, 30), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_np, f"REWARD: {rewards[0]:.4f}", (frame_np.shape[1] - 200, 55), font, 0.6, (100, 230, 255), 2, cv2.LINE_AA)
        
        # Convert back to PyTorch CPU tensor and append
        frame = torch.from_numpy(frame_np)
        video_frames.append(frame)

        if steps >= 390 and video_frames:
            video_tensor = torch.stack(video_frames) # (T, H, W, C)
            io.write_video("pov_operazione.mp4", video_tensor, fps=30)
            print("Video saved: pov_operazione.mp4")
            steps = 0
            video_frames = []
        steps += 1

if __name__ == "__main__":
    main_eval()
    simulation_app.close()
