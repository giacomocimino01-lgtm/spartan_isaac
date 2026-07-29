"""Script VETTORIZZATO dVRK: RL Multi-Ambiente + XIRL Batched + SPARTAN Hive-Mind."""
import argparse
from collections import deque
import csv
import os
import glob
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import math
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torchvision.io as io
torch.cuda.empty_cache()

from app_launcher_utils import pin_process_to_requested_cuda_device, resolve_tcc_checkpoint
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure

# 1. SETUP INIZIALE DI ISAAC SIM
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Macchina a stati per il Peg and Ring con RL.")
parser.add_argument("--num_envs", type=int, default=64, help="Numero di ambienti da spawnare in parallelo.")
parser.add_argument(
    "--randomize_rings",
    action="store_true",
    help="Randomize ring reset positions. By default, rings reset to a fixed deterministic layout.",
)
parser.add_argument(
    "--pretrained_checkpoint",
    type=str,
    default=None,
    help="Path to a zip file of a pre-trained PPO policy (Behavior Cloning).",
)
parser.add_argument(
    "--freeze_policy_timesteps",
    type=int,
    default=1500000,
    help="Number of timesteps to freeze the policy (actor) weights at the beginning of training (to train the critic first).",
)
parser.add_argument(
    "--task_phase",
    type=str,
    default="phase_0",
    choices=["phase_0", "phase_1"],
    help="Goal phase of the task. 'phase_0' places 4 rings on the green peg, 'phase_1' places 2 on red and 2 on blue (starting stacked on green).",
)
parser.add_argument(
    "--goal_dataset_root",
    type=str,
    default="/mnt/data/aiprah/data/sim_dataset_xirl_extra",
    help="Root directory of the dataset used to compute the goal embedding.",
)
parser.add_argument(
    "--disable_obs_normalization",
    action="store_true",
    help="Disable VecNormalize observation normalization. Useful for raw-observation BC checkpoint checks.",
)
parser.add_argument(
    "--total_timesteps",
    type=int,
    default=5_000_000,
    help="Total PPO training timesteps. Use a small value for ablations.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=3e-4,
    help="PPO learning rate. Lower values are useful when fine-tuning a BC-pretrained actor.",
)
parser.add_argument(
    "--ent_coef",
    type=float,
    default=0.015,
    help="PPO entropy coefficient. Lower values reduce exploration pressure after BC pretraining.",
)
parser.add_argument(
    "--log_suffix",
    type=str,
    default="",
    help="Optional suffix appended to the PPO log directory and checkpoint names.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
requested_device = getattr(args_cli, "device", None)
args_cli.device = pin_process_to_requested_cuda_device(requested_device)
if requested_device != args_cli.device and requested_device is not None and "cuda" in requested_device:
    print(
        f"[INFO] Remapped device {requested_device} -> {args_cli.device} "
        f"with CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

# The camera sensor requires Isaac rendering to be enabled.
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg 

import torch.nn.functional as F
from parallel_env import SPARTANStateMachine, get_scene_entity_positions_w, sync_attached_and_frozen_rings

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

REWARD_DEBUG_DUMP_INTERVAL = 500
REWARD_DEBUG_DUMP_ENV_IDS = (0,)
REWARD_DEBUG_DUMP_DIR = "reward_debug_frames"
REWARD_DEBUG_DUMP_MAX_PER_ENV = 200
PPO_TARGET_TRANSITIONS_PER_UPDATE = 8192
PPO_PREFERRED_BATCH_SIZE = 2048
PPO_N_EPOCHS = 5
PPO_LEARNING_RATE = 3e-4
PPO_ENT_COEF = 0.015
EPISODE_CSV_FILENAME = "episode_progress.csv"
INVALID_COMMAND_PENALTY = 1.0
BEST_TERMINAL_DISTANCE_WINDOW_EPISODES = 10
BEST_TERMINAL_DISTANCE_PREFIX = "dvrk_ppo_best_terminal_distance"

# --- Success snapshot / video dump ---
SUCCESS_SNAPSHOT_PROB    = 0.10   # 10%: salva uno screenshot PNG ad ogni successo
SUCCESS_VIDEO_PROB       = 0.02   # 2%:  salva un video MP4 dell'episodio di successo
SUCCESS_DUMP_DIR         = "success_snapshots"
SUCCESS_VIDEO_DIR        = "success_videos"
SUCCESS_VIDEO_FRAME_SKIP = 5      # Accumula 1 frame ogni N step nel buffer video

# ==========================================
# CLASSE XIRL: CLONE DELLA RETE PRE-TRAINATA
# ==========================================
class XIRLResnet18(nn.Module):
    """
    Matches XIRL config:

        model_type = "resnet18_linear"
        embedding_size = 32
        normalize_embeddings = False
        learnable_temp = False

    Compatible with checkpoints containing:
        backbone.*
        encoder.*
    """

    def __init__(self, embedding_size=32):
        super().__init__()

        resnet = models.resnet18(weights=None)
        num_ftrs = resnet.fc.in_features

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.encoder = nn.Linear(num_ftrs, embedding_size)

    def forward(self, x):
        # x: (B, C, H, W)
        feats = self.backbone(x)          # (B, 512, 1, 1)
        feats = torch.flatten(feats, 1)   # (B, 512)
        embs = self.encoder(feats)        # (B, 32)

        # Important: config.model.normalize_embeddings = False
        return embs

# ==========================================
# CLASSE 1.5: IL WRAPPER RL VETTORIZZATO (Eredita da VecEnv!)
# ==========================================
class DVRKVisionHRLWrapper(VecEnv):
    def __init__(self, isaac_env, state_machine, tcc_model=None, task_phase="phase_0"):
        self.env = isaac_env
        self.sm = state_machine
        self.tcc_model = tcc_model
        self.task_phase = task_phase
        
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
                f"[RewardDebug] Dump attivo: every={self.reward_debug_dump_interval} step, "
                f"envs={self.reward_debug_dump_env_ids}, dir={self.reward_debug_dump_dir}"
            )

        # --- Success snapshot / video dump ---
        self.success_snapshot_prob    = SUCCESS_SNAPSHOT_PROB
        self.success_video_prob       = SUCCESS_VIDEO_PROB
        self.success_dump_dir         = SUCCESS_DUMP_DIR
        self.success_video_dir        = SUCCESS_VIDEO_DIR
        self.success_video_frame_skip = SUCCESS_VIDEO_FRAME_SKIP
        # Frame buffer per ogni env: lista di tensor (H, W, 3) uint8 su CPU
        self._success_frame_buffers   = [[] for _ in range(self.num_envs)]
        self._ep_step_for_video       = [0] * self.num_envs   # contatore step locale per il frame-skip
        os.makedirs(self.success_dump_dir, exist_ok=True)
        os.makedirs(self.success_video_dir, exist_ok=True)
        print(
            f"[SuccessDump] snapshot_prob={self.success_snapshot_prob:.0%}, "
            f"video_prob={self.success_video_prob:.0%}, "
            f"frame_skip={self.success_video_frame_skip}"
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
                print(f"[RewardDebug] Ignoro env id non valido: {raw_env_id}")
                continue

            if 0 <= env_id < self.num_envs:
                env_ids.append(env_id)
            else:
                print(f"[RewardDebug] Ignoro env id fuori range: {env_id}")

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

    def _get_manipulable_rings_mask(self) -> torch.Tensor:
        """Restituisce una maschera (num_envs, 4) con 1.0 se l'anello è manipolabile, 0.0 altrimenti."""
        device = self.env.device
        manipulable_mask = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=device)

        for env_i in range(self.num_envs):
            for ring_idx, ring_name in enumerate(self._ring_names):
                # 1. Se è già impugnato dal braccio sinistro o destro, non occorre fare re-grasp
                if (self.sm.attached_target_l[env_i] == ring_name or
                    self.sm.attached_target_r[env_i] == ring_name):
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

        assert aux.shape[1] == self.aux_dim, f"Expected {self.aux_dim}, got {aux.shape[1]}"
        return aux
    
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

        manipulable_mask = self._get_manipulable_rings_mask()

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

    def _peg_ring_count(self, peg_name):
        counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.env.device)
        for ring_name in ["ring_red", "ring_yellow", "ring_green", "ring_blue"]:
            on_peg = torch.tensor(
                [self.sm.ring_support_peg[ring_name][i] == peg_name for i in range(self.num_envs)],
                dtype=torch.bool,
                device=self.env.device,
            )
            counts += (self.sm.frozen_rings_mask[ring_name] & on_peg).long()
        return counts

    def _green_peg_ring_count(self):
        return self._peg_ring_count("peg_green")

    def _task_success(self):
        if self.task_phase == "phase_1":
            red_count = self._peg_ring_count("peg_red")
            blue_count = self._peg_ring_count("peg_blue")
            all_frozen = torch.stack(
                [self.sm.frozen_rings_mask[name] for name in ["ring_red", "ring_yellow", "ring_green", "ring_blue"]],
                dim=1
            ).all(dim=1)
            return all_frozen & (red_count == 2) & (blue_count == 2)
        else:
            return self._peg_ring_count("peg_green") == 4

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
            self.sm.reset_env(i, phase=self.task_phase)

        # Sync attached/frozen rings so they show up at their correct initial stacked/frozen poses
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

    def _sanitize_arm_command(self, env_id: int, verb: str, target: str, manipulable_mask: torch.Tensor):
        if verb == "idle":
            return "idle", "None", False

        if target == "None":
            return "idle", "None", True

        if verb == "grasp":
            if target not in RING_TARGETS:
                return "idle", "None", True

            current_peg = self.sm.ring_support_peg.get(target, [None] * self.num_envs)[env_id]
            if current_peg is not None and current_peg != "None":
                top_rings = self.sm.get_top_rings_on_peg(env_id, current_peg, num_rings=1)
                top_ring = top_rings[0] if top_rings else None
                if top_ring != target:
                    return "idle", "None", True

        if verb == "release":
            return verb, target, False

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
        manipulable_masks = self._get_manipulable_rings_mask()

        for i in range(self.num_envs):
            was_busy_l = self.sm.sub_state_l[i] != "IDLE"
            was_busy_r = self.sm.sub_state_r[i] != "IDLE"

            old_cmd_l = self.sm.current_triplet_l[i]
            old_cmd_r = self.sm.current_triplet_r[i]

            verb_l, tgt_l = VERB_MAP[int(actions[i, 0])], TARGET_MAP[int(actions[i, 1])]
            verb_r, tgt_r = VERB_MAP[int(actions[i, 2])], TARGET_MAP[int(actions[i, 3])]

            verb_l, tgt_l, invalid_l = self._sanitize_arm_command(i, verb_l, tgt_l, manipulable_masks[i])
            verb_r, tgt_r, invalid_r = self._sanitize_arm_command(i, verb_r, tgt_r, manipulable_masks[i])

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

        # Accumula frame nel buffer video (su CPU, campionamento ogni N step)
        rgb_data = self.env.scene.sensors["camera"].data.output["rgb"]
        self._accumulate_success_frames(rgb_data)
        
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
        
        # --- RIBILANCIAMENTO REWARD ---
        rew_distance_raw = torch.norm(new_embs - self.goal_embedding, dim=1) ** 2
        rewards = -rew_distance_raw * 1e-3
        rewards -= invalid_command_counts * INVALID_COMMAND_PENALTY

        green_peg_ring_count = self._green_peg_ring_count()
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
        # ------------------------------

        # Snapshot / video dei successi (prima del reset)
        if task_success.any():
            self._maybe_dump_success_assets(task_success, rgb_data)

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
            if self.task_phase == "phase_1":
                infos[idx]["red_peg_ring_count"] = int(self._peg_ring_count("peg_red")[idx].item())
                infos[idx]["blue_peg_ring_count"] = int(self._peg_ring_count("peg_blue")[idx].item())

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

                self.sm.reset_env(idx, phase=self.task_phase)
                self.current_step[idx] = 0
                self.settle_steps_remaining[idx] = 0

                self.prev_actions[idx] = IDLE_ACTION.to(self.env.device)
                self.last_override_flags[idx] = 0.0

                # Pulisci buffer frame dell'episodio appena concluso
                self._success_frame_buffers[idx].clear()
                self._ep_step_for_video[idx] = 0

            # Sync newly reset environments
            sync_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.device)
            for idx in done_ids:
                sync_mask[idx] = True
            sync_attached_and_frozen_rings(self.env, self.sm, sync_mask)

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

    # ------------------------------------------------------------------
    # Success snapshot / video helpers
    # ------------------------------------------------------------------

    def _accumulate_success_frames(self, rgb_data: torch.Tensor):
        """Accumula i frame RGB nel buffer video di ogni env (campionamento a frame_skip)."""
        for idx in range(self.num_envs):
            self._ep_step_for_video[idx] += 1
            if self._ep_step_for_video[idx] % self.success_video_frame_skip != 0:
                continue
            # Estrai frame (H, W, 3) uint8 su CPU — .clone() è ESSENZIALE:
            # camera.data.output["rgb"] è un buffer aggiornato in-place da IsaacLab;
            # senza clone() tutti i frame del buffer punterebbero alla stessa memoria.
            frame = rgb_data[idx, :, :, :3].clone().cpu().to(torch.uint8)
            self._success_frame_buffers[idx].append(frame)

    def _maybe_dump_success_assets(
        self,
        task_success: torch.Tensor,
        rgb_data: torch.Tensor,
    ):
        """Per ogni env con successo, salva probabilisticamente PNG e/o MP4."""
        for idx in task_success.nonzero(as_tuple=False).flatten().tolist():
            ts = int(self.current_step[idx].item())
            tag = f"env{idx:03d}_step{ts:06d}"

            # --- Screenshot ---
            if random.random() < self.success_snapshot_prob:
                try:
                    # .clone() per non dipendere dal buffer in-place della camera
                    frame = rgb_data[idx, :, :, :3].clone().cpu().to(torch.uint8).permute(2, 0, 1)  # (3, H, W)
                    png_path = os.path.join(self.success_dump_dir, f"success_{tag}.png")
                    io.write_png(frame, png_path)
                    print(f"[SuccessDump] Screenshot salvato: {png_path}", flush=True)
                except Exception as exc:
                    print(f"[SuccessDump] Errore screenshot env {idx}: {exc}", flush=True)

            # --- Video ---
            if random.random() < self.success_video_prob:
                frames = self._success_frame_buffers[idx]
                if len(frames) >= 2:
                    try:
                        # (T, H, W, 3) uint8
                        video_tensor = torch.stack(frames, dim=0)
                        mp4_path = os.path.join(self.success_video_dir, f"success_{tag}.mp4")
                        # fps effettivo ≈ sim_fps / decimation / frame_skip
                        io.write_video(mp4_path, video_tensor, fps=10)
                        print(f"[SuccessDump] Video salvato: {mp4_path} ({len(frames)} frames)", flush=True)
                    except Exception as exc:
                        print(f"[SuccessDump] Errore video env {idx}: {exc}", flush=True)

def compute_average_goal_embedding(tcc_model, preprocess, dataset_path, device):
    embeddings = []
    video_dirs = sorted([os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    print(f"[INFO] Calcolo Goal medio da {len(video_dirs)} video in {dataset_path}...")
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
            print(f"[Warning] Impossibile processare {last_frame_path}: {e}")
    if not embeddings: raise ValueError("Nessun frame valido trovato nel dataset per il calcolo del Goal!")
    stacked_embs = torch.stack(embeddings)
    mean_goal_embedding = torch.mean(stacked_embs, dim=0)
    print(f"[INFO] Goal medio calcolato con successo da {len(embeddings)} frame finali!")
    return mean_goal_embedding

def set_seed(seed=42):
    # 1. Python random (per librerie base)
    random.seed(seed)
    
    # 2. Numpy (usato internamente da Gym/SB3)
    np.random.seed(seed)
    
    # 3. PyTorch (inizializzazione pesi PPO)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Se usi multi-GPU
    
    # Rende le operazioni convoluzionali e GPU deterministiche (sacrifica un filo di velocità per la riproducibilità)
    cudnn.deterministic = True
    cudnn.benchmark = False

# Applica il seed
set_seed(42)


def derive_ppo_n_steps(num_envs: int, target_transitions_per_update: int) -> int:
    safe_num_envs = max(1, num_envs)
    safe_target = max(1, target_transitions_per_update)
    return max(1, math.ceil(safe_target / safe_num_envs))


def derive_ppo_batch_size(effective_batch_size: int, preferred_batch_size: int) -> int:
    safe_effective_batch_size = max(1, int(effective_batch_size))
    start = min(max(1, int(preferred_batch_size)), safe_effective_batch_size)
    for batch_size in range(start, 0, -1):
        if safe_effective_batch_size % batch_size == 0:
            return batch_size
    return 1


class EpisodeCsvLoggerCallback(BaseCallback):
    def __init__(self, file_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.file_path = file_path
        self._file = None
        self._writer = None

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self._file = open(self.file_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "total_timesteps",
                "env_id",
                "episode_reward",
                "episode_length",
                "episode_time_seconds",
                "is_success",
                "rings_complete",
                "terminal_distance",
                "green_peg_ring_count",
                "red_peg_ring_count",
                "blue_peg_ring_count",
                "time_limit_truncated",
            ],
        )
        self._writer.writeheader()
        self._file.flush()
        print(f"[PPO] Episode CSV logger active: {self.file_path}", flush=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None or self._writer is None:
            return True

        wrote_rows = False
        for env_id, info in enumerate(infos):
            episode = info.get("episode")
            if episode is None:
                continue

            self._writer.writerow(
                {
                    "total_timesteps": int(self.num_timesteps),
                    "env_id": env_id,
                    "episode_reward": float(episode.get("r", 0.0)),
                    "episode_length": int(episode.get("l", 0)),
                    "episode_time_seconds": float(episode.get("t", 0.0)),
                    "is_success": bool(info.get("is_success", False)),
                    "rings_complete": bool(info.get("rings_complete", False)),
                    "terminal_distance": float(info.get("terminal_distance", float("nan"))),
                    "green_peg_ring_count": int(info.get("green_peg_ring_count", 0)),
                    "red_peg_ring_count": int(info.get("red_peg_ring_count", 0)),
                    "blue_peg_ring_count": int(info.get("blue_peg_ring_count", 0)),
                    "time_limit_truncated": bool(info.get("TimeLimit.truncated", False)),
                }
            )
            wrote_rows = True

        if wrote_rows and self._file is not None:
            self._file.flush()

        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


class BestTerminalDistanceCheckpointCallback(BaseCallback):
    def __init__(self, save_dir: str, file_prefix: str, window_size: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.file_prefix = file_prefix
        self.window_size = max(1, int(window_size))
        self.best_mean_terminal_distance = float("inf")
        self.recent_terminal_distances = deque(maxlen=self.window_size)

    def _on_training_start(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None:
            return True

        for info in infos:
            episode = info.get("episode")
            terminal_distance = info.get("terminal_distance")
            if episode is None or terminal_distance is None:
                continue

            terminal_distance = float(terminal_distance)
            if not math.isfinite(terminal_distance):
                continue

            self.recent_terminal_distances.append(terminal_distance)

        if len(self.recent_terminal_distances) < self.window_size:
            return True

        mean_terminal_distance = sum(self.recent_terminal_distances) / len(self.recent_terminal_distances)
        if mean_terminal_distance >= self.best_mean_terminal_distance:
            return True

        self.best_mean_terminal_distance = mean_terminal_distance

        model_path = os.path.join(self.save_dir, f"{self.file_prefix}.zip")
        vec_normalize_path = os.path.join(
            self.save_dir,
            f"{self.file_prefix}_vecnormalize.pkl",
        )
        metric_path = os.path.join(
            self.save_dir,
            f"{self.file_prefix}_metrics.txt",
        )

        self.model.save(model_path)

        vec_normalize_env = self.model.get_vec_normalize_env()
        if vec_normalize_env is not None:
            vec_normalize_env.save(vec_normalize_path)

        with open(metric_path, "w", encoding="ascii") as metric_file:
            metric_file.write(f"num_timesteps={int(self.num_timesteps)}\n")
            metric_file.write(f"window_size={len(self.recent_terminal_distances)}\n")
            metric_file.write(f"mean_terminal_distance={mean_terminal_distance:.8f}\n")

        if self.verbose > 0:
            print(
                "[Checkpoint] New best terminal distance: "
                f"mean={mean_terminal_distance:.6f} over last {len(self.recent_terminal_distances)} episodes"
            )

        return True

class FreezePolicyCallback(BaseCallback):
    def __init__(self, freeze_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.freeze_timesteps = freeze_timesteps
        self.policy_frozen = False

    def _on_training_start(self) -> None:
        if self.freeze_timesteps > 0:
            self.freeze_policy()

    def _on_step(self) -> bool:
        if self.policy_frozen and self.num_timesteps >= self.freeze_timesteps:
            self.unfreeze_policy()
        return True

    def freeze_policy(self) -> None:
        print(f"[PPO] Freezing policy (actor) parameters for the first {self.freeze_timesteps} timesteps.", flush=True)
        freeze_count = 0
        freeze_params_names = []
        for name, param in self.model.policy.named_parameters():
            if "policy_net" in name or "action_net" in name:
                param.requires_grad = False
                param.grad = None
                freeze_count += 1
                freeze_params_names.append(name)
        print(f"[PPO] Froze {freeze_count} parameters:", flush=True)
        for pname in freeze_params_names:
            print(f"      - {pname}", flush=True)
        self.policy_frozen = True

    def unfreeze_policy(self) -> None:
        print(f"[PPO] Unfreezing policy (actor) parameters after {self.num_timesteps} timesteps.", flush=True)
        unfreeze_count = 0
        for name, param in self.model.policy.named_parameters():
            if "policy_net" in name or "action_net" in name:
                param.requires_grad = True
                unfreeze_count += 1
        print(f"[PPO] Unfroze {unfreeze_count} parameters.", flush=True)
        self.policy_frozen = False


# ==========================================
# TRAINING MAIN
# ==========================================
def main_train():
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

    # DEBUG: inspect scene entities
    print("\n[DEBUG] Scene entity types:")
    for name in [
        "ring_red", "ring_yellow", "ring_green", "ring_blue",
        "peg_red", "peg_yellow", "peg_green", "peg_blue",
        "peg_gray", "peg_gray1",
    ]:
        entity = isaac_env.scene[name]
        print(
            f"{name:12s} | "
            f"type={type(entity)} | "
            f"has_data={hasattr(entity, 'data')} | "
            f"has_root_pos_w={hasattr(entity, 'data') and hasattr(entity.data, 'root_pos_w')} | "
            f"has_get_world_poses={hasattr(entity, 'get_world_poses')}"
        )

    print("[DEBUG] env_origins:")
    print(isaac_env.scene.env_origins[:min(5, isaac_env.num_envs)])
    print()


    sm = SPARTANStateMachine(isaac_env)
    tcc = XIRLResnet18(embedding_size=32).to(isaac_env.device)

    try:
        experiment_dir = f"/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_{args_cli.task_phase.replace('_', '')}_tcc"
        tcc_ckpt_path = resolve_tcc_checkpoint(experiment_dir)
        print(f"[INFO] Loading XIRL weights from: {tcc_ckpt_path}")
        ckpt = torch.load(
            tcc_ckpt_path,
            map_location=isaac_env.device,
        )

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

        print("[INFO] XIRL weights loaded.")
        print("[XIRL] missing keys:", load_result.missing_keys)
        print("[XIRL] unexpected keys:", load_result.unexpected_keys)

    except Exception as e:
        print(f"[ERR] XIRL weights: {e}")
        raise RuntimeError("XIRL checkpoint failed to load. Aborting training.") from e

    tcc.eval()

    rl_env = DVRKVisionHRLWrapper(isaac_env, sm, tcc, task_phase=args_cli.task_phase)
    
    try:
        dataset_path = f"{args_cli.goal_dataset_root}/train/{args_cli.task_phase}/"
        print(f"[INFO] Computing goal embedding from dataset path: {dataset_path}")
        proc = T.Compose([T.Resize((112, 112), antialias=True)])
        rl_env.goal_embedding = compute_average_goal_embedding(tcc, proc, dataset_path, isaac_env.device)
    except Exception as e:
        raise RuntimeError("Failed to compute goal embedding. Aborting training.") from e

    rl_env = VecMonitor(rl_env)
    norm_obs = not bool(args_cli.disable_obs_normalization)
    if args_cli.pretrained_checkpoint is not None and norm_obs:
        print(
            "[PPO] Warning: VecNormalize(norm_obs=True) is active with a BC checkpoint. "
            "pretrain_ppo.py trains on raw observations; use --disable_obs_normalization "
            "for the raw-BC freeze ablation.",
            flush=True,
        )
    print(f"[PPO] VecNormalize: norm_obs={norm_obs}, norm_reward=True", flush=True)
    rl_env = VecNormalize(
        rl_env,
        norm_obs=norm_obs,
        norm_reward=True,
        clip_obs=np.inf,
        clip_reward=np.inf,
        training=True,
    )

    log_suffix = args_cli.log_suffix.strip()
    log_suffix = f"_{log_suffix}" if log_suffix else ""
    tmp_path = f"/home/aiprah/Documents/m_dVrk/random_sb3_log_sim_{args_cli.task_phase}{log_suffix}/"
    episode_csv_path = os.path.join(tmp_path, EPISODE_CSV_FILENAME)
    vec_normalize_path = os.path.join(tmp_path, "vecnormalize.pkl")
    new_logger = configure(tmp_path, ["stdout","csv", "tensorboard"])
    checkpoint_callback = CheckpointCallback(
        6000,
        "/home/aiprah/Documents/m_dVrk/modelli_salvati_sim",
        f"dvrk_ppo_{args_cli.task_phase}{log_suffix}",
        save_vecnormalize=True,
    )
    
    n_steps = derive_ppo_n_steps(args_cli.num_envs, PPO_TARGET_TRANSITIONS_PER_UPDATE)
    effective_batch_size = n_steps * args_cli.num_envs
    batch_size = derive_ppo_batch_size(effective_batch_size, PPO_PREFERRED_BATCH_SIZE)
    print(
        f"[PPO] num_envs={args_cli.num_envs} | n_steps={n_steps} | "
        f"transitions_per_update={effective_batch_size} | batch_size={batch_size} | "
        f"n_epochs={PPO_N_EPOCHS}"
    , flush=True)
    print(
        f"[PPO] learning_rate={args_cli.learning_rate:g} | ent_coef={args_cli.ent_coef:g}",
        flush=True,
    )
    
    episode_csv_callback = EpisodeCsvLoggerCallback(episode_csv_path)
    best_terminal_distance_callback = BestTerminalDistanceCheckpointCallback(
        save_dir="/home/aiprah/Documents/m_dVrk/modelli_salvati_sim",
        file_prefix=f"{BEST_TERMINAL_DISTANCE_PREFIX}_{args_cli.task_phase}{log_suffix}",
        window_size=BEST_TERMINAL_DISTANCE_WINDOW_EPISODES,
        verbose=1,
    )
    callbacks_list = [
        checkpoint_callback,
        episode_csv_callback,
        best_terminal_distance_callback,
    ]

    freeze_timesteps = args_cli.freeze_policy_timesteps
    if freeze_timesteps > 0:
        if args_cli.pretrained_checkpoint is not None:
            print(f"[PPO] Pre-trained checkpoint loaded. Policy will be frozen for the first {freeze_timesteps} timesteps.", flush=True)
            callbacks_list.append(FreezePolicyCallback(freeze_timesteps=freeze_timesteps))
        else:
            print(f"[PPO] Warning: Policy freezing requested ({freeze_timesteps} timesteps), but no pre-trained checkpoint was provided. Disabling freeze.", flush=True)

    callback = CallbackList(callbacks_list)

    policy_kwargs = dict(
    net_arch=dict(
        pi=[64, 64],
        vf=[64, 64],
    ),
    activation_fn=nn.ReLU,
    )

    if args_cli.pretrained_checkpoint is not None:
        print(f"[PPO] Loading pre-trained policy from {args_cli.pretrained_checkpoint}...", flush=True)
        custom_objects = {
            "learning_rate": args_cli.learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": PPO_N_EPOCHS,
            "ent_coef": args_cli.ent_coef,
        }
        model = PPO.load(
            args_cli.pretrained_checkpoint,
            env=rl_env,
            custom_objects=custom_objects,
            device="cuda",
            tensorboard_log="/home/aiprah/Documents/m_dVrk/tensorboard_logs_sim",
        )
    else:
        print("[PPO] Initializing model with random weights...", flush=True)
        model = PPO(
            "MlpPolicy",
            rl_env,
            verbose=1,
            device="cuda",
            seed=42,
            learning_rate=args_cli.learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=PPO_N_EPOCHS,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=args_cli.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            stats_window_size=1,
            tensorboard_log="/home/aiprah/Documents/m_dVrk/tensorboard_logs_sim",
            policy_kwargs=policy_kwargs,
        )

    model.set_logger(new_logger)
    
    # === DEBUG: Print policy structure ===
    print("\n[DEBUG] === PPO Policy Parameter Structure ===", flush=True)
    total_params = 0
    for name, param in model.policy.named_parameters():
        total_params += param.numel()
        print(f"  {name:60s} | shape: {param.shape} | numel: {param.numel():10d}", flush=True)
    print(f"[DEBUG] Total parameters in policy: {total_params}", flush=True)
    print()
    
    print("[PPO] Model initialized. Starting learn()...", flush=True)
    model.learn(total_timesteps=args_cli.total_timesteps, log_interval=1, callback=callback)
    print("[PPO] learn() completed. Saving final model...", flush=True)
    model.save("/home/aiprah/Documents/m_dVrk/modelli_salvati_sim/dvrk_ppo_finale")

    rl_env.save(vec_normalize_path)

if __name__ == "__main__":
    main_train()
    simulation_app.close()
