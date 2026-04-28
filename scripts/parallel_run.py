"""Script VETTORIZZATO dVRK: RL Multi-Ambiente + XIRL Batched + SPARTAN Hive-Mind."""
import argparse
import threading

# 1. SETUP INIZIALE DI ISAAC SIM
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Macchina a stati per il Peg and Ring con RL.")
parser.add_argument("--num_envs", type=int, default=64, help="Numero di ambienti da spawnare in parallelo.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. IMPORTAZIONI LIBRERIE
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torchvision.io as io

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from isaaclab.envs import ManagerBasedRLEnv
from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg 

VERB_MAP = {0: "reach", 1: "grasp", 2: "release", 3: "idle"}
TARGET_MAP = {
    0: "ring_red", 1: "ring_yellow", 2: "ring_green", 3: "ring_blue",
    4: "peg_red", 5: "peg_yellow", 6: "peg_green", 7: "peg_blue",
    8: "peg_gray", 9: "None"
}

# ==========================================
# CLASSE XIRL: CLONE DELLA RETE PRE-TRAINATA
# ==========================================
class XIRLResnet18(nn.Module):
    def __init__(self, embedding_size=32):
        super().__init__()
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, embedding_size)

    def forward(self, x):
        return self.model(x)

# ==========================================
# CLASSE 1: LA MACCHINA A STATI (Hive-Mind Multi-Env)
# ==========================================
class SPARTANStateMachine:
    Z_TABLE = 0.72              
    SAFE_Z_OFFSET = 0.03        
    GRASP_Z_OFFSET = 0.005      
    TOLERANCE_XY = 0.001        
    TOLERANCE_Z = 0.007         

    def __init__(self, env):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        
        # Inizializziamo liste lunghe "num_envs" per tenere traccia di TUTTI i robot in modo indipendente
        self.current_triplet_r = [None] * self.num_envs
        self.current_triplet_l = [None] * self.num_envs
        self.sub_state_r = ["IDLE"] * self.num_envs
        self.sub_state_l = ["IDLE"] * self.num_envs
        self.step_counter_r = [0] * self.num_envs
        self.step_counter_l = [0] * self.num_envs
        
        self.target_pos_r = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_pos_l = torch.zeros((self.num_envs, 3), device=self.device)
        
        self.current_gripper_state_r = torch.ones(self.num_envs, device=self.device)
        self.current_gripper_state_l = torch.ones(self.num_envs, device=self.device)
        
        self.attached_target_r = [None] * self.num_envs
        self.attached_target_l = [None] * self.num_envs
        self.last_target_r = [None] * self.num_envs
        self.last_target_l = [None] * self.num_envs

    def reset_env(self, env_id):
        """Resetta la mente del singolo robot quando il suo episodio finisce."""
        self.current_triplet_r[env_id] = None
        self.current_triplet_l[env_id] = None
        self.sub_state_r[env_id] = "IDLE"
        self.sub_state_l[env_id] = "IDLE"
        self.step_counter_r[env_id] = 0
        self.step_counter_l[env_id] = 0
        self.attached_target_r[env_id] = None
        self.attached_target_l[env_id] = None
        self.last_target_r[env_id] = None
        self.last_target_l[env_id] = None

    def set_new_triplet(self, verb: str, subject: str, target: str, env_id: int):
        new_cmd = {"verb": verb, "subject": subject, "target": target}
        
        if subject == "right_arm":
            if self.current_triplet_r[env_id] == new_cmd and self.sub_state_r[env_id] != "IDLE": return
            self.current_triplet_r[env_id] = new_cmd
            self.sub_state_r[env_id] = "IDLE" if verb == "idle" else "COMPUTE_TARGET_POS"
            self.step_counter_r[env_id] = 0
        else:
            if self.current_triplet_l[env_id] == new_cmd and self.sub_state_l[env_id] != "IDLE": return
            self.current_triplet_l[env_id] = new_cmd
            self.sub_state_l[env_id] = "IDLE" if verb == "idle" else "COMPUTE_TARGET_POS"
            self.step_counter_l[env_id] = 0

    def get_target_coordinates(self, target_name: str, subject: str, env_id: int):
        if target_name == "None" or target_name is None:
            return torch.tensor([0.375, -0.05, 0.85], device=self.device)

        segno = 1.0 if subject == "right_arm" else -1.0
        offset = torch.tensor([-0.005 * segno, 0.0, 0.0], device=self.device)

        if target_name in self.env.scene.keys():
            entity = self.env.scene[target_name]
            if hasattr(entity, "data") and entity.data.root_pos_w.shape[0] == self.num_envs:
                # RigidObjectCfg: tensor batched (num_envs, 3) -> indice diretto
                return entity.data.root_pos_w[env_id].clone() + offset
            elif hasattr(entity, "get_world_poses"):
                # AssetBaseCfg con spawn=None (Xform tracker, es. peg_*): non è batched.
                # La posizione è in world frame dell'env_0. Per altri env, sommiamo l'offset dell'env.
                pos, _ = entity.get_world_poses()
                local_pos = pos[0].clone()  # posizione locale (identica per tutti gli env)
                env_origin = self.env.scene.env_origins[env_id]  # traslazione dell'env
                # Sottrai l'origine dell'env 0 e aggiungi quella dell'env_id
                env_origin_0 = self.env.scene.env_origins[0]
                return local_pos - env_origin_0 + env_origin + offset
        return torch.tensor([0.5, 0.0, 0.5], device=self.device)
        
    def get_action(self):
        """Genera in parallelo le azioni fisiche per tutti i 64 robot."""
        act_r = torch.zeros((self.num_envs, 6), device=self.device)
        act_l = torch.zeros((self.num_envs, 6), device=self.device)
        grip_r = self.current_gripper_state_r.unsqueeze(1).repeat(1, 2)
        grip_l = self.current_gripper_state_l.unsqueeze(1).repeat(1, 2)

        robot_r = self.env.scene["robot_right"]
        robot_l = self.env.scene["robot_left"]
        tip_r = robot_r.data.body_pos_w[:, robot_r.find_bodies("psm_tool_tip_link")[0][0]]
        tip_l = robot_l.data.body_pos_w[:, robot_l.find_bodies("psm_tool_tip_link")[0][0]]

        # Ciclo ultra-veloce sugli ambienti (In Python prende < 1ms)
        for i in range(self.num_envs):
            # --- BRACCIO DESTRO ---
            if self.sub_state_r[i] != "IDLE" and self.current_triplet_r[i]:
                verb = self.current_triplet_r[i]["verb"]
                if verb == "reach":
                    if self.sub_state_r[i] == "COMPUTE_TARGET_POS":
                        self.sub_state_r[i] = "APPROACH_ABOVE"
                        self.step_counter_r[i] = 0
                    elif self.sub_state_r[i] == "APPROACH_ABOVE":
                        self.target_pos_r[i] = self.get_target_coordinates(self.current_triplet_r[i]["target"], "right_arm", i)
                        dest = self.target_pos_r[i] + torch.tensor([0.0, 0.0, self.SAFE_Z_OFFSET], device=self.device)
                        act_r[i, 0:3] = dest - tip_r[i]
                        if self.step_counter_r[i] > 150: self.sub_state_r[i] = "DESCEND"
                    elif self.sub_state_r[i] == "DESCEND":
                        self.target_pos_r[i] = self.get_target_coordinates(self.current_triplet_r[i]["target"], "right_arm", i)
                        dest = self.target_pos_r[i] + torch.tensor([0.0, 0.0, self.GRASP_Z_OFFSET], device=self.device)
                        act_r[i, 0:3] = dest - tip_r[i]
                        d_xy = torch.norm(dest[0:2] - tip_r[i, 0:2])
                        d_z = torch.abs(dest[2] - tip_r[i, 2])
                        if d_xy < self.TOLERANCE_XY and d_z < self.TOLERANCE_Z:
                            self.sub_state_r[i] = "SETTLE"
                            self.step_counter_r[i] = 0
                    elif self.sub_state_r[i] == "SETTLE":
                        act_r[i, 0:3] = (self.target_pos_r[i] + torch.tensor([0,0,self.GRASP_Z_OFFSET], device=self.device)) - tip_r[i]
                        if self.step_counter_r[i] > 15:
                            self.last_target_r[i] = self.current_triplet_r[i]["target"]
                            self.sub_state_r[i] = "IDLE"

                elif verb == "grasp":
                    if self.sub_state_r[i] == "COMPUTE_TARGET_POS": self.sub_state_r[i] = "CLOSE_GRIPPER"
                    elif self.sub_state_r[i] == "CLOSE_GRIPPER":
                        self.current_gripper_state_r[i] = -1.0
                        grip_r[i] = -1.0
                        if self.step_counter_r[i] > 100:
                            self.attached_target_r[i] = self.current_triplet_r[i]["target"]
                            self.sub_state_r[i] = "LIFT_UP"
                    elif self.sub_state_r[i] == "LIFT_UP":
                        dest = self.target_pos_r[i] + torch.tensor([0.0, 0.0, self.SAFE_Z_OFFSET], device=self.device)
                        act_r[i, 0:3] = dest - tip_r[i]
                        if self.step_counter_r[i] > 250: self.sub_state_r[i] = "IDLE"

                elif verb == "release":
                    if self.sub_state_r[i] == "COMPUTE_TARGET_POS": self.sub_state_r[i] = "APPROACH_PEG"
                    elif self.sub_state_r[i] == "APPROACH_PEG":
                        self.current_gripper_state_r[i] = -1.0
                        grip_r[i] = -1.0
                        tgt = self.current_triplet_r[i]["target"]
                        peg = tgt if tgt.startswith("peg_") else (self.last_target_r[i] if str(self.last_target_r[i]).startswith("peg_") else None)
                        if peg:
                            pos_w = self.get_target_coordinates(peg, "right_arm", i)
                            act_r[i, 0:3] = (pos_w + torch.tensor([0,0,self.SAFE_Z_OFFSET], device=self.device)) - tip_r[i]
                            if self.step_counter_r[i] > 120: self.sub_state_r[i] = "DESCEND_TO_PEG"
                        else:
                            act_r[i, 0:3] = self.target_pos_r[i] - tip_r[i]
                            if self.step_counter_r[i] > 50: self.sub_state_r[i] = "OPEN_GRIPPER"
                    elif self.sub_state_r[i] == "DESCEND_TO_PEG":
                        tgt = self.current_triplet_r[i]["target"]
                        peg = tgt if tgt.startswith("peg_") else self.last_target_r[i]
                        dest = self.get_target_coordinates(peg, "right_arm", i).clone()
                        dest[2] = self.Z_TABLE
                        act_r[i, 0:3] = dest - tip_r[i]
                        if self.step_counter_r[i] > 120: self.sub_state_r[i] = "OPEN_GRIPPER"
                    elif self.sub_state_r[i] == "OPEN_GRIPPER":
                        self.current_gripper_state_r[i] = 1.0
                        grip_r[i] = 1.0
                        if self.step_counter_r[i] == 10:
                            obj = self.attached_target_r[i]
                            if obj and obj != "None":
                                ring = self.env.scene[obj]
                                new_s = ring.data.root_state_w.clone()
                                new_s[i, 2] = self.Z_TABLE
                                ring.write_root_state_to_sim(new_s)
                            self.attached_target_r[i] = None
                        if self.step_counter_r[i] > 80: self.sub_state_r[i] = "IDLE"
                self.step_counter_r[i] += 1
            else:
                # Logica IDLE
                act_r[i, 0:3] = self.target_pos_r[i] - tip_r[i] if torch.any(self.target_pos_r[i] != 0) else 0.0

            # --- LA STESSA IDENTICA LOGICA PER IL BRACCIO SINISTRO (Sintetizzata per spazio) ---
            if self.sub_state_l[i] != "IDLE" and self.current_triplet_l[i]:
                verb = self.current_triplet_l[i]["verb"]
                if verb == "reach":
                    if self.sub_state_l[i] == "COMPUTE_TARGET_POS": self.sub_state_l[i] = "APPROACH_ABOVE"; self.step_counter_l[i] = 0
                    elif self.sub_state_l[i] == "APPROACH_ABOVE":
                        self.target_pos_l[i] = self.get_target_coordinates(self.current_triplet_l[i]["target"], "left_arm", i)
                        act_l[i, 0:3] = (self.target_pos_l[i] + torch.tensor([0,0,self.SAFE_Z_OFFSET], device=self.device)) - tip_l[i]
                        if self.step_counter_l[i] > 150: self.sub_state_l[i] = "DESCEND"
                    elif self.sub_state_l[i] == "DESCEND":
                        self.target_pos_l[i] = self.get_target_coordinates(self.current_triplet_l[i]["target"], "left_arm", i)
                        dest = self.target_pos_l[i] + torch.tensor([0,0,self.GRASP_Z_OFFSET], device=self.device)
                        act_l[i, 0:3] = dest - tip_l[i]
                        if torch.norm(dest[0:2] - tip_l[i, 0:2]) < self.TOLERANCE_XY and torch.abs(dest[2] - tip_l[i, 2]) < self.TOLERANCE_Z:
                            self.sub_state_l[i] = "SETTLE"; self.step_counter_l[i] = 0
                    elif self.sub_state_l[i] == "SETTLE":
                        act_l[i, 0:3] = (self.target_pos_l[i] + torch.tensor([0,0,self.GRASP_Z_OFFSET], device=self.device)) - tip_l[i]
                        if self.step_counter_l[i] > 15: self.last_target_l[i] = self.current_triplet_l[i]["target"]; self.sub_state_l[i] = "IDLE"
                elif verb == "grasp":
                    if self.sub_state_l[i] == "COMPUTE_TARGET_POS": self.sub_state_l[i] = "CLOSE_GRIPPER"
                    elif self.sub_state_l[i] == "CLOSE_GRIPPER":
                        self.current_gripper_state_l[i] = -1.0; grip_l[i] = -1.0
                        if self.step_counter_l[i] > 100: self.attached_target_l[i] = self.current_triplet_l[i]["target"]; self.sub_state_l[i] = "LIFT_UP"
                    elif self.sub_state_l[i] == "LIFT_UP":
                        act_l[i, 0:3] = (self.target_pos_l[i] + torch.tensor([0,0,self.SAFE_Z_OFFSET], device=self.device)) - tip_l[i]
                        if self.step_counter_l[i] > 250: self.sub_state_l[i] = "IDLE"
                elif verb == "release":
                    if self.sub_state_l[i] == "COMPUTE_TARGET_POS": self.sub_state_l[i] = "APPROACH_PEG"
                    elif self.sub_state_l[i] == "APPROACH_PEG":
                        self.current_gripper_state_l[i] = -1.0; grip_l[i] = -1.0
                        tgt = self.current_triplet_l[i]["target"]
                        peg = tgt if tgt.startswith("peg_") else (self.last_target_l[i] if str(self.last_target_l[i]).startswith("peg_") else None)
                        if peg:
                            act_l[i, 0:3] = (self.get_target_coordinates(peg, "left_arm", i) + torch.tensor([0,0,self.SAFE_Z_OFFSET], device=self.device)) - tip_l[i]
                            if self.step_counter_l[i] > 120: self.sub_state_l[i] = "DESCEND_TO_PEG"
                        else:
                            act_l[i, 0:3] = self.target_pos_l[i] - tip_l[i]
                            if self.step_counter_l[i] > 50: self.sub_state_l[i] = "OPEN_GRIPPER"
                    elif self.sub_state_l[i] == "DESCEND_TO_PEG":
                        tgt = self.current_triplet_l[i]["target"]
                        peg = tgt if tgt.startswith("peg_") else self.last_target_l[i]
                        dest = self.get_target_coordinates(peg, "left_arm", i).clone()
                        dest[2] = self.Z_TABLE
                        act_l[i, 0:3] = dest - tip_l[i]
                        if self.step_counter_l[i] > 120: self.sub_state_l[i] = "OPEN_GRIPPER"
                    elif self.sub_state_l[i] == "OPEN_GRIPPER":
                        self.current_gripper_state_l[i] = 1.0; grip_l[i] = 1.0
                        if self.step_counter_l[i] == 10:
                            obj = self.attached_target_l[i]
                            if obj and obj != "None":
                                ring = self.env.scene[obj]
                                new_s = ring.data.root_state_w.clone()
                                new_s[i, 2] = self.Z_TABLE
                                ring.write_root_state_to_sim(new_s)
                            self.attached_target_l[i] = None
                        if self.step_counter_l[i] > 80: self.sub_state_l[i] = "IDLE"
                self.step_counter_l[i] += 1
            else:
                act_l[i, 0:3] = self.target_pos_l[i] - tip_l[i] if torch.any(self.target_pos_l[i] != 0) else 0.0

        return torch.cat([act_r, grip_r, act_l, grip_l], dim=-1)

# ==========================================
# CLASSE 1.5: IL WRAPPER RL VETTORIZZATO (Eredita da VecEnv!)
# ==========================================
class DVRKVisionHRLWrapper(VecEnv):
    def __init__(self, isaac_env, state_machine, tcc_model=None):
        self.env = isaac_env
        self.sm = state_machine
        self.tcc_model = tcc_model
        
        self.num_envs = isaac_env.num_envs
        self.stack_size = 3                 
        self.emb_dim = 32
        
        act_space = spaces.MultiDiscrete([len(VERB_MAP), len(TARGET_MAP), len(VERB_MAP), len(TARGET_MAP)])
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.stack_size * self.emb_dim,), dtype=np.float32)

        self.render_mode = None
        
        super().__init__(self.num_envs, obs_space, act_space)
        
        self.emb_buffer = torch.zeros((self.num_envs, self.stack_size, self.emb_dim), dtype=torch.float32, device=self.env.device)
        self.preprocess = T.Compose([T.Resize((224, 224)), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.max_steps = 400
        self.current_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.env.device)
        self.goal_embedding = None
        self.actions = None

    def _get_batched_embeddings(self):
        """Elabora le 64 foto in parallelo sulla GPU in 0.05 secondi."""
        rgb_data = self.env.scene.sensors["camera"].data.output["rgb"] # [num_envs, H, W, 4]
        img_batch = rgb_data[:, :, :, :3].permute(0, 3, 1, 2).float() / 255.0
        img_batch = self.preprocess(img_batch)
        with torch.no_grad():
            return self.tcc_model(img_batch).squeeze() # Restituisce [num_envs, 32]

    def reset(self):
        # Questo viene chiamato da SB3 all'inizio del training globale
        self.env.reset()
        self.current_step[:] = 0
        for i in range(self.num_envs):
            self.sm.reset_env(i)
        
        self.emb_buffer.zero_()
        return self._get_obs()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        actions = self.actions # actions è [num_envs, 4] numpy array
        
        # 1. Assegna i comandi a tutti gli ambienti
        for i in range(self.num_envs):
            verb_l, tgt_l = VERB_MAP[actions[i, 0]], TARGET_MAP[actions[i, 1]]
            verb_r, tgt_r = VERB_MAP[actions[i, 2]], TARGET_MAP[actions[i, 3]]
            
            if tgt_l == "None": verb_l = "idle"
            if tgt_r == "None": verb_r = "idle"

            self.sm.set_new_triplet(verb_l, "left_arm", tgt_l, i)
            self.sm.set_new_triplet(verb_r, "right_arm", tgt_r, i)
        
        # 2. Loop Fisico Vettorizzato (Le 64 braccia si muovono assieme)
        robot_r = self.env.scene["robot_right"]
        robot_l = self.env.scene["robot_left"]
        
        for _ in range(1000):
            _, _, terminated_il, truncated_il, _ = self.env.step(self.sm.get_action())

            # Sincronizza i reset triggerati da IsaacLab (ring fuori bounds, timeout interno)
            # con la state machine, IMMEDIATAMENTE nel loop fisico.
            reset_il = (terminated_il | truncated_il).nonzero(as_tuple=False).squeeze(-1)
            if reset_il.numel() > 0:
                for idx in reset_il.tolist():
                    self.sm.reset_env(idx)
                    self.current_step[idx] = 0
                    self.emb_buffer[idx].zero_()

            # Snap Grasp Vettorizzato con Vincolo di Distanza
            tip_r = robot_r.data.body_pos_w[:, robot_r.find_bodies("psm_tool_tip_link")[0][0]].clone()
            tip_l = robot_l.data.body_pos_w[:, robot_l.find_bodies("psm_tool_tip_link")[0][0]].clone()
            
            for ring_name in ["ring_red", "ring_yellow", "ring_green", "ring_blue"]:
                ring = self.env.scene[ring_name]
                new_s = ring.data.root_state_w.clone()
                ring_pos = new_s[:, 0:3]
                
                # Applica Snap Grasp in blocco per TUTTI gli env
                for i in range(self.num_envs):
                    if self.sm.attached_target_r[i] == ring_name:
                        if torch.norm(tip_r[i] - ring_pos[i]) < 0.02:
                            new_s[i, 0:3] = tip_r[i] + torch.tensor([0.005, 0.0, 0.0], device=self.env.device)
                            new_s[i, 7:13] = 0.0
                        else:
                            self.sm.attached_target_r[i] = None
                            
                    elif self.sm.attached_target_l[i] == ring_name:
                        if torch.norm(tip_l[i] - ring_pos[i]) < 0.02:
                            new_s[i, 0:3] = tip_l[i] + torch.tensor([-0.005, 0.0, 0.0], device=self.env.device)
                            new_s[i, 7:13] = 0.0
                        else:
                            self.sm.attached_target_l[i] = None
                            
                ring.write_root_state_to_sim(new_s)

            # Controlla se TUTTI I 64 AMBIENTI hanno finito
            tutti_idle = True
            for i in range(self.num_envs):
                if self.sm.sub_state_r[i] != "IDLE" or self.sm.sub_state_l[i] != "IDLE":
                    tutti_idle = False; break
            if tutti_idle: break

        # 3. Osservazioni Batched
        new_embs = self._get_batched_embeddings() # [num_envs, 32]
        self.emb_buffer = torch.roll(self.emb_buffer, shifts=-1, dims=1)
        self.emb_buffer[:, -1, :] = new_embs
        
        # 4. CRUSCOTTO DI DEBUG (stampa periodica nella console, visibile in Isaac Sim)
        step0 = self.current_step[0].item()
        if step0 % 10 == 0:
            cmd_r = self.sm.current_triplet_r[0]
            cmd_l = self.sm.current_triplet_l[0]
            print(f"[ENV0 Step {step0}/{self.max_steps}] "
                  f"R: {cmd_r['verb'] if cmd_r else 'idle'}->{cmd_r['target'] if cmd_r else 'None'} ({self.sm.sub_state_r[0]}) | "
                  f"L: {cmd_l['verb'] if cmd_l else 'idle'}->{cmd_l['target'] if cmd_l else 'None'} ({self.sm.sub_state_l[0]})")

        # 5. Reward & Dones (Vettorizzate!)
        self.current_step += 1
        
        # Distanze calcolate in blocco per tutti e 64 gli ambienti
        distanze = torch.norm(new_embs - self.goal_embedding, dim=1) 
        rewards = -distanze * 100.0
        
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.device)
        infos = [{} for _ in range(self.num_envs)]
        
        # Nota: il check ring_out_of_bounds è ora gestito nativamente da IsaacLab
        # (TerminationsCfg.ring_out_of_bounds) e sincronizzato nel loop fisico sopra.

        # Stuck-state detection: se un braccio resta nello stesso sub_state troppo a lungo → reset
        MAX_STUCK_STEPS = 1500  # step fisici massimi per sub_state (escluso IDLE)
        stuck = torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.device)
        for i in range(self.num_envs):
            stuck_r = (self.sm.sub_state_r[i] != "IDLE" and self.sm.step_counter_r[i] > MAX_STUCK_STEPS)
            stuck_l = (self.sm.sub_state_l[i] != "IDLE" and self.sm.step_counter_l[i] > MAX_STUCK_STEPS)
            if stuck_r or stuck_l:
                stuck[i] = True
                rewards[i] -= 0.0  # penalità per env bloccato
                if i == 0:
                    print(f"[ENV 0] STUCK! R:{self.sm.sub_state_r[i]}({self.sm.step_counter_r[i]}) L:{self.sm.sub_state_l[i]}({self.sm.step_counter_l[i]})")
        dones = dones | stuck

        # Vittoria e Time-out
        vittoria = distanze < 0.05
        rewards[vittoria] += 100.0
        dones = dones | vittoria
        
        truncated = self.current_step >= self.max_steps
        dones = dones | truncated

        # --- GESTIONE DEI RESET INDIPENDENTI ---
        obs_numpy = self._get_obs()
        dones_numpy = dones.cpu().numpy()
        
        if dones.any():
            env_da_resettare = dones.nonzero(as_tuple=False).squeeze(-1).tolist()
            if not isinstance(env_da_resettare, list): env_da_resettare = [env_da_resettare]
            
            for i in env_da_resettare:
                if i == 0: print(f"[ENV 0] Fine Episodio! (Dist: {distanze[0]:.4f})")
                infos[i]["terminal_observation"] = obs_numpy[i].copy()
                
                # Resettiamo la fisica SOLO di quegli ambienti (env_ids deve essere un tensor)
                self.env.reset(env_ids=torch.tensor([i], device=self.env.device))
                # Resettiamo la mente di quegli ambienti
                self.sm.reset_env(i)
                self.current_step[i] = 0
                self.emb_buffer[i].zero_()
                
            # Aggiorniamo le osservazioni degli ambienti appena resettati
            obs_numpy = self._get_obs()

        return obs_numpy, rewards.cpu().numpy(), dones_numpy, infos

    def _get_obs(self):
        # Flatten della History: da [num_envs, 3, 32] a [num_envs, 96]
        return self.emb_buffer.view(self.num_envs, -1).cpu().numpy()

    def get_attr(self, attr_name, indices=None):
        # Calcola quanti ambienti stiamo interrogando
        n = self.num_envs if indices is None else (1 if isinstance(indices, int) else len(indices))
        # Restituisce una lista con il valore replicato 'n' volte
        return [getattr(self, attr_name, None)] * n

    def set_attr(self, attr_name, value, indices=None): pass
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): 
        n = self.num_envs if indices is None else (1 if isinstance(indices, int) else len(indices))
        return [None] * n
        
    def env_is_wrapped(self, wrapper_class, indices=None): 
        n = self.num_envs if indices is None else (1 if isinstance(indices, int) else len(indices))
        return [False] * n
        
    def close(self): pass

# ==========================================
# TRAINING MAIN
# ==========================================
def main_train():
    env_cfg = MDvrkEnvCfg(); env_cfg.scene.num_envs = args_cli.num_envs
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    sm = SPARTANStateMachine(isaac_env)
    tcc = XIRLResnet18(32)
    
    try:
        ckpt = torch.load("/home/zaza/isaac/m_dVrk/Data/4001.ckpt", map_location=isaac_env.device)
        sd = ckpt['model'] if 'model' in ckpt else ckpt
        clean_sd = {k.replace('resnet.', '').replace('net.', '').replace('model.', ''): v for k, v in sd.items()}
        tcc.model.load_state_dict(clean_sd, strict=False)
        tcc.to(isaac_env.device).eval()
        print("[INFO] Pesi caricati!")
    except Exception as e: print(f"[ERR] Pesi: {e}")

    rl_env = DVRKVisionHRLWrapper(isaac_env, sm, tcc)
    
    try:
        img_g = io.read_image("/home/zaza/isaac/m_dVrk/Data/goal.png")[:3].unsqueeze(0).float().to(isaac_env.device)/255.0
        proc = T.Compose([T.Resize((224, 224)), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad(): rl_env.goal_embedding = tcc(proc(img_g)).squeeze()
        print("[INFO] Goal pronto!")
    except Exception as e: 
        rl_env.goal_embedding = torch.zeros(32, device=isaac_env.device)

    # Niente Monitor, il VecEnv di Isaac gestisce i log nativamente.
    model = A2C("MlpPolicy", rl_env, verbose=1, device="cuda", tensorboard_log="/home/zaza/isaac/m_dVrk/tensorboard_logs/")
    model.learn(total_timesteps=1_000_000, callback=CheckpointCallback(10000, "/home/zaza/isaac/m_dVrk/modelli_salvati/", "dvrk_a2c"))
    model.save("/home/zaza/isaac/m_dVrk/modelli_salvati/dvrk_finale")

if __name__ == "__main__":
    main_train()
    simulation_app.close()