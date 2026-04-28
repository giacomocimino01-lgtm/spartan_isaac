"""Script completo dVRK: Macchina a Stati SPARTAN + RL Wrapper + Visualizzazione OpenCV."""
import argparse
import threading

# 1. SETUP INIZIALE DI ISAAC SIM
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Macchina a stati per il Peg and Ring con RL.")
parser.add_argument("--num_envs", type=int, default=1, help="Numero di ambienti da spawnare in parallelo.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. IMPORTAZIONI LIBRERIE
import cv2
import numpy as np
import torch
import omni.ui as ui

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

import gymnasium as gym
from gymnasium import spaces
from collections import deque

import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as T
import torchvision.io as io

# Importa la configurazione del tuo ambiente
from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg 

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

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
# CLASSE 1: LA MACCHINA A STATI (Il Cervello Mid-Level)
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
        self.current_triplet = {"right_arm": None, "left_arm": None}
        self.sub_state = {"right_arm": "IDLE", "left_arm": "IDLE"}
        self.step_counter = {"right_arm": 0, "left_arm": 0}
        self.target_pos = {
            "right_arm": torch.zeros((env.num_envs, 3), device=self.device),
            "left_arm": torch.zeros((env.num_envs, 3), device=self.device)
        }
        self.current_gripper_state = {"right_arm": 1.0, "left_arm": 1.0}
        self.attached_target = {"right_arm": None, "left_arm": None}
        self.last_target = {"right_arm": None, "left_arm": None}

    def set_new_triplet(self, verb: str, subject: str, target: str):
        new_cmd = {"verb": verb, "subject": subject, "target": target}
        
        # FIX: Ignora il comando SOLO se lo sta attualmente eseguendo.
        # Se ha finito (è in IDLE), deve poterlo rifare se l'IA lo richiede!
        if self.current_triplet[subject] == new_cmd and self.sub_state[subject] != "IDLE": 
            return
            
        print(f"[SM] NUOVO COMANDO: <{verb}, {subject}, {target}>")
        self.current_triplet[subject] = new_cmd
        
        if verb == "idle":
            self.sub_state[subject] = "IDLE"
        else:
            self.sub_state[subject] = "COMPUTE_TARGET_POS" 
            
        self.step_counter[subject] = 0

    def get_target_coordinates(self, target_name: str, subject: str):
        # PROTEZIONE CONTRO TARGET 'None'
        if target_name == "None" or target_name is None:
            return torch.tensor([[0.375, -0.05, 0.85]], device=self.device).repeat(self.env.num_envs, 1)

        segno = 1.0 if subject == "right_arm" else -1.0
        vettore_compensazione = torch.tensor([[-0.005 * segno, 0.0, 0.0]], device=self.device)

        if target_name in self.env.scene.keys():
            entity = self.env.scene[target_name]
            if hasattr(entity, "data"):
                pos = entity.data.root_pos_w.clone()
                return pos + vettore_compensazione
            elif hasattr(entity, "get_world_poses"):
                pos, _ = entity.get_world_poses()
                return pos.clone() + vettore_compensazione
        return torch.tensor([[0.5, 0.0, 0.5]], device=self.device).repeat(self.env.num_envs, 1)
        
    def get_action(self):
        action_right = torch.zeros((self.env.num_envs, 6), device=self.device)
        action_left = torch.zeros((self.env.num_envs, 6), device=self.device)
        gripper_right = torch.full((self.env.num_envs, 2), self.current_gripper_state["right_arm"], device=self.device)
        gripper_left = torch.full((self.env.num_envs, 2), self.current_gripper_state["left_arm"], device=self.device)

        for arm_name in ["right_arm", "left_arm"]:
            scene_name = "robot_right" if arm_name == "right_arm" else "robot_left"
            robot = self.env.scene[scene_name]
            body_idx = robot.find_bodies("psm_tool_tip_link")[0][0]
            tip_pos_w = robot.data.body_pos_w[:, body_idx]
            idle_action = torch.zeros((self.env.num_envs, 6), device=self.device)
            if torch.any(self.target_pos[arm_name] != 0):
                idle_action[:, 0:3] = self.target_pos[arm_name] - tip_pos_w
            if arm_name == "right_arm": action_right[:] = idle_action
            if arm_name == "left_arm": action_left[:] = idle_action

        for subject in ["right_arm", "left_arm"]:
            if self.current_triplet[subject] is not None and self.sub_state[subject] != "IDLE":
                verb = self.current_triplet[subject]["verb"]
                scene_name = "robot_right" if subject == "right_arm" else "robot_left"
                robot = self.env.scene[scene_name]
                active_arm = action_right if subject == "right_arm" else action_left
                active_gripper = gripper_right if subject == "right_arm" else gripper_left
                body_idx = robot.find_bodies("psm_tool_tip_link")[0][0]
                tip_pos_w = robot.data.body_pos_w[:, body_idx]

                if verb == "reach":
                    if self.sub_state[subject] == "COMPUTE_TARGET_POS":
                        self.sub_state[subject] = "APPROACH_ABOVE"
                        self.step_counter[subject] = 0
                    elif self.sub_state[subject] == "APPROACH_ABOVE":
                        self.target_pos[subject] = self.get_target_coordinates(self.current_triplet[subject]["target"], subject)
                        destinazione = self.target_pos[subject] + torch.tensor([[0.0, 0.0, self.SAFE_Z_OFFSET]], device=self.device)
                        active_arm[:, 0:3] = destinazione - tip_pos_w
                        if self.step_counter[subject] > 150: self.sub_state[subject] = "DESCEND"
                    elif self.sub_state[subject] == "DESCEND":
                        self.target_pos[subject] = self.get_target_coordinates(self.current_triplet[subject]["target"], subject)
                        dest_reale = self.target_pos[subject] + torch.tensor([[0.0, 0.0, self.GRASP_Z_OFFSET]], device=self.device)
                        active_arm[:, 0:3] = dest_reale - tip_pos_w
                        dist_xy = torch.norm(dest_reale[:, 0:2] - tip_pos_w[:, 0:2], dim=-1)
                        dist_z = torch.abs(dest_reale[:, 2] - tip_pos_w[:, 2])
                        if dist_xy[0] < self.TOLERANCE_XY and dist_z[0] < self.TOLERANCE_Z:
                            self.sub_state[subject] = "SETTLE"
                            self.step_counter[subject] = 0
                    elif self.sub_state[subject] == "SETTLE":
                        active_arm[:, 0:3] = (self.target_pos[subject] + self.GRASP_Z_OFFSET) - tip_pos_w
                        if self.step_counter[subject] > 15:
                            self.last_target[subject] = self.current_triplet[subject]["target"]
                            self.sub_state[subject] = "IDLE"

                elif verb == "grasp":
                    if self.sub_state[subject] == "COMPUTE_TARGET_POS":
                        self.sub_state[subject] = "CLOSE_GRIPPER"
                    elif self.sub_state[subject] == "CLOSE_GRIPPER":
                        self.current_gripper_state[subject] = -1.0
                        active_gripper[:] = -1.0
                        if self.step_counter[subject] > 100:
                            self.attached_target[subject] = self.current_triplet[subject]["target"]
                            self.sub_state[subject] = "LIFT_UP"
                    elif self.sub_state[subject] == "LIFT_UP":
                        dest = self.target_pos[subject] + torch.tensor([[0.0, 0.0, self.SAFE_Z_OFFSET]], device=self.device)
                        active_arm[:, 0:3] = dest - tip_pos_w
                        if self.step_counter[subject] > 250: self.sub_state[subject] = "IDLE"

                elif verb == "release":
                    if self.sub_state[subject] == "COMPUTE_TARGET_POS":
                        self.sub_state[subject] = "APPROACH_PEG"
                    elif self.sub_state[subject] == "APPROACH_PEG":
                        self.current_gripper_state[subject] = -1.0
                        active_gripper[:] = -1.0
                        target_cmd = self.current_triplet[subject]["target"]
                        last_tgt = self.last_target[subject]
                        target_peg = target_cmd if target_cmd.startswith("peg_") else (last_tgt if str(last_tgt).startswith("peg_") else None)
                        if target_peg:
                            target_pos_w = self.get_target_coordinates(target_peg, subject)
                            active_arm[:, 0:3] = (target_pos_w + self.SAFE_Z_OFFSET) - tip_pos_w
                            if self.step_counter[subject] > 120: self.sub_state[subject] = "DESCEND_TO_PEG"
                        else:
                            active_arm[:, 0:3] = self.target_pos[subject] - tip_pos_w
                            if self.step_counter[subject] > 50: self.sub_state[subject] = "OPEN_GRIPPER"
                    elif self.sub_state[subject] == "DESCEND_TO_PEG":
                        target_cmd = self.current_triplet[subject]["target"]
                        target_peg = target_cmd if target_cmd.startswith("peg_") else self.last_target[subject]
                        target_pos_w = self.get_target_coordinates(target_peg, subject)
                        dest = target_pos_w.clone()
                        dest[:, 2] = self.Z_TABLE
                        active_arm[:, 0:3] = dest - tip_pos_w
                        if self.step_counter[subject] > 120: self.sub_state[subject] = "OPEN_GRIPPER"
                    elif self.sub_state[subject] == "OPEN_GRIPPER":
                        self.current_gripper_state[subject] = 1.0
                        active_gripper[:] = 1.0
                        if self.step_counter[subject] == 10:
                            obj_name = self.attached_target[subject]
                            # PROTEZIONE RELEASE
                            if obj_name is not None and obj_name != "None":
                                ring = self.env.scene[obj_name]
                                new_state = ring.data.root_state_w.clone()
                                new_state[:, 2] = self.Z_TABLE
                                ring.write_root_state_to_sim(new_state)
                            self.attached_target[subject] = None
                        if self.step_counter[subject] > 80: self.sub_state[subject] = "IDLE"

                self.step_counter[subject] += 1
        return torch.cat([action_right, gripper_right, action_left, gripper_left], dim=-1)

# ==========================================
# CLASSE 1.5: IL WRAPPER PER IL REINFORCEMENT LEARNING
# ==========================================
class DVRKVisionHRLWrapper(gym.Env):
    def __init__(self, isaac_env, state_machine, tcc_model=None, gui=None):
        super().__init__()
        self.gui = gui
        self.env = isaac_env
        self.sm = state_machine
        self.tcc_model = tcc_model
        self.physics_steps_per_rl_step = 10 
        self.stack_size = 3                 
        self.emb_dim = 32
        self.action_space = spaces.MultiDiscrete([len(VERB_MAP), len(TARGET_MAP), len(VERB_MAP), len(TARGET_MAP)])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.stack_size * self.emb_dim,), dtype=np.float32)
        self.emb_buffer = deque(maxlen=self.stack_size)
        self.preprocess = T.Compose([T.Resize((224, 224)), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.max_steps = 40
        self.current_step = 0
        self.goal_embedding = None

    def _compute_reward(self, current_emb):
        if self.goal_embedding is None: return 0.0
        cur = torch.tensor(current_emb, device=self.env.device) if isinstance(current_emb, np.ndarray) else current_emb
        distanza = torch.norm(cur - self.goal_embedding)
        return -distanza.item() # Reward originale (negativa, scala 0-1)
    
    def _get_real_embedding(self):
        rgb_data = self.env.scene.sensors["camera"].data.output["rgb"]
        img = rgb_data[0, :, :, :3].permute(2, 0, 1).unsqueeze(0).float().to(self.env.device) / 255.0
        img = self.preprocess(img)
        with torch.no_grad():
            return self.tcc_model(img).squeeze().cpu().numpy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        self.current_step = 0
        self.sm.attached_target = {"right_arm": None, "left_arm": None}
        self.sm.sub_state = {"right_arm": "IDLE", "left_arm": "IDLE"}
        dummy_emb = np.zeros(self.emb_dim, dtype=np.float32)
        for _ in range(self.stack_size): self.emb_buffer.append(dummy_emb)
        return np.concatenate(self.emb_buffer), {}

    def step(self, action):
        verb_l, tgt_l = VERB_MAP[action[0]], TARGET_MAP[action[1]]
        verb_r, tgt_r = VERB_MAP[action[2]], TARGET_MAP[action[3]]
        
        # Protezione: se target è None, forziamo Idle
        if tgt_l == "None": verb_l = "idle"
        if tgt_r == "None": verb_r = "idle"

        self.sm.set_new_triplet(verb_l, "left_arm", tgt_l)
        self.sm.set_new_triplet(verb_r, "right_arm", tgt_r)
        
        for _ in range(1000): 
            self.env.step(self.sm.get_action())

            if self.gui is not None:
                self.gui.update(self.sm)

            # Snap Grasp Cinematico con VINCOLO DI DISTANZA
            for arm, obj in self.sm.attached_target.items():
                if obj and obj.startswith("ring_"):
                    ring = self.env.scene[obj]
                    robot = self.env.scene["robot_right" if arm=="right_arm" else "robot_left"]
                    
                    # 1. Troviamo la posizione della punta della pinza
                    tip_pos = robot.data.body_pos_w[:, robot.find_bodies("psm_tool_tip_link")[0][0]].clone()
                    
                    # 2. Troviamo la posizione attuale dell'anello
                    ring_pos = ring.data.root_pos_w[0]
                    
                    # 3. Calcoliamo la distanza tra i due
                    distanza_mano_anello = torch.norm(tip_pos - ring_pos)
                    
                    # 4. SOGLIA DI PRESA: l'anello si attacca solo se siamo entro 2 cm (0.02 metri)
                    # Se l'IA prova a fare 'grasp' da lontano, l'anello NON si attacca.
                    if distanza_mano_anello < 0.02:
                        new_s = ring.data.root_state_w.clone()
                        new_s[:, 0:3] = tip_pos + torch.tensor([0.005 if arm=="right_arm" else -0.005, 0.0, 0.0], device=self.env.device)
                        new_s[:, 7:13] = 0.0
                        ring.write_root_state_to_sim(new_s)
                    else:
                        # Se è troppo lontano, resettiamo il target attaccato a None
                        # Forza l'IA a dover fare prima un 'reach' corretto.
                        self.sm.attached_target[arm] = None
                        
        real_emb = self._get_real_embedding()
        self.emb_buffer.append(real_emb)
        
        # VISUALIZZAZIONE OPENCV
        if "camera" in self.env.scene.sensors:
            img = self.env.scene.sensors["camera"].data.output["rgb"][0].cpu().numpy().astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR) if img.shape[-1]==4 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("Telecamera dVRK - View IA", img_bgr)
            cv2.waitKey(1)

        self.current_step += 1
        reward = self._compute_reward(real_emb)
        done = False
        
        # FALLIMENTO WORKSPACE
        X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN = 0.20, 0.55, -0.25, 0.15, 0.68
        for r in ["ring_red", "ring_yellow", "ring_green", "ring_blue"]:
            pos = self.env.scene[r].data.root_pos_w[0]
            if pos[0]<X_MIN or pos[0]>X_MAX or pos[1]<Y_MIN or pos[1]>Y_MAX or pos[2]<Z_MIN:
                print(f"[RL] FALLIMENTO: {r} fuori!"); reward -= 50.0; done = True; break
        
        # VITTORIA
        if not done and reward > -0.005:
            print(f"[RL] VITTORIA!"); reward += 100.0; done = True

        truncated = self.current_step >= self.max_steps
        return np.concatenate(self.emb_buffer), reward, done, truncated, {"dist": -reward}


# ==========================================
# CLASSE 2: INTERFACCIA GRAFICA OMNIVERSE (GUI)
# ==========================================
import omni.ui as ui

class StateMachineUI:
    def __init__(self):
        # Finestra Braccio Destro
        self.win_r = ui.Window("SPARTAN - Right Arm", width=250, height=350, dockPreference=ui.DockPreference.RIGHT_TOP)
        self.state_labels_r = {}
        
        # Finestra Braccio Sinistro
        self.win_l = ui.Window("SPARTAN - Left Arm", width=250, height=350, dockPreference=ui.DockPreference.RIGHT_BOTTOM)
        self.state_labels_l = {}
        
        # Stili
        self.style_active_r = {"color": 0xFF00A5FF, "font_size": 18}  # Arancione per Right
        self.style_active_l = {"color": 0xFFFF00FF, "font_size": 18}  # Magenta per Left
        self.style_inactive = {"color": 0xFF666666, "font_size": 14} 
        self.style_header = {"color": 0xFFFFFFFF, "font_size": 20, "margin_width": 5} 
        
        self.all_states = [
            "IDLE", "COMPUTE_TARGET_POS", "APPROACH_ABOVE", "DESCEND", "SETTLE",
            "CLOSE_GRIPPER", "LIFT_UP", "APPROACH_PEG", "DESCEND_TO_PEG",
            "OPEN_GRIPPER", "RETREAT"
        ]
        
        self._build_ui(self.win_r, self.state_labels_r, "RIGHT ARM")
        self._build_ui(self.win_l, self.state_labels_l, "LEFT ARM")

    def _build_ui(self, window, labels_dict, title):
        with window.frame:
            with ui.VStack(spacing=5):
                ui.Label(f"COMMAND ({title}):", style=self.style_header)
                verb_label = ui.Label("WAITING...", style={"color": 0xFF00FF00, "font_size": 16}) 
                labels_dict["_VERB_"] = verb_label # Salviamo la label speciale per il verbo
                ui.Line(style={"color": 0xFF555555}) 
                
                ui.Label("STATE TREE:", style=self.style_header)
                for state in self.all_states:
                    with ui.HStack(height=0):
                        ui.Spacer(width=15) 
                        label = ui.Label(f"• {state}", style=self.style_inactive)
                        labels_dict[state] = label

    def update(self, sm):
        # --- Aggiorna Destro ---
        cmd_r = sm.current_triplet["right_arm"]
        verb_r = f"{cmd_r['verb']} -> {cmd_r['target']}" if cmd_r else "IDLE / NONE"
        sub_r = sm.sub_state["right_arm"]
        
        self.state_labels_r["_VERB_"].text = str(verb_r).upper()
        for state_name in self.all_states:
            self.state_labels_r[state_name].style = self.style_active_r if state_name == sub_r else self.style_inactive

        # --- Aggiorna Sinistro ---
        cmd_l = sm.current_triplet["left_arm"]
        verb_l = f"{cmd_l['verb']} -> {cmd_l['target']}" if cmd_l else "IDLE / NONE"
        sub_l = sm.sub_state["left_arm"]
        
        self.state_labels_l["_VERB_"].text = str(verb_l).upper()
        for state_name in self.all_states:
            self.state_labels_l[state_name].style = self.style_active_l if state_name == sub_l else self.style_inactive


# ==========================================
# TRAINING MAIN
# ==========================================
def main_train():
    env_cfg = MDvrkEnvCfg(); env_cfg.scene.num_envs = args_cli.num_envs
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    sm = SPARTANStateMachine(isaac_env)
    sm_gui = StateMachineUI()
    tcc = XIRLResnet18(32)
    
    try:
        ckpt = torch.load("/home/zaza/isaac/m_dVrk/Data/4001.ckpt", map_location=isaac_env.device)
        sd = ckpt['model'] if 'model' in ckpt else ckpt
        clean_sd = {k.replace('resnet.', '').replace('net.', '').replace('model.', ''): v for k, v in sd.items()}
        tcc.model.load_state_dict(clean_sd, strict=False)
        tcc.to(isaac_env.device).eval()
        print("[INFO] Pesi caricati!")
    except Exception as e: print(f"[ERR] Pesi: {e}")

    rl_env = DVRKVisionHRLWrapper(isaac_env, sm, tcc, gui=sm_gui)
    
    try:
        img_g = io.read_image("/home/zaza/isaac/m_dVrk/Data/goal.png")[:3].unsqueeze(0).float().to(isaac_env.device)/255.0
        proc = T.Compose([T.Resize((224, 224)), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad(): rl_env.goal_embedding = tcc(proc(img_g)).squeeze()
        print("[INFO] Goal pronto!")
    except Exception as e: 
        print(f"[ERR] Goal: {e}")
        rl_env.goal_embedding = torch.zeros(32, device=isaac_env.device)

    rl_env = Monitor(rl_env)
    model = A2C("MlpPolicy", DummyVecEnv([lambda: rl_env]), verbose=1, device="cuda", tensorboard_log="/home/zaza/isaac/m_dVrk/tensorboard_logs/")
    #model = A2C("MlpPolicy", DummyVecEnv([lambda: rl_env]), verbose=1, tensorboard_log="/home/zaza/isaac/m_dVrk/tensorboard_logs/")
    model.learn(total_timesteps=100_000, callback=CheckpointCallback(10000, "/home/zaza/isaac/m_dVrk/modelli_salvati/", "dvrk_a2c"))
    model.save("/home/zaza/isaac/m_dVrk/modelli_salvati/dvrk_finale")

if __name__ == "__main__":
    main_train()
    simulation_app.close()