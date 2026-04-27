"""Script per testare la Macchina a Stati sul dVRK usando il Manager-based workflow."""
import argparse
import threading

# 1. SETUP INIZIALE DI ISAAC SIM (Deve avvenire prima degli altri import)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Macchina a stati per il Peg and Ring.")
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

# Importa la configurazione del tuo ambiente
from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg 

# ==========================================
# CLASSE 1: LA MACCHINA A STATI (Il Cervello)
# ==========================================
class SPARTANStateMachine:
    # --- COSTANTI DI CONFIGURAZIONE ---
    Z_TABLE = 0.72              # Altezza in cui l'anello riposa sulla basetta
    SAFE_Z_OFFSET = 0.03        # Altezza di sicurezza per sorvolare i pioli (3 cm)
    GRASP_Z_OFFSET = 0.005      # Offset di discesa per afferrare l'anello
    
    # NUOVE SOGLIE DISACCOPPIATE (Cilindro di tolleranza)
    TOLERANCE_XY = 0.001        # Tolleranza radiale strettissima (2 mm) per centrare il buco
    TOLERANCE_Z = 0.007         # Tolleranza verticale più rilassata (6 mm) per l'errore a regime

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

        self.current_gripper_state = {
            "right_arm": 1.0, 
            "left_arm": 1.0
        }
        self.attached_target = {"right_arm": None, "left_arm": None}
        self.last_target = {"right_arm": None, "left_arm": None}

        self.peg_inventory = {
            "peg_red": 0, "peg_yellow": 0, "peg_green": 0, 
            "peg_blue": 0, "peg_gray": 0, "peg_gray1": 0
        }
        self.RING_THICKNESS = 0.004 # 4 millimetri per anello (2 reali + margine di sicurezza visivo)

    def set_new_triplet(self, verb: str, subject: str, target: str):
        new_cmd = {"verb": verb, "subject": subject, "target": target}
        
        if self.current_triplet[subject] == new_cmd:
            return
            
        print(f"[SM] Ricevuto nuovo comando: <{verb}, {subject}, {target}>")
        self.current_triplet[subject] = new_cmd
        self.sub_state[subject] = "COMPUTE_TARGET_POS"
        self.step_counter[subject] = 0

    def get_target_coordinates(self, target_name: str, subject: str):
        """Trova le coordinate globali vettorizzate compensate in base al braccio."""
        
        # Il Segno: +1.0 per il destro, -1.0 per il sinistro
        segno = 1.0 if subject == "right_arm" else -1.0
        
        # Compensazione speculare!
        vettore_compensazione = torch.tensor([[-0.005 * segno, 0.0, 0.0]], device=self.device)

        if target_name in self.env.scene.keys():
            entity = self.env.scene[target_name]
            if hasattr(entity, "data"):
                pos = entity.data.root_pos_w.clone()
                return pos + vettore_compensazione
            elif hasattr(entity, "get_world_poses"):
                pos, _ = entity.get_world_poses()
                return pos.clone() + vettore_compensazione
            else:
                return torch.tensor([[0.5, 0.0, 0.5]], device=self.device).repeat(self.env.num_envs, 1)
        else:
            print(f"[ERRORE] Target '{target_name}' non trovato nella scena!")
            return torch.tensor([[0.5, 0.0, 0.5]], device=self.device).repeat(self.env.num_envs, 1)
        
    def get_action(self):
        """Calcola e restituisce il tensore delle azioni per entrambi i bracci (MODALITÀ RELATIVA)."""
        action_right = torch.zeros((self.env.num_envs, 6), device=self.device)
        action_left = torch.zeros((self.env.num_envs, 6), device=self.device)
        
        gripper_right = torch.full((self.env.num_envs, 2), self.current_gripper_state["right_arm"], device=self.device)
        gripper_left = torch.full((self.env.num_envs, 2), self.current_gripper_state["left_arm"], device=self.device)

        # --- LOGICA DI IDLE (Relativa) ---
        for arm_name in ["right_arm", "left_arm"]:
            scene_name = "robot_right" if arm_name == "right_arm" else "robot_left"
            robot = self.env.scene[scene_name]
            
            body_idx = robot.find_bodies("psm_tool_tip_link")[0][0]
            tip_pos_w = robot.data.body_pos_w[:, body_idx]
            
            idle_action = torch.zeros((self.env.num_envs, 6), device=self.device)
            
            if torch.any(self.target_pos[arm_name] != 0):
                errore = self.target_pos[arm_name] - tip_pos_w
                idle_action[:, 0:3] = errore
            else:
                idle_action[:, 0:3] = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
            
            if arm_name == "right_arm": action_right[:] = idle_action
            if arm_name == "left_arm": action_left[:] = idle_action

        # --- LOGICA ATTIVA INDIPENDENTE (Relativa) ---
        for subject in ["right_arm", "left_arm"]:
            if self.current_triplet[subject] is not None and self.sub_state[subject] != "IDLE":
                verb = self.current_triplet[subject]["verb"]
                
                scene_name = "robot_right" if subject == "right_arm" else "robot_left"
                robot = self.env.scene[scene_name]
                
                active_arm = action_right if subject == "right_arm" else action_left
                active_gripper = gripper_right if subject == "right_arm" else gripper_left

                body_idx = robot.find_bodies("psm_tool_tip_link")[0][0]
                tip_pos_w = robot.data.body_pos_w[:, body_idx]

                # === AZIONE: REACH ===
                if verb == "reach":
                    if self.sub_state[subject] == "COMPUTE_TARGET_POS":
                        self.sub_state[subject] = "APPROACH_ABOVE"
                        self.step_counter[subject] = 0

                    elif self.sub_state[subject] == "APPROACH_ABOVE":
                        self.target_pos[subject] = self.get_target_coordinates(self.current_triplet[subject]["target"], subject)
                        destinazione = self.target_pos[subject] + torch.tensor([[0.0, 0.0, self.SAFE_Z_OFFSET]], device=self.device)
                        active_arm[:, 0:3] = destinazione - tip_pos_w
                        
                        if self.step_counter[subject] > 150:
                            self.sub_state[subject] = "DESCEND"

                    elif self.sub_state[subject] == "DESCEND":
                        self.target_pos[subject] = self.get_target_coordinates(self.current_triplet[subject]["target"], subject)
                        destinazione_reale = self.target_pos[subject] + torch.tensor([[0.0, 0.0, self.GRASP_Z_OFFSET]], device=self.device)
                        active_arm[:, 0:3] = destinazione_reale - tip_pos_w
                        
                        errore_xy = destinazione_reale[:, 0:2] - tip_pos_w[:, 0:2]
                        distanza_xy = torch.norm(errore_xy, dim=-1)
                        distanza_z = torch.abs(destinazione_reale[:, 2] - tip_pos_w[:, 2])
                        
                        if distanza_xy[0] < self.TOLERANCE_XY and distanza_z[0] < self.TOLERANCE_Z:
                            print(f"[SM {subject}] Target agganciato! Err_XY: {distanza_xy[0].item():.4f}m | Err_Z: {distanza_z[0].item():.4f}m")
                            self.sub_state[subject] = "SETTLE"
                            self.step_counter[subject] = 0

                    elif self.sub_state[subject] == "SETTLE":
                        destinazione_reale = self.target_pos[subject] + torch.tensor([[0.0, 0.0, self.GRASP_Z_OFFSET]], device=self.device)
                        active_arm[:, 0:3] = destinazione_reale - tip_pos_w
                        
                        if self.step_counter[subject] > 15:
                            self.last_target[subject] = self.current_triplet[subject]["target"]
                            self.target_pos[subject] = destinazione_reale
                            self.sub_state[subject] = "IDLE"

                # === AZIONE: GRASP ===
                elif verb == "grasp":
                    if self.sub_state[subject] == "COMPUTE_TARGET_POS":
                        self.sub_state[subject] = "CLOSE_GRIPPER"
                        self.step_counter[subject] = 0

                    elif self.sub_state[subject] == "CLOSE_GRIPPER":
                        self.current_gripper_state[subject] = -1.0
                        active_gripper[:] = self.current_gripper_state[subject]
                        
                        destinazione = self.target_pos[subject]
                        active_arm[:, 0:3] = destinazione - tip_pos_w
                        
                        if self.step_counter[subject] > 100:
                            oggetto_da_prendere = self.current_triplet[subject]["target"]
                            self.attached_target[subject] = oggetto_da_prendere
                            
                            altro_braccio = "left_arm" if subject == "right_arm" else "right_arm"
                            if self.attached_target[altro_braccio] == oggetto_da_prendere:
                                print(f"[SM] SCAMBIO! Il {altro_braccio} rilascia {oggetto_da_prendere} al {subject}.")
                                self.attached_target[altro_braccio] = None
                                self.current_gripper_state[altro_braccio] = 1.0 
                                
                            self.sub_state[subject] = "LIFT_UP"
                    
                    elif self.sub_state[subject] == "LIFT_UP":
                        active_gripper[:] = self.current_gripper_state[subject]
                        destinazione = self.target_pos[subject] + torch.tensor([[0.0, 0.0, self.SAFE_Z_OFFSET]], device=self.device)
                        active_arm[:, 0:3] = destinazione - tip_pos_w
                        
                        if self.step_counter[subject] > 250:
                            self.target_pos[subject] = destinazione
                            self.sub_state[subject] = "IDLE"

                # === AZIONE: RELEASE ===
                elif verb == "release":
                    if self.sub_state[subject] == "COMPUTE_TARGET_POS":
                        self.sub_state[subject] = "APPROACH_PEG"
                        self.step_counter[subject] = 0

                    elif self.sub_state[subject] == "APPROACH_PEG":
                        self.current_gripper_state[subject] = -1.0 
                        active_gripper[:] = self.current_gripper_state[subject]
                        
                        target_cmd = self.current_triplet[subject]["target"]
                        last_tgt = self.last_target[subject]
                        target_peg = target_cmd if target_cmd.startswith("peg_") else (last_tgt if str(last_tgt).startswith("peg_") else None)
                        
                        if target_peg is not None:
                            target_pos_w = self.get_target_coordinates(target_peg, subject)
                            destinazione = target_pos_w + torch.tensor([[0.0, 0.0, self.SAFE_Z_OFFSET]], device=self.device)
                            active_arm[:, 0:3] = destinazione - tip_pos_w
                            
                            if self.step_counter[subject] > 120:
                                self.sub_state[subject] = "DESCEND_TO_PEG"
                                self.step_counter[subject] = 0
                        else:
                            active_arm[:, 0:3] = self.target_pos[subject] - tip_pos_w
                            if self.step_counter[subject] > 50:
                                self.sub_state[subject] = "OPEN_GRIPPER"
                                self.step_counter[subject] = 0

                    elif self.sub_state[subject] == "DESCEND_TO_PEG":
                        self.current_gripper_state[subject] = -1.0 
                        active_gripper[:] = self.current_gripper_state[subject]
                        
                        target_cmd = self.current_triplet[subject]["target"]
                        last_tgt = self.last_target[subject]
                        target_peg = target_cmd if target_cmd.startswith("peg_") else (last_tgt if str(last_tgt).startswith("peg_") else None)
                        
                        target_pos_w = self.get_target_coordinates(target_peg, subject)
                        destinazione = target_pos_w.clone()
                        
                        num_anelli_presenti = self.peg_inventory.get(target_peg, 0)
                        altezza_pila = self.Z_TABLE + (num_anelli_presenti * self.RING_THICKNESS)
                        destinazione[:, 2] = altezza_pila
                        
                        active_arm[:, 0:3] = destinazione - tip_pos_w
                        
                        if self.step_counter[subject] > 120:
                            self.sub_state[subject] = "OPEN_GRIPPER"
                            self.step_counter[subject] = 0

                    elif self.sub_state[subject] == "OPEN_GRIPPER":
                        self.current_gripper_state[subject] = 1.0 
                        active_gripper[:] = self.current_gripper_state[subject]
                        
                        target_cmd = self.current_triplet[subject]["target"]
                        last_tgt = self.last_target[subject]
                        target_peg = target_cmd if target_cmd.startswith("peg_") else (last_tgt if str(last_tgt).startswith("peg_") else None)

                        if target_peg is not None:
                            target_pos_w = self.get_target_coordinates(target_peg, subject)
                            destinazione = target_pos_w.clone()
                            
                            num_anelli_presenti = self.peg_inventory.get(target_peg, 0)
                            altezza_pila = self.Z_TABLE + (num_anelli_presenti * self.RING_THICKNESS)
                            destinazione[:, 2] = altezza_pila
                            
                            active_arm[:, 0:3] = destinazione - tip_pos_w
                        else:
                            active_arm[:, 0:3] = self.target_pos[subject] - tip_pos_w

                        # Snap Release Magico
                        if self.step_counter[subject] == 10:
                            obj_name = self.attached_target[subject]
                            if target_peg is not None and obj_name is not None:
                                ring = self.env.scene[obj_name]
                                new_state = ring.data.root_state_w.clone()
                                
                                peg_entity = self.env.scene[target_peg]
                                if hasattr(peg_entity, "get_world_poses"):
                                    pos_vera, _ = peg_entity.get_world_poses()
                                    new_state[:, 0:3] = pos_vera.clone()
                                elif hasattr(peg_entity, "data"):
                                    new_state[:, 0:3] = peg_entity.data.root_pos_w.clone()
                                    
                                # FIX PILA DI ANELLI (STACKING)
                                num_anelli_presenti = self.peg_inventory.get(target_peg, 0)
                                nuova_z = self.Z_TABLE + (num_anelli_presenti * self.RING_THICKNESS)
                                new_state[:, 2] = nuova_z
                                
                                self.peg_inventory[target_peg] = num_anelli_presenti + 1
                                
                                new_state[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
                                new_state[:, 7:13] = 0.0
                                ring.write_root_state_to_sim(new_state)

                            self.attached_target[subject] = None
                            self.last_target[subject] = None
                        
                        if self.step_counter[subject] > 80:
                            if target_peg is not None:
                                self.sub_state[subject] = "RETREAT"
                            else:
                                self.sub_state[subject] = "IDLE"
                            self.step_counter[subject] = 0

                elif self.sub_state[subject] == "RETREAT":
                    self.current_gripper_state[subject] = 1.0
                    active_gripper[:] = self.current_gripper_state[subject]
                    
                    target_cmd = self.current_triplet[subject]["target"]
                    last_tgt = self.last_target[subject]
                    target_peg = target_cmd if target_cmd.startswith("peg_") else (last_tgt if str(last_tgt).startswith("peg_") else None)
                    
                    if target_peg is not None:
                         target_pos_w = self.get_target_coordinates(target_peg, subject)
                         destinazione = target_pos_w + torch.tensor([[0.0, 0.0, 0.10]], device=self.device)
                         active_arm[:, 0:3] = destinazione - tip_pos_w

                    if self.step_counter[subject] > 100:
                        if target_peg is not None:
                             self.target_pos[subject] = destinazione
                        self.sub_state[subject] = "IDLE"

                self.step_counter[subject] += 1

        return torch.cat([action_right, gripper_right, action_left, gripper_left], dim=-1)

# ==========================================
# CLASSE 2: INTERFACCIA GRAFICA (GUI)
# ==========================================
class StateMachineUI:
    def __init__(self):
        self.window = ui.Window("SPARTAN State Machine", width=300, height=350)
        self.state_labels = {}
        self.style_active = {"color": 0xFF00FF00, "font_size": 20}  
        self.style_inactive = {"color": 0xFF888888, "font_size": 16} 
        self.style_header = {"color": 0xFFFFFFFF, "font_size": 24, "margin_width": 10} 
        self._build_ui()

    def _build_ui(self):
        with self.window.frame:
            with ui.VStack(spacing=10):
                ui.Label("VERB ATTUALE (Right Arm):", style=self.style_header)
                self.verb_label = ui.Label("Nessuno", style={"color": 0xFF00A5FF, "font_size": 20}) 
                ui.Line(style={"color": 0xFF555555}) 
                ui.Label("ALBERO DEGLI STATI:", style=self.style_header)
                
                all_states = [
                    "IDLE", "COMPUTE_TARGET_POS", "APPROACH_ABOVE", "DESCEND", "SETTLE",
                    "CLOSE_GRIPPER", "LIFT_UP", "APPROACH_PEG", "DESCEND_TO_PEG",
                    "OPEN_GRIPPER", "RETREAT"
                ]
                
                for state in all_states:
                    with ui.HStack(height=0):
                        ui.Spacer(width=20) 
                        label = ui.Label(f"• {state}", style=self.style_inactive)
                        self.state_labels[state] = label

    def update(self, current_verb, current_sub_state):
        verb_text = str(current_verb).upper() if current_verb else "IN ATTESA..."
        self.verb_label.text = f"> {verb_text} <"
        for state_name, label in self.state_labels.items():
            label.style = self.style_active if state_name == current_sub_state else self.style_inactive

# ==========================================
# VARIABILI E FUNZIONI GLOBALI (Terminale)
# ==========================================
trigger_comando = False
comando_utente = None

def ascolta_terminale():
    global trigger_comando, comando_utente
    while True:
        testo = input("\n[TERMINALE] Scrivi comando (es: 'reach right_arm ring_red'):\n> ")
        parti = testo.strip().split()
        if len(parti) == 3:
            comando_utente = (parti[0], parti[1], parti[2])
            trigger_comando = True
        else:
            print("[TERMINALE] Errore: scrivi 3 parole separate da spazio (Verbo Soggetto Target).")

# ==========================================
# FUNZIONE PRINCIPALE (Main Loop)
# ==========================================
def main():
    global trigger_comando
    
    # Inizializzazione Ambiente
    env_cfg = MDvrkEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    state_machine = SPARTANStateMachine(env)
    obs, _ = env.reset()
    
    # Avvio thread del terminale
    keyboard_thread = threading.Thread(target=ascolta_terminale, daemon=True)
    keyboard_thread.start()

    sm_gui = StateMachineUI()

    # Inizializzazione Debug Markers (Terne)
    marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/DebugFrames")
    marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
    debug_frames = VisualizationMarkers(marker_cfg)

    # --- CICLO DI SIMULAZIONE ---
    while simulation_app.is_running():
        
        # 2. Ascolto Comandi
        if trigger_comando and state_machine.sub_state["right_arm"] == "IDLE" and state_machine.sub_state["left_arm"] == "IDLE":
            if comando_utente is not None:
                verb, subject, target = comando_utente
                state_machine.set_new_triplet(verb, subject, target)
            trigger_comando = False

        # 3. Step Fisico
        actions = state_machine.get_action()
        obs, rewards, dones, _, _ = env.step(actions)

        # 4. Rendering Camera OpenCV
        if "camera" in env.scene.sensors:
            rgb_data = env.scene.sensors["camera"].data.output["rgb"]
            img = rgb_data[0].cpu().numpy().astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR) if img.shape[-1] == 4 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[-1] == 3 else img
            cv2.imshow("Telecamera dVRK - View", img_bgr)
            cv2.waitKey(1)  

        # 5. Snap Grasp Cinematico (Forzatura posizione in mano)
        for arm_name, attached_obj in state_machine.attached_target.items():
            if attached_obj is not None:
                ring = env.scene[attached_obj]
                scene_name = "robot_right" if arm_name == "right_arm" else "robot_left"
                robot = env.scene[scene_name]
                
                body_idx = robot.find_bodies("psm_tool_tip_link")[0][0]
                tip_pos_w = robot.data.body_pos_w[:, body_idx].clone()

                new_state = ring.data.root_state_w.clone()
                
                # OFFSET REALISTICO SPECULARE
                segno = 1.0 if arm_name == "right_arm" else -1.0
                offset_presa = torch.tensor([0.005 * segno, 0.0, 0.0], device=env.device)
                
                new_state[:, 0:3] = tip_pos_w + offset_presa
                new_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
                new_state[:, 7:13] = 0.0 
                ring.write_root_state_to_sim(new_state)
        
        # 6. Aggiornamento GUI (Mostriamo il braccio destro come riferimento per non sdoppiare la UI)
        cmd_dx = state_machine.current_triplet["right_arm"]
        verb = cmd_dx["verb"] if cmd_dx is not None else None
        stato_dx = state_machine.sub_state["right_arm"]
        sm_gui.update(current_verb=verb, current_sub_state=stato_dx)

if __name__ == "__main__":
    main()
    simulation_app.close()