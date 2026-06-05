"""Script to test the State Machine on the dVRK using the Manager-based workflow."""
import argparse
import threading

# 1. INITIAL SETUP OF ISAAC SIM (Must occur before other imports)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="State machine for the Peg and Ring.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn in parallel.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. LIBRARY IMPORTS
import cv2
import numpy as np
import torch
import omni.ui as ui
import torchvision.io as io

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

# Import environment configuration
from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg 

from parallel_env import SPARTANStateMachine, sync_attached_and_frozen_rings

# ==========================================
# CLASS 1: GRAPHICAL USER INTERFACE (GUI)
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
                ui.Label("CURRENT VERB (Right Arm):", style=self.style_header)
                self.verb_label = ui.Label("None", style={"color": 0xFF00A5FF, "font_size": 20}) 
                ui.Line(style={"color": 0xFF555555}) 
                ui.Label("STATE TREE:", style=self.style_header)
                
                all_states = [
                    "IDLE", "MOVE_TO_REACH_ENTRY", "APPROACH_ABOVE", "DESCEND", "SETTLE",
                    "CLOSE_GRIPPER", "LIFT_UP", "OPEN_GRIPPER"
                ]
                
                for state in all_states:
                    with ui.HStack(height=0):
                        ui.Spacer(width=20) 
                        label = ui.Label(f"• {state}", style=self.style_inactive)
                        self.state_labels[state] = label

    def update(self, current_verb, current_sub_state):
        verb_text = str(current_verb).upper() if current_verb else "WAITING..."
        self.verb_label.text = f"> {verb_text} <"
        for state_name, label in self.state_labels.items():
            label.style = self.style_active if state_name == current_sub_state else self.style_inactive

# ==========================================
# GLOBAL VARIABLES AND FUNCTIONS (Terminal)
# ==========================================
trigger_comando = False
comando_utente = None
stop_recording = False

def ascolta_terminale():
    global trigger_comando, comando_utente, stop_recording
    while True:
        testo = input("\n[TERMINAL] Write command (e.g., 'reach right_arm ring_red'):\n> ")
        parti = testo.strip().split()
        if len(parti) == 3:
            comando_utente = (parti[0], parti[1], parti[2])
            trigger_comando = True
        else:
            stop_recording = True
            print("[TERMINAL] Stop recording command received. Saving video...")
            print("[TERMINAL] Error: write 3 space-separated words (Verb Subject Target).")

# ==========================================
# MAIN FUNCTION (Main Loop)
# ==========================================
def main():
    global trigger_comando
    
    # Environment Initialization
    env_cfg = MDvrkEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    state_machine = SPARTANStateMachine(env)
    obs, _ = env.reset()
    
    # Start terminal listener thread
    keyboard_thread = threading.Thread(target=ascolta_terminale, daemon=True)
    keyboard_thread.start()

    sm_gui = StateMachineUI()

    # Initialize Debug Markers (Frames)
    marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/DebugFrames")
    marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
    debug_frames = VisualizationMarkers(marker_cfg)

    video_frames = []

    # --- SIMULATION LOOP ---
    while simulation_app.is_running():
        
        # 2. Listen to Commands
        if trigger_comando and state_machine.all_idle(0):
            if comando_utente is not None:
                verb, subject, target = comando_utente
                state_machine.set_new_triplet(verb, subject, target, env_id=0)
            trigger_comando = False

        # 3. Physical Step
        actions = state_machine.get_action()
        obs, rewards, dones, _, _ = env.step(actions)

        # 4. OpenCV Camera Rendering
        if "camera" in env.scene.sensors:
            rgb_data = env.scene.sensors["camera"].data.output["rgb"]
            img = rgb_data[0].cpu().numpy().astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR) if img.shape[-1] == 4 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[-1] == 3 else img
            cv2.imshow("dVRK Camera - View", img_bgr)
            video_frames.append(img)
            if stop_recording:
                video_tensor = torch.stack(video_frames)
                io.write_video("dvrk_simulation_recording.mp4", video_tensor, fps=30)
            cv2.waitKey(1)  

        # 5. Kinematic Snap Grasp (Force position in hand / Stacking)
        active_env_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        sync_attached_and_frozen_rings(env, state_machine, active_env_mask)
        
        # 6. GUI Update (Showing right arm as a reference to keep the UI unified)
        cmd_dx = state_machine.current_triplet_r[0]
        verb = cmd_dx["verb"] if cmd_dx is not None else None
        stato_dx = state_machine.sub_state_r[0]
        sm_gui.update(current_verb=verb, current_sub_state=stato_dx)

if __name__ == "__main__":
    main()
    simulation_app.close()