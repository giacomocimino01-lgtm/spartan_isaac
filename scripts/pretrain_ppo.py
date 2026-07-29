"""Supervised pretraining (Behavior Cloning) of the PPO policy.

This script instantiates PPO (with a single headless env to define spaces), loads the collected
expert demonstrations npz file, and trains the policy network using Cross-Entropy loss on the
MultiDiscrete action heads.

Example usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ${IsaacLab_PATH}/isaaclab.sh -p scripts/pretrain_ppo.py --epochs 20 --batch_size 256
    PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 nohup ${IsaacLab_PATH}/isaaclab.sh -p scripts/pretrain_ppo.py --epochs 20 --batch_size 256 > outlog.log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from isaaclab.app import AppLauncher

# Setup argparse for AppLauncher (needed to initialize Isaac Sim)
parser = argparse.ArgumentParser(description="Pretrain PPO policy via Behavior Cloning.")
parser.add_argument("--dataset_path", type=str, default="/home/aiprah/Documents/m_dVrk/dataset_supervisionato_randomized.npz", help="Path to npz dataset.")
parser.add_argument("--output_path", type=str, default="/home/aiprah/Documents/m_dVrk/modelli_salvati_sim/randomized_dvrk_ppo_pretrained", help="Path to save the pretrained model zip.")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force headless and 1 env for lightweight spaces initialization
args_cli.headless = True
args_cli.num_envs = 1

# The camera sensor requires Isaac rendering to be enabled.
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from isaaclab.envs import ManagerBasedRLEnv
from m_dVrk.tasks.manager_based.m_dvrk.m_dvrk_env_cfg import MDvrkEnvCfg

# Define identical maps and wrapper skeleton (just to initialize spaces)
VERB_MAP = {0: "reach", 1: "grasp", 2: "release", 3: "idle"}
TARGET_MAP = {
    0: "ring_red", 1: "ring_yellow", 2: "ring_green", 3: "ring_blue",
    4: "peg_red", 5: "peg_yellow", 6: "peg_green", 7: "peg_blue",
    8: "peg_gray", 9: "None"
}
VERB_TO_ID = {v: k for k, v in VERB_MAP.items()}
TARGET_TO_ID = {v: k for k, v in TARGET_MAP.items()}

class DummyHRLWrapper(VecEnv):
    def __init__(self, isaac_env, obs_dim: int):
        """Lightweight wrapper that only defines observation/action spaces.
        
        Args:
            isaac_env: Isaac environment (used only for num_envs).
            obs_dim: Observation dimension, read from the collected dataset.
        """
        self.env = isaac_env
        self.num_envs = isaac_env.num_envs
        self.obs_dim = obs_dim
        
        act_space = spaces.MultiDiscrete([len(VERB_MAP), len(TARGET_MAP), len(VERB_MAP), len(TARGET_MAP)])

        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.render_mode = None
        super().__init__(self.num_envs, obs_space, act_space)

    def reset(self):
        return np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
    def step_async(self, actions): pass
    def step_wait(self):
        return np.zeros((self.num_envs, self.obs_dim), dtype=np.float32), np.zeros(self.num_envs), np.zeros(self.num_envs), []
    def close(self): pass
    def get_attr(self, attr_name, indices=None): return [None] * self.num_envs
    def set_attr(self, attr_name, value, indices=None): pass
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): return [None] * self.num_envs
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs

def main():
    if not os.path.exists(args_cli.dataset_path):
        print(f"[pretrain] ERROR: Dataset file {args_cli.dataset_path} does not exist. Run collect_ppo_demos.py first.")
        sys.exit(1)

    # Load dataset
    print(f"[pretrain] Loading dataset from {args_cli.dataset_path}...")
    data = np.load(args_cli.dataset_path)
    obs_array = data["obs"]
    act_array = data["actions"]
    print(f"[pretrain] Loaded {len(obs_array)} transitions. Obs shape: {obs_array.shape}, Actions shape: {act_array.shape}")

    obs_tensor = torch.tensor(obs_array, dtype=torch.float32)
    act_tensor = torch.tensor(act_array, dtype=torch.long)
    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=args_cli.batch_size, shuffle=True)

    # Initialize environment just to setup dimensions
    # obs_dim is inferred from the dataset so it always matches
    dataset_obs_dim = obs_array.shape[1]
    print(f"[pretrain] Using obs_dim={dataset_obs_dim} (from dataset)")

    env_cfg = MDvrkEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.num_rerenders_on_reset = 2
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    rl_env = DummyHRLWrapper(isaac_env, obs_dim=dataset_obs_dim)

    # Setup the same network architecture as parallel_run.py
    policy_kwargs = dict(
        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64],
        ),
        activation_fn=nn.ReLU,
    )

    # Instantiate SB3 PPO model
    print("[pretrain] Instantiating PPO model...")
    model = PPO(
        "MlpPolicy",
        rl_env,
        verbose=1,
        device="cuda",
        policy_kwargs=policy_kwargs,
    )
    policy = model.policy.to("cuda")

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.lr)
    loss_fn = nn.CrossEntropyLoss()

    print("[pretrain] Starting Behavior Cloning training...")
    policy.train()
    
    for epoch in range(args_cli.epochs):
        epoch_loss = 0.0
        correct_verb_l, correct_tgt_l = 0, 0
        correct_verb_r, correct_tgt_r = 0, 0
        total_samples = 0

        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to("cuda")
            batch_act = batch_act.to("cuda")

            # Forward pass: extract features and compute policy latent representation
            features = policy.extract_features(batch_obs)
            latent_pi, _ = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi)

            # Split logits for MultiDiscrete components: [verb_l (4), tgt_l (10), verb_r (4), tgt_r (10)]
            logits_split = torch.split(logits, [4, 10, 4, 10], dim=-1)

            loss = 0.0
            for i in range(4):
                loss += loss_fn(logits_split[i], batch_act[:, i])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_obs)
            
            # Compute accuracy for monitoring
            total_samples += len(batch_obs)
            correct_verb_l += (logits_split[0].argmax(dim=-1) == batch_act[:, 0]).sum().item()
            correct_tgt_l += (logits_split[1].argmax(dim=-1) == batch_act[:, 1]).sum().item()
            correct_verb_r += (logits_split[2].argmax(dim=-1) == batch_act[:, 2]).sum().item()
            correct_tgt_r += (logits_split[3].argmax(dim=-1) == batch_act[:, 3]).sum().item()

        avg_loss = epoch_loss / total_samples
        acc_vl = correct_verb_l / total_samples * 100
        acc_tl = correct_tgt_l / total_samples * 100
        acc_vr = correct_verb_r / total_samples * 100
        acc_tr = correct_tgt_r / total_samples * 100

        print(
            f"Epoch {epoch+1:02d}/{args_cli.epochs:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Acc L-Arm: [Verb: {acc_vl:.1f}%, Tgt: {acc_tl:.1f}%] | "
            f"Acc R-Arm: [Verb: {acc_vr:.1f}%, Tgt: {acc_tr:.1f}%]"
        )

    # Save pretrained model
    os.makedirs(os.path.dirname(args_cli.output_path), exist_ok=True)
    model.save(args_cli.output_path)
    print(f"[pretrain] Pretrained PPO model saved successfully to {args_cli.output_path}.zip")

    rl_env.close()
    isaac_env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
