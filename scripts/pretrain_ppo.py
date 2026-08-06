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

# Load optional project paths from configs/defaults.yaml
def _load_project_paths():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = os.path.join(repo_root, "configs", "defaults.yaml")
    try:
        import yaml
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("paths", {})
    except Exception:
        pass
    return {}

_paths = _load_project_paths()
ARTIFACTS_DIR = _paths.get("artifacts_dir", "artifacts/")

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

# Single source of truth — import from m_dVrk package
from m_dVrk.hrl.constants import VERB_MAP, TARGET_MAP, VERB_TO_ID, TARGET_TO_ID, OBS_DIM
from m_dVrk.hrl.bc import DummyHRLWrapper, train_bc

# DummyHRLWrapper is now imported from m_dVrk.hrl.bc above.

def main():
    # Prefer dataset under artifacts if default path missing
    if not os.path.exists(args_cli.dataset_path):
        alt = os.path.join(ARTIFACTS_DIR, "datasets", "bc", os.path.basename(args_cli.dataset_path))
        if os.path.exists(alt):
            args_cli.dataset_path = alt
        else:
            print(f"[pretrain] ERROR: Dataset file {args_cli.dataset_path} does not exist. Tried alternative {alt}.")
            sys.exit(1)

    print(f"[pretrain] Loading dataset from {args_cli.dataset_path}...")
    data = np.load(args_cli.dataset_path)
    obs_array = data["obs"]
    act_array = data["actions"]
    print(f"[pretrain] Loaded {len(obs_array)} transitions. Obs shape: {obs_array.shape}, Actions shape: {act_array.shape}")

    dataset_obs_dim = obs_array.shape[1]
    print(f"[pretrain] Using obs_dim={dataset_obs_dim} (from dataset)")

    # Enforce single source of truth for observation layout
    if dataset_obs_dim != OBS_DIM:
        raise SystemExit(
            f"Dataset observation dimension ({dataset_obs_dim}) does not match canonical OBS_DIM ({OBS_DIM}). "
            "Regenerate dataset using the canonical wrapper or update hrl/constants if intentionally changed."
        )

    env_cfg = MDvrkEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.num_rerenders_on_reset = 2
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    rl_env = DummyHRLWrapper(isaac_env, obs_dim=dataset_obs_dim)

    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=nn.ReLU,
    )

    print("[pretrain] Instantiating PPO model...")
    model = PPO(
        "MlpPolicy",
        rl_env,
        verbose=1,
        device="cuda",
        policy_kwargs=policy_kwargs,
    )

    # Delegate training to hrl.bc.train_bc
    train_bc(
        model,
        obs_array,
        act_array,
        epochs=args_cli.epochs,
        batch_size=args_cli.batch_size,
        lr=args_cli.lr,
        device="cuda",
    )

    # Ensure output saved under artifacts/checkpoints/bc by default
    out_dir = os.path.dirname(args_cli.output_path)
    if not out_dir or not os.path.commonpath([os.path.abspath(out_dir), os.path.abspath(ARTIFACTS_DIR)]) == os.path.abspath(ARTIFACTS_DIR):
        args_cli.output_path = os.path.join(ARTIFACTS_DIR, "checkpoints", "bc", os.path.basename(args_cli.output_path))
    os.makedirs(os.path.dirname(args_cli.output_path), exist_ok=True)
    model.save(args_cli.output_path)
    print(f"[pretrain] Pretrained PPO model saved to {args_cli.output_path}.zip")

    rl_env.close()
    isaac_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
