# dVRK Peg and Ring Simulation (Isaac Lab Extension)

This repository implements a simulation environment for a da Vinci Research Kit (dVRK) robotic setup performing a **Peg and Ring** insertion task. It is structured as an external extension to **Isaac Lab** (formerly Orbit), allowing isolated development.

The system uses **Stable Baselines 3 (PPO)** for Reinforcement Learning training, guided by a **TCC (Time-Contrastive Networks / XIRL)** vision-based reward model, and coordinate-level behaviors managed via a custom **SPARTAN State Machine**.

---

## 1. Prerequisites

Before installing this repository, ensure your system has the following installed:

1. **NVIDIA Driver & CUDA**: Compatible with the chosen Isaac Sim version.
2. **NVIDIA Isaac Sim**: Supported versions are `4.5.0`, `5.0.0`, or `5.1.0`.
3. **Isaac Lab**: Follow the official [Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
   - Ensure the Conda or `uv` environment created during Isaac Lab's installation is activated.

---

## 2. Installation

1. Clone or copy this repository outside of your main `IsaacLab` directory.
2. Activate your Isaac Lab virtual/conda environment.
3. Install the local `m_dVrk` package in editable mode:
   ```bash
   # From the repository root
   python -m pip install -e source/m_dVrk
   ```
4. Install additional required dependencies (such as Stable Baselines 3 for RL training):
   ```bash
   pip install stable-baselines3
   ```

### Verify Environment Registration
Verify that the package is correctly installed and registered as an Isaac Lab task by running:
```bash
python scripts/list_envs.py
```
If successful, you should see the registered task **`Template-M-Dvrk-v0`** in the printed table.

---

## 3. Workflow & Usage

Executing this project involves four main steps: **data collection**, **script path configuration**, **agent training**, and **evaluation**.

---

### Step 1: Collect TCC Demonstrations Dataset
To train the agent using TCC/XIRL-based rewards, you must first collect a dataset of successful demonstrations using the scripted state machine.

Run the collection script to generate trajectories:
```bash
python scripts/collect_tcc_dataset.py \
    --num_envs 8 \
    --num_videos 32 \
    --output_root ./Data/sim_dataset_xirl_extra \
    --split train \
    --class_name phase_0 \
    --headless
```
- `--output_root`: Root directory where frames will be saved. Default will point to `/mnt/data/aiprah/...` if not specified.
- `--split`: Target dataset subdirectory (usually `train` or `valid`).
- `--class_name`: The activity class directory (e.g., `phase_0`).

---

### Step 2: Configure Local Script Paths
Several scripts contain hardcoded path parameters that point to the original developer's filesystem. **You must update these lines in your local copy before running training or evaluation.**

#### 1. In `scripts/parallel_run.py`:
- **Line 978**: Modify the XIRL checkpoint path to point to your local copy of `4001.ckpt` (e.g. in the repo's `Data/` folder):
  ```python
  # Change:
  "/tmp/xirl/sim_pretrain_runs/200_sim_phase0_tcc/checkpoints/4001.ckpt"
  # To:
  "./Data/4001.ckpt"
  ```
- **Line 1013**: Modify the demonstration dataset path used for computing the goal embedding:
  ```python
  # Change:
  dataset_path = "/mnt/data/aiprah/data/sim_dataset_xirl_extra/train/phase_0/"
  # To:
  dataset_path = "./Data/sim_dataset_xirl_extra/train/phase_0/"
  ```

#### 2. In `scripts/eval_run.py`:
- **Line 29**: Modify the default `--checkpoint` path to point to your saved policy ZIP file.
- **Line 30**: Modify the default `--dataset_path` value:
  ```python
  # Change:
  default="/mnt/data/aiprah/data/sim_dataset_xirl_extra/train/phase_0/"
  # To:
  default="./Data/sim_dataset_xirl_extra/train/phase_0/"
  ```
- **Line 738**: Modify the XIRL features checkpoint path:
  ```python
  # Change:
  ckpt = torch.load("/home/aiprah/Documents/m_dVrk/Data/4001.ckpt", map_location=isaac_env.device)
  # To:
  ckpt = torch.load("./Data/4001.ckpt", map_location=isaac_env.device)
  ```

---

### Step 3: Train the RL Agent
Once the paths are configured, start the vectorised multi-environment Reinforcement Learning training:
```bash
python scripts/parallel_run.py --num_envs 64
```
- Add the `--randomize_rings` flag to randomize the initial ring configuration during resets:
  ```bash
  python scripts/parallel_run.py --num_envs 64 --randomize_rings
  ```
- Checkpoints will be saved to `modelli_salvati_sim/` and logs written to `sb3_log_sim/` or `tensorboard_logs/`.

---

### Step 4: Evaluate the Policy
To run policy rollouts with a trained agent and monitor performance:
```bash
python scripts/eval_run.py \
    --num_envs 1 \
    --checkpoint ./modelli_salvati_sim/dvrk_ppo_best_terminal_distance.zip \
    --dataset_path ./Data/sim_dataset_xirl_extra/train/phase_0/
```

---

### Step 5: Test the State Machine Interactively (Optional)
If you want to manually trigger and test individual state machine verbs (`reach`, `grasp`, `release`, `idle`) for each arm, you can run the interactive testing environment:
```bash
python scripts/old_run_state_machine.py --num_envs 1
```
- This runs the simulation with a graphical window showing the dVRK Camera View.
- In your command-line terminal, you will be prompted to enter commands in the following format:
  `[Verb] [Subject/Arm] [Target]`
- **Examples**:
  - `reach right_arm ring_red`
  - `grasp right_arm ring_red`
  - `reach right_arm peg_green`
  - `release right_arm peg_green`
  - `idle right_arm None`
- To stop the run, press any key that does not follow the format (e.g. typing `exit`), which will save a recording of the window to `dvrk_simulation_recording.mp4`.

---

## 4. Code Structure

- `source/m_dVrk/`: Core Python package containing task environments, manager scripts, assets, and UI configurations.
  - `m_dVrk/tasks/manager_based/m_dvrk/m_dvrk_env_cfg.py`: Isaac Lab configuration for environment elements (robot configuration, sensory setups, scene creation).
- `scripts/`: Execution scripts for running tasks.
  - `parallel_run.py`: Vectorised RL training script utilizing SB3 PPO.
  - `eval_run.py`: Evaluation script for trained models.
  - `collect_tcc_dataset.py`: Trajectory frame collection script.
  - `old_run_state_machine.py`: Interactive script for manual state machine test.
  - `parallel_env.py`: Shared coordinate-level State Machine implementation (`SPARTANStateMachine`).
- `Data/`: Directory storing pretrained checkpoints (`4001.ckpt`, `real.ckpt`).