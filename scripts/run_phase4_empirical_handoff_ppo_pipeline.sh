#!/usr/bin/env bash
set -euo pipefail

# Collect real Phase-3 -> park handoff states, then fine-tune the current
# Phase-4 policy on a mixture of empirical (80%) and canonical (20%) resets.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ISAACLAB_ROOT="${IsaacLab_PATH:-/home/aiprah/Documents/IsaacLab}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_INDEX="${CUDA_INDEX:-0}"
CONDA_ROOT="${CONDA_ROOT:-/home/aiprah/miniconda3}"

BANK_PATH="artifacts/datasets/bc/phase4_handoff_joint_bank80_seed20261018.npz"
DEMO_PATH="artifacts/datasets/bc/phase4_handoff_bank80_seed20261018_demos.npz"
COLLECT_LOG="artifacts/jobs/phase4_handoff_joint_bank80_seed20261018.log"
TRAIN_LOG="artifacts/jobs/phase4_empirical_handoff_bank80_replay_600k.log"

mkdir -p artifacts/jobs artifacts/datasets/bc

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$CUDA_INDEX"
export PYTHONUNBUFFERED=1

# nohup/non-interactive SSH shells do not inherit the interactive Conda env.
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate env_isaaclab

if [[ ! -s "$BANK_PATH" ]]; then
  echo "[pipeline] Collecting 80 real Phase-3 -> park entry states..."
  "$ISAACLAB_ROOT/isaaclab.sh" -p scripts/collect_phase4_true_handoff_demos.py \
    --num_envs 8 \
    --num_episodes 80 \
    --output_path "$DEMO_PATH" \
    --entry_state_bank_output "$BANK_PATH" \
    --phase3_checkpoint artifacts/evaluations/phase3_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/phase3_bimanual_minimal_best_50ep.zip \
    --phase3_tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase3_tcc/checkpoints/2600.ckpt \
    --phase4_tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase4_tcc/checkpoints/2600.ckpt \
    --dataset_root /mnt/data/aiprah/data/random_sim_dataset_tcc/train \
    --seed 20261018 \
    --action_mask_mode minimal \
    --headless \
    --device "$DEVICE" \
    > "$COLLECT_LOG" 2>&1
fi

echo "[pipeline] Entry-state bank ready: $BANK_PATH"
echo "[pipeline] Starting 600k PPO fine-tuning..."

"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/parallel_run.py \
  --num_envs 64 \
  --seed 20261018 \
  --task_phase phase_4 \
  --goal_dataset_root /mnt/data/aiprah/data/random_sim_dataset_tcc \
  --pretrained_checkpoint artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_4_true_handoff_bc_replay_probe_400k.zip \
  --disable_obs_normalization \
  --action_masking \
  --action_mask_mode minimal \
  --phase_entry_empirical_bank "$BANK_PATH" \
  --phase_entry_empirical_probability 0.80 \
  --phase_entry_empirical_velocity \
  --bc_replay_dataset artifacts/datasets/bc/phase4_park_and_true_handoff_bc_adapt.npz \
  --bc_replay_coef 0.10 \
  --bc_replay_batch_size 256 \
  --bc_replay_decision_fraction 0.50 \
  --freeze_policy_timesteps 50000 \
  --learning_rate 3e-6 \
  --lr_schedule low_high_cosine \
  --freeze_learning_rate 2e-6 \
  --lr_ramp_timesteps 100000 \
  --lr_decay_start_fraction 0.75 \
  --final_learning_rate 3e-7 \
  --ent_coef 0.0001 \
  --ent_decay_start_fraction 0.75 \
  --final_ent_coef 0 \
  --gamma 0.999 \
  --gae_lambda 0.999 \
  --total_timesteps 600000 \
  --log_suffix empirical_handoff_bank80_replay_600k \
  --headless \
  > "$TRAIN_LOG" 2>&1

echo "[pipeline] PPO complete. Log: $TRAIN_LOG"
