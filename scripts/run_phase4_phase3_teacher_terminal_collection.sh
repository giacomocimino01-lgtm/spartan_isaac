#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"
TCC_ROOT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs"
PHASE3_POLICY="artifacts/evaluations/phase3_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/phase3_bimanual_minimal_best_50ep.zip"

NUM_EPISODES="${NUM_EPISODES:-160}"
SEED="${SEED:-20261026}"
LABEL="phase4_phase3_teacher_terminal${NUM_EPISODES}_seed${SEED}"
DEMOS="artifacts/datasets/bc/${LABEL}_demos.npz"
BANK="artifacts/datasets/bc/${LABEL}_bank.npz"
STATUS="artifacts/jobs/${LABEL}.status"

cd "$REPO_ROOT"
source "$CONDA_SH"
conda activate env_isaaclab
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p artifacts/jobs artifacts/datasets/bc
printf 'COLLECTING\n' > "$STATUS"

"$ISAACLAB_SH" -p scripts/collect_phase4_true_handoff_demos.py \
    --num_envs 8 \
    --num_episodes "$NUM_EPISODES" \
    --output_path "$DEMOS" \
    --entry_state_bank_output "$BANK" \
    --phase3_source teacher \
    --phase3_checkpoint "$PHASE3_POLICY" \
    --phase3_tcc_checkpoint "$TCC_ROOT/random_sim_phase3_tcc/checkpoints/2600.ckpt" \
    --phase4_tcc_checkpoint "$TCC_ROOT/random_sim_phase4_tcc/checkpoints/2600.ckpt" \
    --dataset_root /mnt/data/aiprah/data/random_sim_dataset_tcc/train \
    --seed "$SEED" \
    --max_phase3_steps 900 \
    --max_phase4_steps 900 \
    --action_mask_mode minimal \
    --headless \
    --device cuda:0

printf 'COMPLETED\n' > "$STATUS"
echo "[teacher-terminal] Demos: $DEMOS"
echo "[teacher-terminal] Bank:  $BANK"

