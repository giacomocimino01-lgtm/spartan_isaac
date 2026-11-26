#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"
TCC_ROOT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs"
DATA_ROOT="/mnt/data/aiprah/data/random_sim_dataset_tcc/train"

EPISODES="${EPISODES:-10}"
SEED="${SEED:-20261014}"
PHASE4_CHECKPOINT="${PHASE4_CHECKPOINT:-artifacts/checkpoints/bc/phase4_bc_true_handoff_adapt10_seed42.zip}"
RUN_LABEL="${RUN_LABEL:-newbc}"
OUTPUT_DIR="${1:-$REPO_ROOT/artifacts/evaluations/phase34_true_handoff_${RUN_LABEL}_${EPISODES}ep_seed${SEED}}"

cd "$REPO_ROOT"
source "$CONDA_SH"
conda activate env_isaaclab
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_DIR"

exec "$ISAACLAB_SH" -p scripts/eval_full_task.py \
    --phase phase_3 \
        artifacts/evaluations/phase3_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/phase3_bimanual_minimal_best_50ep.zip \
        "$TCC_ROOT/random_sim_phase3_tcc/checkpoints/2600.ckpt" \
        "$DATA_ROOT/phase_3" \
    --phase phase_4 \
        "$PHASE4_CHECKPOINT" \
        "$TCC_ROOT/random_sim_phase4_tcc/checkpoints/2600.ckpt" \
        "$DATA_ROOT/phase_4" \
    --start_embeddings \
        artifacts/evaluations/phase_router_phase01234_first_frame_20261007/start_embeddings.pt \
    --sequence phase_3 phase_4 \
    --phase_success_mode symbolic \
    --episodes "$EPISODES" \
    --seed "$SEED" \
    --max_steps_per_phase 900 \
    --episode_length_s 240 \
    --action_mask_mode minimal \
    --output_dir "$OUTPUT_DIR" \
    --video_frame_stride 5 \
    --video_fps 15 \
    --headless \
    --device cuda:0
