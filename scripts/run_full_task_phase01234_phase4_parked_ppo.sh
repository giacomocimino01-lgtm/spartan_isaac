#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"
TCC_ROOT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs"
DATA_ROOT="/mnt/data/aiprah/data/random_sim_dataset_tcc/train"
OUTPUT_DIR="${1:-$REPO_ROOT/artifacts/evaluations/full_task_phase01234_phase4_parked_ppo_random075_seed42_20261014}"

cd "$REPO_ROOT"
source "$CONDA_SH"
conda activate env_isaaclab
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_DIR"

exec "$ISAACLAB_SH" -p scripts/eval_full_task.py \
    --phase phase_0 \
        artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_0_adaptive075_minimalmask_finetune_3m.zip \
        "$TCC_ROOT/random_sim_phase0_tcc/checkpoints/4001.ckpt" \
        "$DATA_ROOT/phase_0" \
    --phase phase_1 \
        artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_1_native_minimal_recovery_mixed_replay_probe_300k.zip \
        "$TCC_ROOT/random_sim_phase1_tcc/checkpoints/31000.ckpt" \
        "$DATA_ROOT/phase_1" \
    --phase phase_2 \
        artifacts/evaluations/phase2_native_minimal_full_probe_50ep/phase2_native_minimal_full_best_50ep.zip \
        "$TCC_ROOT/random_sim_phase2_tcc/checkpoints/4000.ckpt" \
        "$DATA_ROOT/phase_2" \
    --phase phase_3 \
        artifacts/evaluations/phase3_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/phase3_bimanual_minimal_best_50ep.zip \
        "$TCC_ROOT/random_sim_phase3_tcc/checkpoints/2600.ckpt" \
        "$DATA_ROOT/phase_3" \
    --phase phase_4 \
        artifacts/evaluations/phase4_park_neighborhood_scratchbc_replay_probe_400k/parked_exact_rank_50ep/phase4_parked_exact_best_50ep.zip \
        "$TCC_ROOT/random_sim_phase4_tcc/checkpoints/2600.ckpt" \
        "$DATA_ROOT/phase_4" \
    --start_embeddings \
        artifacts/evaluations/phase_router_phase01234_first_frame_20261007/start_embeddings.pt \
    --sequence phase_0 phase_1 phase_2 phase_3 phase_4 \
    --phase_success_mode tcc \
    --tcc_success_threshold phase_0 150 \
    --tcc_success_threshold phase_1 50 \
    --tcc_success_threshold phase_2 600 \
    --tcc_success_threshold phase_3 50 \
    --tcc_success_threshold phase_4 100 \
    --tcc_success_consecutive_steps 5 \
    --tcc_success_min_steps 20 \
    --episodes 1 \
    --seed 42 \
    --max_steps_per_phase 700 \
    --episode_length_s 240 \
    --randomize_rings \
    --ring_randomization_scale 0.75 \
    --action_mask_mode minimal \
    --output_dir "$OUTPUT_DIR" \
    --video_frame_stride 5 \
    --video_fps 15 \
    --headless \
    --device cuda:0
