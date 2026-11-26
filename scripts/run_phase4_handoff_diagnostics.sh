#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"
ENV_PYTHON="/home/aiprah/miniconda3/envs/env_isaaclab/bin/python"
TCC_ROOT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs"
DATA_ROOT="/mnt/data/aiprah/data/random_sim_dataset_tcc/train"

PHASE3_POLICY="artifacts/evaluations/phase3_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/phase3_bimanual_minimal_best_50ep.zip"
PHASE4_POLICY="artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_4_full_state_handoff_bank80_replay_600k.zip"
FULL_STATE_BANK="artifacts/datasets/bc/phase4_full_state_handoff_bank80_seed20261021.npz"
TRAINING_DEMOS="artifacts/datasets/bc/phase4_full_state_handoff80_seed20261021_demos.npz"
START_EMBEDDINGS="artifacts/evaluations/phase_router_phase01234_first_frame_20261007/start_embeddings.pt"

BANK_OUTPUT="artifacts/evaluations/phase4_fullstate_bank80_deterministic_best"
HANDOFF_OUTPUT="artifacts/evaluations/phase34_true_handoff_fullstate_best_stochastic_phase4_20ep_seed20261016"
ANALYSIS_OUTPUT="artifacts/evaluations/phase4_handoff_diagnostics_best"
STATUS_FILE="artifacts/jobs/phase4_handoff_diagnostics.status"

cd "$REPO_ROOT"
source "$CONDA_SH"
conda activate env_isaaclab
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p "$BANK_OUTPUT" "$HANDOFF_OUTPUT" "$ANALYSIS_OUTPUT" artifacts/jobs
printf 'RUNNING_BANK_EVAL\n' > "$STATUS_FILE"

"$ISAACLAB_SH" -p scripts/eval_run.py \
    --checkpoint "$PHASE4_POLICY" \
    --tcc_checkpoint "$TCC_ROOT/random_sim_phase4_tcc/checkpoints/2600.ckpt" \
    --dataset_path "$DATA_ROOT/phase_4" \
    --task_phase phase_4 \
    --episodes 80 \
    --seed 20261023 \
    --phase_entry_empirical_bank "$FULL_STATE_BANK" \
    --phase_entry_empirical_probability 1.0 \
    --phase_entry_empirical_sample_mode sequential \
    --phase_entry_empirical_velocity \
    --action_masking \
    --action_mask_mode minimal \
    --metrics_output "$BANK_OUTPUT/episodes.csv" \
    --disable_video \
    --quiet_steps \
    --headless \
    --device cuda:0

printf 'RUNNING_STOCHASTIC_HANDOFF_EVAL\n' > "$STATUS_FILE"

"$ISAACLAB_SH" -p scripts/eval_full_task.py \
    --phase phase_3 \
        "$PHASE3_POLICY" \
        "$TCC_ROOT/random_sim_phase3_tcc/checkpoints/2600.ckpt" \
        "$DATA_ROOT/phase_3" \
    --phase phase_4 \
        "$PHASE4_POLICY" \
        "$TCC_ROOT/random_sim_phase4_tcc/checkpoints/2600.ckpt" \
        "$DATA_ROOT/phase_4" \
    --start_embeddings "$START_EMBEDDINGS" \
    --sequence phase_3 phase_4 \
    --phase_success_mode symbolic \
    --episodes 20 \
    --seed 20261016 \
    --max_steps_per_phase 900 \
    --episode_length_s 240 \
    --action_mask_mode minimal \
    --stochastic_phase phase_4 \
    --output_dir "$HANDOFF_OUTPUT" \
    --disable_video \
    --headless \
    --device cuda:0

printf 'ANALYZING_ENTRY_COVERAGE\n' > "$STATUS_FILE"

"$ENV_PYTHON" evaluation/phase4_handoff_diagnostics.py \
    --training_demos "$TRAINING_DEMOS" \
    --evaluation_entries "$HANDOFF_OUTPUT/phase_entry_observations.npz" \
    --evaluation_episodes "$HANDOFF_OUTPUT/episodes.csv" \
    --bank_evaluation "$BANK_OUTPUT/episodes.csv" \
    --output_dir "$ANALYSIS_OUTPUT"

printf 'COMPLETED\n' > "$STATUS_FILE"
echo "[diagnostics] Complete."

