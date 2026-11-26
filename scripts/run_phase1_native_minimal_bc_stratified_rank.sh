#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
SUFFIX="native_minimal_bc_stratified50_gamma0999_gae0999_probe_400k"
BC_CHECKPOINT="artifacts/checkpoints/bc/phase1_bc_native_minimal_160.zip"
BEST_DISTANCE_CHECKPOINT="artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_1_${SUFFIX}.zip"
STEP_CHECKPOINT="artifacts/checkpoints/ppo/dvrk_ppo_phase_1_${SUFFIX}_384000_steps.zip"
FINAL_CHECKPOINT="artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_1_${SUFFIX}.zip"
RANK_DIR="artifacts/evaluations/phase1_${SUFFIX}_rank_50ep"
RANK_LOG="$REPO_ROOT/rank_phase1_${SUFFIX}_50ep.log"
JOB_DIR="$REPO_ROOT/artifacts/jobs/phase1_native_minimal_bc_stratified_probe"
STATUS_FILE="$JOB_DIR/status.txt"

cd "$REPO_ROOT"
source /home/aiprah/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PATH="$ENV_BIN:$PATH"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export IsaacLab_PATH=/home/aiprah/Documents/IsaacLab

write_status() {
    local state="$1"
    local detail="$2"
    local tmp="$STATUS_FILE.tmp"
    {
        printf 'updated_at=%s\n' "$(date -Is)"
        printf 'state=%s\n' "$state"
        printf 'detail=%s\n' "$detail"
        printf 'ranking_log=%s\n' "$RANK_LOG"
        printf 'ranking_dir=%s\n' "$RANK_DIR"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

for checkpoint in \
    "$BC_CHECKPOINT" \
    "$BEST_DISTANCE_CHECKPOINT" \
    "$STEP_CHECKPOINT" \
    "$FINAL_CHECKPOINT"; do
    if [[ ! -f "$checkpoint" ]]; then
        write_status "FAILED_RANKING" "Missing checkpoint: $checkpoint"
        exit 1
    fi
done

write_status "RANKING" "Deterministic 50-episode ranking of BC and three stratified-replay checkpoints"
if ! "$ENV_BIN/python" scripts/rank_checkpoints.py \
    --checkpoints \
    "$BC_CHECKPOINT" \
    "$BEST_DISTANCE_CHECKPOINT" \
    "$STEP_CHECKPOINT" \
    "$FINAL_CHECKPOINT" \
    --episodes 50 \
    --seed 42 \
    --task_phase phase_1 \
    --tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase1_tcc/checkpoints/31000.ckpt \
    --dataset_path /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_1 \
    --action_masking \
    --action_mask_mode minimal \
    --output_dir "$RANK_DIR" \
    --best_name phase1_bc_stratified50_best_50ep \
    --device cuda:0 > "$RANK_LOG" 2>&1; then
    write_status "FAILED_RANKING" "Inspect $RANK_LOG"
    exit 1
fi

write_status "COMPLETED" "Training report and deterministic ranking completed"
