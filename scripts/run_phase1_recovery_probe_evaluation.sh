#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
JOB_DIR="artifacts/jobs/phase1_recovery_probe_evaluation"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/job.lock"
LOG_FILE="$REPO_ROOT/phase1_recovery_probe_evaluation.log"
TCC_CHECKPOINT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase1_tcc/checkpoints/31000.ckpt"
DATASET_PATH="/mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_1"
BASE_CHECKPOINT="artifacts/evaluations/phase1_native_minimal_bc_stratified50_gamma0999_gae0999_probe_400k_rank_50ep/phase1_bc_stratified50_best_50ep.zip"
BEST_CHECKPOINT="artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_1_native_minimal_recovery_mixed_replay_probe_300k.zip"
FINAL_CHECKPOINT="artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_1_native_minimal_recovery_mixed_replay_probe_300k.zip"

mkdir -p "$JOB_DIR"
cd "$REPO_ROOT"
source /home/aiprah/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PATH="$ENV_BIN:$PATH"
export TERM=xterm-256color
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export IsaacLab_PATH=/home/aiprah/Documents/IsaacLab

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "The Phase-1 recovery probe evaluation is already running." >&2
    exit 3
fi

write_status() {
    local state="$1"
    local detail="$2"
    local tmp="$STATUS_FILE.tmp"
    {
        printf 'updated_at=%s\n' "$(date -Is)"
        printf 'state=%s\n' "$state"
        printf 'detail=%s\n' "$detail"
        printf 'log=%s\n' "$LOG_FILE"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

for required in "$TCC_CHECKPOINT" "$BASE_CHECKPOINT" "$BEST_CHECKPOINT" "$FINAL_CHECKPOINT"; do
    if [[ ! -f "$required" ]]; then
        write_status "FAILED_SETUP" "Missing required file: $required"
        exit 2
    fi
done
if [[ ! -d "$DATASET_PATH" ]]; then
    write_status "FAILED_SETUP" "Missing dataset directory: $DATASET_PATH"
    exit 2
fi

CHECKPOINTS=("$BASE_CHECKPOINT" "$BEST_CHECKPOINT" "$FINAL_CHECKPOINT")

run_ranking() {
    local state="$1"
    local initial_state="$2"
    local episodes="$3"
    local output_dir="$4"
    local best_name="$5"

    write_status "$state" "initial_state=$initial_state episodes=$episodes"
    "$ENV_BIN/python" scripts/rank_checkpoints.py \
        --checkpoints "${CHECKPOINTS[@]}" \
        --episodes "$episodes" \
        --seed 42 \
        --task_phase phase_1 \
        --phase1_initial_state "$initial_state" \
        --tcc_checkpoint "$TCC_CHECKPOINT" \
        --dataset_path "$DATASET_PATH" \
        --action_masking \
        --action_mask_mode minimal \
        --output_dir "$output_dir" \
        --best_name "$best_name" \
        --device cuda:0 >> "$LOG_FILE" 2>&1
}

: > "$LOG_FILE"
if ! run_ranking \
    "EVALUATING_RED3_BLUE1" \
    "red3_blue1" \
    20 \
    "artifacts/evaluations/phase1_recovery_probe_red3_blue1_20ep" \
    "phase1_recovery_probe_red3_blue1_best"; then
    write_status "FAILED_RED3_BLUE1" "Inspect $LOG_FILE"
    exit 1
fi

if ! run_ranking \
    "EVALUATING_RED1_BLUE3" \
    "red1_blue3" \
    20 \
    "artifacts/evaluations/phase1_recovery_probe_red1_blue3_20ep" \
    "phase1_recovery_probe_red1_blue3_best"; then
    write_status "FAILED_RED1_BLUE3" "Inspect $LOG_FILE"
    exit 1
fi

if ! run_ranking \
    "EVALUATING_STACK_GREEN" \
    "stack_green" \
    50 \
    "artifacts/evaluations/phase1_recovery_probe_stack_green_50ep" \
    "phase1_recovery_probe_stack_green_best"; then
    write_status "FAILED_STACK_GREEN" "Inspect $LOG_FILE"
    exit 1
fi

write_status "COMPLETED" "All deterministic recovery and full-task evaluations completed"
