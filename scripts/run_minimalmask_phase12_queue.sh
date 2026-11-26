#!/usr/bin/env bash
set -uo pipefail

# Sequential Phase-1/Phase-2 fine-tuning queue for the minimal physical mask.
# The queue is intentionally synchronous: Phase 2 starts only after Phase 1
# and its report have completed successfully.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
GOAL_DATASET_ROOT="/mnt/data/aiprah/data/random_sim_dataset_tcc"
TOTAL_TIMESTEPS=3000000

PHASE1_SUFFIX="minimalmask_finetune_3m"
PHASE2_SUFFIX="minimalmask_finetune_3m"
PHASE1_CHECKPOINT="artifacts/checkpoints/ppo/modelli_salvati_sim/dvrk_ppo_best_terminal_distance_phase_1_bc_raw_no_norm_lr3e-5_ent0.001_freeze500k_5m.zip"
PHASE2_CHECKPOINT="artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_2_relaxed_physicalmask_finetune_1p5m.zip"

JOB_DIR="$REPO_ROOT/artifacts/jobs/phase12_minimalmask_queue"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/queue.lock"
QUEUE_LOG="$REPO_ROOT/phase12_minimalmask_queue.log"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

mkdir -p "$JOB_DIR"
cd "$REPO_ROOT"
export PATH="$ENV_BIN:$PATH"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "A phase12 minimal-mask queue is already running." >&2
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
        printf 'queue_log=%s\n' "$QUEUE_LOG"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
    printf '[%s] %s: %s\n' "$(date -Is)" "$state" "$detail"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        write_status "FAILED" "Missing required file: $1"
        exit 2
    fi
}

require_file "$ISAACLAB_SH"
require_file "$PHASE1_CHECKPOINT"
require_file "$PHASE2_CHECKPOINT"
require_file "evaluation/training_analysis.py"

run_training() {
    local phase="$1"
    local suffix="$2"
    local checkpoint="$3"
    local train_log="$REPO_ROOT/train_${phase}_${suffix}.log"
    local log_dir="artifacts/logs/random_sb3_log_sim_${phase}_${suffix}"
    local -a command=(
        "$ISAACLAB_SH" -p scripts/parallel_run.py
        --num_envs 64
        --task_phase "$phase"
        --goal_dataset_root "$GOAL_DATASET_ROOT"
        --pretrained_checkpoint "$checkpoint"
        --disable_obs_normalization
        --action_masking
        --action_mask_mode minimal
        --freeze_policy_timesteps 0
        --learning_rate 3e-5
        --lr_schedule low_high_cosine
        --freeze_learning_rate 1e-5
        --lr_ramp_timesteps 250000
        --lr_decay_start_fraction 0.70
        --final_learning_rate 3e-6
        --ent_coef 0.001
        --ent_decay_start_fraction 0.70
        --final_ent_coef 0
        --total_timesteps "$TOTAL_TIMESTEPS"
        --log_suffix "$suffix"
        --headless
    )

    write_status "RUNNING_${phase}" "Training log: $train_log"
    if (( DRY_RUN )); then
        printf 'DRY RUN:'
        printf ' %q' "${command[@]}"
        printf '\n'
        return 0
    fi

    if ! "${command[@]}" > "$train_log" 2>&1; then
        write_status "FAILED_${phase}" "Training failed; inspect $train_log"
        return 1
    fi

    write_status "REPORTING_${phase}" "Generating analysis from $log_dir"
    if ! "$ENV_BIN/python" evaluation/training_analysis.py --log_dir "$log_dir" >> "$QUEUE_LOG" 2>&1; then
        write_status "FAILED_${phase}_REPORT" "Report generation failed for $log_dir"
        return 1
    fi
    write_status "COMPLETED_${phase}" "Training and report completed"
}

write_status "STARTED" "Phase 1 followed by Phase 2 on GPU 0"

if ! run_training "phase_1" "$PHASE1_SUFFIX" "$PHASE1_CHECKPOINT"; then
    exit 1
fi

if ! run_training "phase_2" "$PHASE2_SUFFIX" "$PHASE2_CHECKPOINT"; then
    exit 1
fi

write_status "COMPLETED" "Phase 1 and Phase 2 training reports are ready"
