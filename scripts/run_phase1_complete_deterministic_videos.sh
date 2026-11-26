#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
CHECKPOINT="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_1_native_minimal_recovery_mixed_replay_probe_300k.zip"
TCC_CHECKPOINT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase1_tcc/checkpoints/31000.ckpt"
DATASET_PATH="/mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_1"
OUTPUT_DIR="$REPO_ROOT/artifacts/evaluations/phase1_complete_deterministic_videos"
JOB_DIR="$REPO_ROOT/artifacts/jobs/phase1_complete_deterministic_videos"
STATUS_FILE="$JOB_DIR/status.txt"
LOG_FILE="$REPO_ROOT/phase1_complete_deterministic_videos.log"

mkdir -p "$OUTPUT_DIR" "$JOB_DIR"
cd "$REPO_ROOT"
source /home/aiprah/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PATH="/home/aiprah/miniconda3/envs/env_isaaclab/bin:$PATH"
export TERM=xterm-256color
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
        printf 'log=%s\n' "$LOG_FILE"
        printf 'output_dir=%s\n' "$OUTPUT_DIR"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

for required in "$ISAACLAB_SH" "$CHECKPOINT" "$TCC_CHECKPOINT"; do
    if [[ ! -f "$required" ]]; then
        write_status "FAILED_SETUP" "Missing required file: $required"
        exit 2
    fi
done
if [[ ! -d "$DATASET_PATH" ]]; then
    write_status "FAILED_SETUP" "Missing dataset directory: $DATASET_PATH"
    exit 2
fi

write_status "RUNNING" "Recording 3 consecutive deterministic Phase-1 episodes"
: > "$LOG_FILE"
cd "$OUTPUT_DIR"
if ! "$ISAACLAB_SH" -p "$REPO_ROOT/scripts/eval_run.py" \
    --checkpoint "$CHECKPOINT" \
    --tcc_checkpoint "$TCC_CHECKPOINT" \
    --dataset_path "$DATASET_PATH" \
    --task_phase phase_1 \
    --phase1_initial_state stack_green \
    --episodes 3 \
    --seed 20260923 \
    --metrics_output "$OUTPUT_DIR/episodes.csv" \
    --action_masking \
    --action_mask_mode minimal \
    --quiet_steps \
    --headless > "$LOG_FILE" 2>&1; then
    write_status "FAILED_EVALUATION" "Inspect $LOG_FILE"
    exit 1
fi

video_count="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
write_status "COMPLETED" "Recorded $video_count videos and deterministic episode metrics"
