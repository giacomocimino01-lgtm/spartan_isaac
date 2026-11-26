#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
BASE_CHECKPOINT="artifacts/evaluations/phase1_native_minimal_bc_stratified50_gamma0999_gae0999_probe_400k_rank_50ep/phase1_bc_stratified50_best_50ep.zip"
BASE_VECNORMALIZE="artifacts/evaluations/phase1_native_minimal_bc_stratified50_gamma0999_gae0999_probe_400k_rank_50ep/phase1_bc_stratified50_best_50ep_vecnormalize.pkl"
BASE_DATASET="artifacts/datasets/bc/dataset_supervisionato_phase1_native_minimal_160.npz"
RECOVERY_STATUS="artifacts/jobs/phase1_recovery_feasibility/status.txt"
JOB_DIR="artifacts/jobs/phase1_recovery_curriculum_probe"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/job.lock"
MERGED_DATASET="artifacts/datasets/bc/dataset_phase1_native_minimal_with_recovery_weighted.npz"
SUFFIX="native_minimal_recovery_mixed_replay_probe_300k"
TRAIN_LOG="$REPO_ROOT/train_phase1_${SUFFIX}.log"

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
    echo "The Phase-1 recovery curriculum probe is already running." >&2
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
        printf 'training_log=%s\n' "$TRAIN_LOG"
        printf 'merged_dataset=%s\n' "$MERGED_DATASET"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

for required in "$BASE_CHECKPOINT" "$BASE_VECNORMALIZE" "$BASE_DATASET" "$RECOVERY_STATUS"; do
    if [[ ! -f "$required" ]]; then
        write_status "FAILED_SETUP" "Missing required file: $required"
        exit 2
    fi
done

RECOVERY_DIR="$(awk -F= '$1=="run_dir" {print substr($0, index($0,"=")+1)}' "$RECOVERY_STATUS")"
RED3_DATASET="$RECOVERY_DIR/red3_blue1.npz"
BLUE3_DATASET="$RECOVERY_DIR/red1_blue3.npz"
for required in "$RED3_DATASET" "$BLUE3_DATASET"; do
    if [[ ! -s "$required" ]]; then
        write_status "FAILED_SETUP" "Missing recovery dataset: $required"
        exit 2
    fi
done

write_status "MERGING_REPLAY" "Weighting each recovery dataset eight times"
if ! "$ENV_BIN/python" scripts/merge_bc_datasets.py \
    --dataset "$BASE_DATASET" 1 \
    --dataset "$RED3_DATASET" 8 \
    --dataset "$BLUE3_DATASET" 8 \
    --output "$MERGED_DATASET" >> "$TRAIN_LOG" 2>&1; then
    write_status "FAILED_MERGE" "Inspect $TRAIN_LOG"
    exit 1
fi

write_status "TRAINING" "300k recovery-mixed PPO probe from the 52% checkpoint"
if ! "$ISAACLAB_SH" -p scripts/parallel_run.py \
    --num_envs 64 \
    --task_phase phase_1 \
    --phase1_initial_state recovery_mixed \
    --goal_dataset_root /mnt/data/aiprah/data/random_sim_dataset_tcc \
    --pretrained_checkpoint "$BASE_CHECKPOINT" \
    --pretrained_vecnormalize "$BASE_VECNORMALIZE" \
    --disable_obs_normalization \
    --action_masking \
    --action_mask_mode minimal \
    --bc_replay_dataset "$MERGED_DATASET" \
    --bc_replay_coef 0.1 \
    --bc_replay_batch_size 256 \
    --bc_replay_decision_fraction 0.5 \
    --gamma 0.999 \
    --gae_lambda 0.999 \
    --freeze_policy_timesteps 0 \
    --learning_rate 2e-6 \
    --lr_schedule constant \
    --ent_coef 0.0001 \
    --ent_decay_start_fraction 0.70 \
    --final_ent_coef 0 \
    --total_timesteps 300000 \
    --log_suffix "$SUFFIX" \
    --headless > "$TRAIN_LOG" 2>&1; then
    write_status "FAILED_TRAINING" "Inspect $TRAIN_LOG"
    exit 1
fi

write_status "COMPLETED" "Recovery-mixed probe completed; rank recovery and full-stack checkpoints next"
