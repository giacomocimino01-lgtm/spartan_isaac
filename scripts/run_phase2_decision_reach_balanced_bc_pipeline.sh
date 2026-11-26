#!/usr/bin/env bash
set -uo pipefail

# Phase-2 decision-only BC with extra sampling weight on the four decisions
# that select the next ring. Nothing else differs from the unbalanced run.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
SOURCE_DATASET="artifacts/datasets/bc/dataset_supervisionato_phase2_native_minimal_fixed160.npz"
TRAIN_DATASET="artifacts/datasets/bc/phase2_decision_reach4x_fixed128_train.npz"
VALID_DATASET="artifacts/datasets/bc/phase2_decision_reach4x_fixed32_valid.npz"
BC_STEM="artifacts/checkpoints/bc/phase2_bc_decision_reach4x_fixed128"
BC_CHECKPOINT="${BC_STEM}.zip"
EVAL_DIR="artifacts/evaluations/phase2_bc_decision_reach4x_fixed128_50ep"
EVAL_CSV="$EVAL_DIR/episodes.csv"
JOB_DIR="artifacts/jobs/phase2_decision_reach_balanced_bc_pipeline"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/job.lock"
BUILD_LOG="$REPO_ROOT/build_phase2_decision_reach4x.log"
BC_LOG="$REPO_ROOT/pretrain_phase2_decision_reach4x_fixed128.log"
TRAIN_CHECK_LOG="$REPO_ROOT/check_phase2_decision_reach4x_train.log"
VALID_CHECK_LOG="$REPO_ROOT/check_phase2_decision_reach4x_valid.log"
EVAL_LOG="$REPO_ROOT/eval_phase2_decision_reach4x_fixed128_50ep.log"

mkdir -p "$JOB_DIR" "$EVAL_DIR"
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
    echo "The Phase-2 reach-balanced BC pipeline is already running." >&2
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
        printf 'train_dataset=%s\n' "$TRAIN_DATASET"
        printf 'valid_dataset=%s\n' "$VALID_DATASET"
        printf 'bc_checkpoint=%s\n' "$BC_CHECKPOINT"
        printf 'evaluation_csv=%s\n' "$EVAL_CSV"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

write_status "BUILDING_DATASET" "Repeating only reach-to-ring decisions 4x in the training split"
if ! "$ENV_BIN/python" scripts/build_decision_bc_dataset.py \
    --source "$SOURCE_DATASET" \
    --train_output "$TRAIN_DATASET" \
    --valid_output "$VALID_DATASET" \
    --decisions_per_episode 16 \
    --validation_episodes 32 \
    --split_seed 20260921 \
    --ring_reach_repeat 4 > "$BUILD_LOG" 2>&1; then
    write_status "FAILED_DATASET" "Inspect $BUILD_LOG"
    exit 1
fi

write_status "TRAINING_BC" "Training the reach-balanced BC with unchanged hyperparameters"
if ! "$ISAACLAB_SH" -p scripts/pretrain_ppo.py \
    --dataset_path "$TRAIN_DATASET" \
    --output_path "$BC_STEM" \
    --epochs 80 \
    --batch_size 256 \
    --lr 1e-3 \
    --headless > "$BC_LOG" 2>&1; then
    write_status "FAILED_BC" "Inspect $BC_LOG"
    exit 1
fi

write_status "CHECKING_OFFLINE" "Measuring train and held-out validation exact-action accuracy"
if ! "$ENV_BIN/python" scripts/check_bc_policy.py \
    --checkpoint "$BC_CHECKPOINT" --dataset_path "$TRAIN_DATASET" \
    --normalization raw --task_phase phase_2 --device cuda > "$TRAIN_CHECK_LOG" 2>&1; then
    write_status "FAILED_TRAIN_CHECK" "Inspect $TRAIN_CHECK_LOG"
    exit 1
fi
if ! "$ENV_BIN/python" scripts/check_bc_policy.py \
    --checkpoint "$BC_CHECKPOINT" --dataset_path "$VALID_DATASET" \
    --normalization raw --task_phase phase_2 --device cuda > "$VALID_CHECK_LOG" 2>&1; then
    write_status "FAILED_VALID_CHECK" "Inspect $VALID_CHECK_LOG"
    exit 1
fi

write_status "EVALUATING_BC" "Running 50 deterministic episodes with the fully-minimal mask"
if ! "$ISAACLAB_SH" -p scripts/eval_run.py \
    --checkpoint "$BC_CHECKPOINT" \
    --tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase2_tcc/checkpoints/4000.ckpt \
    --dataset_path /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_2 \
    --task_phase phase_2 --episodes 50 --seed 42 \
    --metrics_output "$EVAL_CSV" --disable_video --quiet_steps \
    --action_masking --action_mask_mode minimal --headless > "$EVAL_LOG" 2>&1; then
    write_status "FAILED_BC_EVAL" "Inspect $EVAL_LOG"
    exit 1
fi

"$ENV_BIN/python" - "$TRAIN_CHECK_LOG" "$VALID_CHECK_LOG" "$EVAL_CSV" >> "$EVAL_LOG" 2>&1 <<'PY'
import csv
import re
import sys

def accuracy(path):
    text = open(path, encoding="utf-8").read()
    matches = re.findall(r"exact action accuracy=([0-9.]+)%", text)
    return float(matches[-1]) if matches else float("nan")

with open(sys.argv[3], newline="") as stream:
    rows = list(csv.DictReader(stream))
successes = sum(str(row["success"]).lower() in {"true", "1", "yes"} for row in rows)
rate = successes / len(rows) if rows else 0.0
print(
    f"[summary] train_exact={accuracy(sys.argv[1]):.3f}% "
    f"valid_exact={accuracy(sys.argv[2]):.3f}% "
    f"rollout_success={successes}/{len(rows)} ({rate:.1%})"
)
PY

write_status "COMPLETED" "Reach-balanced minimal-mask BC evaluated; PPO was not started"
