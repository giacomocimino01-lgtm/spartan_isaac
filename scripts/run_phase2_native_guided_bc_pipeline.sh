#!/usr/bin/env bash
set -uo pipefail

# Build a deterministic Phase-2 BC natively under guided observation
# semantics.  This removes the random arm/order label ambiguity in the legacy
# guided demonstrations.  PPO is deliberately gated on 50 rollout episodes.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
SOURCE_DATASET="artifacts/datasets/bc/dataset_supervisionato_phase2_native_guided_fixed160.npz"
TRAIN_DATASET="artifacts/datasets/bc/phase2_native_guided_fixed128_train.npz"
VALID_DATASET="artifacts/datasets/bc/phase2_native_guided_fixed32_valid.npz"
BC_STEM="artifacts/checkpoints/bc/phase2_bc_native_guided_fixed128"
BC_CHECKPOINT="${BC_STEM}.zip"
EVAL_DIR="artifacts/evaluations/phase2_bc_native_guided_fixed128_50ep"
EVAL_CSV="$EVAL_DIR/episodes.csv"
JOB_DIR="artifacts/jobs/phase2_native_guided_bc_pipeline"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/job.lock"
COLLECT_LOG="$REPO_ROOT/collect_phase2_native_guided_fixed160.log"
BUILD_LOG="$REPO_ROOT/build_phase2_native_guided_fixed160.log"
BC_LOG="$REPO_ROOT/pretrain_phase2_native_guided_fixed128.log"
TRAIN_CHECK_LOG="$REPO_ROOT/check_phase2_native_guided_train.log"
VALID_CHECK_LOG="$REPO_ROOT/check_phase2_native_guided_valid.log"
EVAL_LOG="$REPO_ROOT/eval_phase2_native_guided_fixed128_50ep.log"

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
    echo "The native-guided Phase-2 BC pipeline is already running." >&2
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
        printf 'source_dataset=%s\n' "$SOURCE_DATASET"
        printf 'train_dataset=%s\n' "$TRAIN_DATASET"
        printf 'valid_dataset=%s\n' "$VALID_DATASET"
        printf 'bc_checkpoint=%s\n' "$BC_CHECKPOINT"
        printf 'evaluation_csv=%s\n' "$EVAL_CSV"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

write_status "COLLECTING" "Collecting 160 successful native-guided decision traces"
if ! "$ISAACLAB_SH" -p scripts/collect_ppo_demos.py \
    --num_envs 8 \
    --num_episodes 160 \
    --output_path "$SOURCE_DATASET" \
    --task_phase phase_2 \
    --action_mask_mode guided \
    --manipulability_scale 0 \
    --ring_order fixed \
    --arm_mode left \
    --decision_only \
    --seed 5252 \
    --headless > "$COLLECT_LOG" 2>&1; then
    write_status "FAILED_COLLECTION" "Inspect $COLLECT_LOG"
    exit 1
fi

write_status "SPLITTING" "Creating a disjoint 128/32 episode train/validation split"
if ! "$ENV_BIN/python" scripts/build_decision_bc_dataset.py \
    --source "$SOURCE_DATASET" \
    --train_output "$TRAIN_DATASET" \
    --valid_output "$VALID_DATASET" \
    --decisions_per_episode 16 \
    --validation_episodes 32 \
    --split_seed 20260922 > "$BUILD_LOG" 2>&1; then
    write_status "FAILED_SPLIT" "Inspect $BUILD_LOG"
    exit 1
fi

write_status "TRAINING_BC" "Training the deterministic guided BC for 80 epochs"
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

write_status "CHECKING_OFFLINE" "Measuring exact train and held-out validation accuracy"
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

write_status "EVALUATING_GUIDED" "Running the 50-episode deterministic guided-mask gate"
if ! "$ISAACLAB_SH" -p scripts/eval_run.py \
    --checkpoint "$BC_CHECKPOINT" \
    --tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase2_tcc/checkpoints/4000.ckpt \
    --dataset_path /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_2 \
    --task_phase phase_2 --episodes 50 --seed 42 \
    --metrics_output "$EVAL_CSV" \
    --disable_video --quiet_steps \
    --action_masking --action_mask_mode guided --headless > "$EVAL_LOG" 2>&1; then
    write_status "FAILED_GUIDED_EVAL" "Inspect $EVAL_LOG"
    exit 1
fi

if ! "$ENV_BIN/python" - "$TRAIN_CHECK_LOG" "$VALID_CHECK_LOG" "$EVAL_CSV" >> "$EVAL_LOG" 2>&1 <<'PY'
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
    f"[guided BC summary] train_exact={accuracy(sys.argv[1]):.3f}% "
    f"valid_exact={accuracy(sys.argv[2]):.3f}% "
    f"rollout_success={successes}/{len(rows)} ({rate:.1%})"
)
if len(rows) != 50 or rate < 0.90:
    raise SystemExit("Native-guided BC did not reach the 90% rollout gate")
PY
then
    write_status "HALTED_GUIDED_GATE" "New guided BC stayed below 90%; PPO was not started"
    exit 1
fi

write_status "COMPLETED" "Native-guided BC passed the 90% gate; ready for guided-to-minimal curriculum"
