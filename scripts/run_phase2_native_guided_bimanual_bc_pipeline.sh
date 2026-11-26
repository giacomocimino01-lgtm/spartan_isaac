#!/usr/bin/env bash
set -uo pipefail

# Validate and build a canonical bimanual Phase-2 BC under guided semantics.
# The expert concurrently reaches/grasps one red-stack and one blue-stack ring,
# then serializes the two centre releases.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
SMOKE_DATASET="artifacts/datasets/bc/phase2_native_guided_bimanual_smoke8.npz"
SOURCE_DATASET="artifacts/datasets/bc/dataset_supervisionato_phase2_native_guided_bimanual160.npz"
TRAIN_DATASET="artifacts/datasets/bc/phase2_native_guided_bimanual128_train.npz"
VALID_DATASET="artifacts/datasets/bc/phase2_native_guided_bimanual32_valid.npz"
BC_STEM="artifacts/checkpoints/bc/phase2_bc_native_guided_bimanual128"
BC_CHECKPOINT="${BC_STEM}.zip"
EVAL_DIR="artifacts/evaluations/phase2_bc_native_guided_bimanual128_50ep"
EVAL_CSV="$EVAL_DIR/episodes.csv"
JOB_DIR="artifacts/jobs/phase2_native_guided_bimanual_bc_pipeline"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/job.lock"
SMOKE_LOG="$REPO_ROOT/collect_phase2_native_guided_bimanual_smoke8.log"
COLLECT_LOG="$REPO_ROOT/collect_phase2_native_guided_bimanual160.log"
BUILD_LOG="$REPO_ROOT/build_phase2_native_guided_bimanual160.log"
BC_LOG="$REPO_ROOT/pretrain_phase2_native_guided_bimanual128.log"
TRAIN_CHECK_LOG="$REPO_ROOT/check_phase2_native_guided_bimanual_train.log"
VALID_CHECK_LOG="$REPO_ROOT/check_phase2_native_guided_bimanual_valid.log"
EVAL_LOG="$REPO_ROOT/eval_phase2_native_guided_bimanual128_50ep.log"

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
    echo "The native-guided bimanual Phase-2 BC pipeline is already running." >&2
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

collect() {
    local episodes="$1"
    local output="$2"
    local seed="$3"
    local log="$4"
    "$ISAACLAB_SH" -p scripts/collect_ppo_demos.py \
        --num_envs 8 \
        --num_episodes "$episodes" \
        --output_path "$output" \
        --task_phase phase_2 \
        --phase2_strategy bimanual \
        --action_mask_mode guided \
        --manipulability_scale 0 \
        --ring_order fixed \
        --arm_mode alternate \
        --decision_only \
        --seed "$seed" \
        --headless > "$log" 2>&1
}

write_status "SMOKE_TEST" "Testing eight successful bimanual scripted episodes"
if ! collect 8 "$SMOKE_DATASET" 6262 "$SMOKE_LOG"; then
    write_status "FAILED_SMOKE" "Bimanual expert failed; inspect $SMOKE_LOG"
    exit 1
fi
if ! "$ENV_BIN/python" - "$SMOKE_DATASET" >> "$SMOKE_LOG" 2>&1 <<'PY'
import numpy as np
import sys

data = np.load(sys.argv[1], allow_pickle=True)
metadata = data["metadata"].item()
print(f"[smoke] obs={data['obs'].shape} actions={data['actions'].shape}")
if data["obs"].shape != (8 * 12, 244):
    raise SystemExit("Expected exactly 12 semantic decisions per smoke episode")
if metadata.get("phase2_strategy") != "bimanual":
    raise SystemExit("Missing bimanual metadata")
if metadata.get("record_mode") != "decision_only":
    raise SystemExit("Smoke dataset is not decision-only")
PY
then
    write_status "FAILED_SMOKE_GATE" "Bimanual smoke dataset failed structural checks"
    exit 1
fi

write_status "COLLECTING" "Collecting 160 native-guided bimanual decision traces"
if ! collect 160 "$SOURCE_DATASET" 6363 "$COLLECT_LOG"; then
    write_status "FAILED_COLLECTION" "Inspect $COLLECT_LOG"
    exit 1
fi

write_status "SPLITTING" "Creating a disjoint 128/32 episode train/validation split"
if ! "$ENV_BIN/python" scripts/build_decision_bc_dataset.py \
    --source "$SOURCE_DATASET" \
    --train_output "$TRAIN_DATASET" \
    --valid_output "$VALID_DATASET" \
    --decisions_per_episode 12 \
    --validation_episodes 32 \
    --split_seed 20260922 > "$BUILD_LOG" 2>&1; then
    write_status "FAILED_SPLIT" "Inspect $BUILD_LOG"
    exit 1
fi

write_status "TRAINING_BC" "Training the canonical bimanual BC for 100 epochs"
if ! "$ISAACLAB_SH" -p scripts/pretrain_ppo.py \
    --dataset_path "$TRAIN_DATASET" \
    --output_path "$BC_STEM" \
    --epochs 100 \
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

write_status "EVALUATING_GUIDED" "Running 50 deterministic guided-mask bimanual episodes"
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
bimanual = sum(int(row["bimanual_steps"]) for row in rows)
rate = successes / len(rows) if rows else 0.0
print(
    f"[bimanual BC summary] train_exact={accuracy(sys.argv[1]):.3f}% "
    f"valid_exact={accuracy(sys.argv[2]):.3f}% "
    f"rollout_success={successes}/{len(rows)} ({rate:.1%}) "
    f"total_bimanual_steps={bimanual}"
)
PY

write_status "COMPLETED" "Canonical bimanual BC trained and evaluated; curriculum not started"
