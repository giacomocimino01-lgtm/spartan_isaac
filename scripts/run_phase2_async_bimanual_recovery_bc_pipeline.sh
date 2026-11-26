#!/usr/bin/env bash
set -uo pipefail

# Rebuild Phase-2 BC with an asynchronous bimanual expert.  Every dataset is
# rejected before training unless all expert labels are legal under the same
# guided action mask used by PPO and evaluation.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
DATA_DIR="artifacts/datasets/bc"
NOMINAL_SOURCE="$DATA_DIR/phase2_async_bimanual_full160.npz"
NOMINAL_TRAIN="$DATA_DIR/phase2_async_bimanual_full128_train.npz"
NOMINAL_VALID="$DATA_DIR/phase2_async_bimanual_full32_valid.npz"
CENTER2_SOURCE="$DATA_DIR/phase2_async_bimanual_center2_128.npz"
CENTER2_TRAIN="$DATA_DIR/phase2_async_bimanual_center2_104_train.npz"
CENTER2_VALID="$DATA_DIR/phase2_async_bimanual_center2_24_valid.npz"
CENTER3_SOURCE="$DATA_DIR/phase2_async_bimanual_center3_128.npz"
CENTER3_TRAIN="$DATA_DIR/phase2_async_bimanual_center3_104_train.npz"
CENTER3_VALID="$DATA_DIR/phase2_async_bimanual_center3_24_valid.npz"
MIXED_TRAIN="$DATA_DIR/phase2_async_bimanual_recovery_mixed_train.npz"
MIXED_VALID="$DATA_DIR/phase2_async_bimanual_recovery_mixed_valid.npz"
SMOKE_DATASET="$DATA_DIR/phase2_async_bimanual_smoke8.npz"
BC_STEM="artifacts/checkpoints/bc/phase2_bc_async_bimanual_recovery"
BC_CHECKPOINT="${BC_STEM}.zip"
EVAL_ROOT="artifacts/evaluations/phase2_bc_async_bimanual_recovery"
JOB_DIR="artifacts/jobs/phase2_async_bimanual_recovery_bc_pipeline"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/job.lock"

mkdir -p "$JOB_DIR" "$EVAL_ROOT" "$DATA_DIR"
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
    echo "The asynchronous Phase-2 BC pipeline is already running." >&2
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
        printf 'mixed_train=%s\n' "$MIXED_TRAIN"
        printf 'mixed_valid=%s\n' "$MIXED_VALID"
        printf 'bc_checkpoint=%s\n' "$BC_CHECKPOINT"
        printf 'evaluation_root=%s\n' "$EVAL_ROOT"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

fail() {
    write_status "$1" "$2"
    exit 1
}

collect() {
    local initial_state="$1"
    local episodes="$2"
    local output="$3"
    local seed="$4"
    local log="$5"
    "$ISAACLAB_SH" -p scripts/collect_ppo_demos.py \
        --num_envs 8 --num_episodes "$episodes" \
        --output_path "$output" \
        --task_phase phase_2 --phase2_initial_state "$initial_state" \
        --phase2_strategy bimanual --action_mask_mode guided \
        --manipulability_scale 0 --ring_order fixed --arm_mode alternate \
        --decision_only --seed "$seed" --headless > "$log" 2>&1
}

split() {
    local source="$1"
    local train="$2"
    local valid="$3"
    local valid_episodes="$4"
    local seed="$5"
    local log="$6"
    "$ENV_BIN/python" scripts/split_episode_bc_dataset.py \
        --source "$source" --train_output "$train" --valid_output "$valid" \
        --validation_episodes "$valid_episodes" --split_seed "$seed" \
        > "$log" 2>&1
}

mask_check() {
    local dataset="$1"
    local log="$2"
    PYTHONPATH=source "$ENV_BIN/python" scripts/check_bc_dataset_mask.py \
        --dataset "$dataset" --task_phase phase_2 \
        --action_mask_mode guided > "$log" 2>&1
}

evaluate() {
    local initial_state="$1"
    local episodes="$2"
    local csv="$3"
    local log="$4"
    "$ISAACLAB_SH" -p scripts/eval_run.py \
        --checkpoint "$BC_CHECKPOINT" \
        --tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase2_tcc/checkpoints/4000.ckpt \
        --dataset_path /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_2 \
        --task_phase phase_2 --phase2_initial_state "$initial_state" \
        --episodes "$episodes" --seed 42 --metrics_output "$csv" \
        --disable_video --quiet_steps --action_masking \
        --action_mask_mode guided --headless > "$log" 2>&1
}

write_status "SMOKE_TEST" "Collecting eight real asynchronous bimanual episodes"
collect red2_blue2 8 "$SMOKE_DATASET" 8108 \
    collect_phase2_async_bimanual_smoke8.log || \
    fail "FAILED_SMOKE" "Inspect collect_phase2_async_bimanual_smoke8.log"
mask_check "$SMOKE_DATASET" check_phase2_async_bimanual_smoke8_mask.log || \
    fail "FAILED_SMOKE_MASK" "At least one asynchronous expert label is masked"

write_status "COLLECTING_FULL" "Collecting 160 complete asynchronous bimanual demonstrations"
collect red2_blue2 160 "$NOMINAL_SOURCE" 8160 \
    collect_phase2_async_bimanual_full160.log || \
    fail "FAILED_FULL_COLLECTION" "Inspect collect_phase2_async_bimanual_full160.log"
split "$NOMINAL_SOURCE" "$NOMINAL_TRAIN" "$NOMINAL_VALID" 32 20261001 \
    split_phase2_async_bimanual_full160.log || \
    fail "FAILED_FULL_SPLIT" "Inspect split_phase2_async_bimanual_full160.log"

write_status "COLLECTING_CENTER2" "Collecting 128 asynchronous center2 recoveries"
collect center2 128 "$CENTER2_SOURCE" 8228 \
    collect_phase2_async_bimanual_center2_128.log || \
    fail "FAILED_CENTER2_COLLECTION" "Inspect collect_phase2_async_bimanual_center2_128.log"
split "$CENTER2_SOURCE" "$CENTER2_TRAIN" "$CENTER2_VALID" 24 20261002 \
    split_phase2_async_bimanual_center2_128.log || \
    fail "FAILED_CENTER2_SPLIT" "Inspect split_phase2_async_bimanual_center2_128.log"

write_status "COLLECTING_CENTER3" "Collecting 128 one-ring recoveries"
collect center3 128 "$CENTER3_SOURCE" 8328 \
    collect_phase2_async_bimanual_center3_128.log || \
    fail "FAILED_CENTER3_COLLECTION" "Inspect collect_phase2_async_bimanual_center3_128.log"
split "$CENTER3_SOURCE" "$CENTER3_TRAIN" "$CENTER3_VALID" 24 20261003 \
    split_phase2_async_bimanual_center3_128.log || \
    fail "FAILED_CENTER3_SPLIT" "Inspect split_phase2_async_bimanual_center3_128.log"

write_status "MERGING" "Mixing full x1, center2 x2 and center3 x3 supervision"
"$ENV_BIN/python" scripts/merge_bc_datasets.py \
    --dataset "$NOMINAL_TRAIN" 1 \
    --dataset "$CENTER2_TRAIN" 2 \
    --dataset "$CENTER3_TRAIN" 3 \
    --output "$MIXED_TRAIN" > merge_phase2_async_bimanual_train.log 2>&1 || \
    fail "FAILED_TRAIN_MERGE" "Inspect merge_phase2_async_bimanual_train.log"
"$ENV_BIN/python" scripts/merge_bc_datasets.py \
    --dataset "$NOMINAL_VALID" 1 \
    --dataset "$CENTER2_VALID" 1 \
    --dataset "$CENTER3_VALID" 1 \
    --output "$MIXED_VALID" > merge_phase2_async_bimanual_valid.log 2>&1 || \
    fail "FAILED_VALID_MERGE" "Inspect merge_phase2_async_bimanual_valid.log"

write_status "CHECKING_MASK" "Requiring every train and validation label to be mask-compatible"
mask_check "$MIXED_TRAIN" check_phase2_async_bimanual_train_mask.log || \
    fail "FAILED_TRAIN_MASK" "Training labels are not 100% mask-compatible"
mask_check "$MIXED_VALID" check_phase2_async_bimanual_valid_mask.log || \
    fail "FAILED_VALID_MASK" "Validation labels are not 100% mask-compatible"

write_status "TRAINING_BC" "Training mask-compatible asynchronous bimanual BC for 120 epochs"
"$ISAACLAB_SH" -p scripts/pretrain_ppo.py \
    --dataset_path "$MIXED_TRAIN" --output_path "$BC_STEM" \
    --epochs 120 --batch_size 256 --lr 1e-3 --headless \
    > pretrain_phase2_async_bimanual_recovery.log 2>&1 || \
    fail "FAILED_BC" "Inspect pretrain_phase2_async_bimanual_recovery.log"

write_status "CHECKING_OFFLINE" "Measuring exact train and held-out validation accuracy"
"$ENV_BIN/python" scripts/check_bc_policy.py \
    --checkpoint "$BC_CHECKPOINT" --dataset_path "$MIXED_TRAIN" \
    --normalization raw --task_phase phase_2 --device cuda \
    > check_phase2_async_bimanual_train.log 2>&1 || \
    fail "FAILED_TRAIN_CHECK" "Inspect check_phase2_async_bimanual_train.log"
"$ENV_BIN/python" scripts/check_bc_policy.py \
    --checkpoint "$BC_CHECKPOINT" --dataset_path "$MIXED_VALID" \
    --normalization raw --task_phase phase_2 --device cuda \
    > check_phase2_async_bimanual_valid.log 2>&1 || \
    fail "FAILED_VALID_CHECK" "Inspect check_phase2_async_bimanual_valid.log"

write_status "EVALUATING_FULL" "Evaluating 50 complete Phase-2 episodes"
evaluate red2_blue2 50 "$EVAL_ROOT/full_50ep.csv" \
    eval_phase2_async_bimanual_full_50ep.log || \
    fail "FAILED_FULL_EVAL" "Inspect eval_phase2_async_bimanual_full_50ep.log"

write_status "EVALUATING_CENTER2" "Evaluating 30 center2 recoveries"
evaluate center2 30 "$EVAL_ROOT/center2_30ep.csv" \
    eval_phase2_async_bimanual_center2_30ep.log || \
    fail "FAILED_CENTER2_EVAL" "Inspect eval_phase2_async_bimanual_center2_30ep.log"

write_status "EVALUATING_CENTER3" "Evaluating 30 center3 recoveries"
evaluate center3 30 "$EVAL_ROOT/center3_30ep.csv" \
    eval_phase2_async_bimanual_center3_30ep.log || \
    fail "FAILED_CENTER3_EVAL" "Inspect eval_phase2_async_bimanual_center3_30ep.log"

"$ENV_BIN/python" - "$EVAL_ROOT" > summarize_phase2_async_bimanual.log 2>&1 <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for path in sorted(root.glob("*.csv")):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    successes = sum(row["success"].lower() in {"true", "1", "yes"} for row in rows)
    success_rows = [row for row in rows if row["success"].lower() in {"true", "1", "yes"}]
    summary[path.stem] = {
        "successes": successes,
        "episodes": len(rows),
        "success_rate": successes / len(rows) if rows else 0.0,
        "mean_success_steps": (
            sum(int(row["episode_steps"]) for row in success_rows) / len(success_rows)
            if success_rows else None
        ),
        "mean_bimanual_steps": (
            sum(int(row["bimanual_steps"]) for row in rows) / len(rows)
            if rows else 0.0
        ),
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

write_status "COMPLETED" "Mask-compatible asynchronous bimanual BC trained and evaluated; PPO not started"
