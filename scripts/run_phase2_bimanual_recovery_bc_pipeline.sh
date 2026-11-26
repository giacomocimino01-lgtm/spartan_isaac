#!/usr/bin/env bash
set -uo pipefail

# Targeted Phase-2 recovery curriculum for the canonical guided bimanual BC.
# It preserves the 2+2 full-task demonstrations and oversamples physically
# valid late states with two or three rings already at board centre.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
NOMINAL_TRAIN="artifacts/datasets/bc/phase2_native_guided_bimanual128_train.npz"
NOMINAL_VALID="artifacts/datasets/bc/phase2_native_guided_bimanual32_valid.npz"
CENTER2_SOURCE="artifacts/datasets/bc/phase2_bimanual_recovery_center2_128.npz"
CENTER2_TRAIN="artifacts/datasets/bc/phase2_bimanual_recovery_center2_104_train.npz"
CENTER2_VALID="artifacts/datasets/bc/phase2_bimanual_recovery_center2_24_valid.npz"
CENTER3_SOURCE="artifacts/datasets/bc/phase2_bimanual_recovery_center3_128.npz"
CENTER3_TRAIN="artifacts/datasets/bc/phase2_bimanual_recovery_center3_104_train.npz"
CENTER3_VALID="artifacts/datasets/bc/phase2_bimanual_recovery_center3_24_valid.npz"
MIXED_TRAIN="artifacts/datasets/bc/phase2_bimanual_recovery_mixed_train.npz"
MIXED_VALID="artifacts/datasets/bc/phase2_bimanual_recovery_mixed_valid.npz"
BC_STEM="artifacts/checkpoints/bc/phase2_bc_native_guided_bimanual_recovery"
BC_CHECKPOINT="${BC_STEM}.zip"
EVAL_ROOT="artifacts/evaluations/phase2_bc_native_guided_bimanual_recovery"
JOB_DIR="artifacts/jobs/phase2_bimanual_recovery_bc_pipeline"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/job.lock"

mkdir -p "$JOB_DIR" "$EVAL_ROOT"
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
    echo "The Phase-2 bimanual recovery pipeline is already running." >&2
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
        printf 'full_evaluation=%s/full_50ep.csv\n' "$EVAL_ROOT"
        printf 'center2_evaluation=%s/center2_30ep.csv\n' "$EVAL_ROOT"
        printf 'center3_evaluation=%s/center3_30ep.csv\n' "$EVAL_ROOT"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

fail() {
    write_status "$1" "$2"
    exit 1
}

collect_recovery() {
    local initial_state="$1"
    local output="$2"
    local seed="$3"
    local log="$4"
    "$ISAACLAB_SH" -p scripts/collect_ppo_demos.py \
        --num_envs 8 --num_episodes 128 \
        --output_path "$output" \
        --task_phase phase_2 \
        --phase2_initial_state "$initial_state" \
        --phase2_strategy bimanual \
        --action_mask_mode guided \
        --manipulability_scale 0 \
        --ring_order fixed --arm_mode alternate \
        --decision_only --seed "$seed" --headless > "$log" 2>&1
}

split_recovery() {
    local source="$1"
    local train="$2"
    local valid="$3"
    local decisions="$4"
    local seed="$5"
    local log="$6"
    "$ENV_BIN/python" scripts/build_decision_bc_dataset.py \
        --source "$source" --train_output "$train" --valid_output "$valid" \
        --decisions_per_episode "$decisions" \
        --validation_episodes 24 --split_seed "$seed" > "$log" 2>&1
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

if [[ ! -f "$NOMINAL_TRAIN" || ! -f "$NOMINAL_VALID" ]]; then
    fail "FAILED_PREREQUISITE" "Canonical bimanual train/validation datasets are missing"
fi

write_status "COLLECTING_CENTER2" "Collecting 128 clean recoveries from two completed rings"
collect_recovery center2 "$CENTER2_SOURCE" 7202 \
    collect_phase2_bimanual_recovery_center2.log || \
    fail "FAILED_CENTER2_COLLECTION" "Inspect collect_phase2_bimanual_recovery_center2.log"

write_status "SPLITTING_CENTER2" "Creating disjoint 104/24 center2 splits"
split_recovery "$CENTER2_SOURCE" "$CENTER2_TRAIN" "$CENTER2_VALID" 6 20260923 \
    build_phase2_bimanual_recovery_center2.log || \
    fail "FAILED_CENTER2_SPLIT" "Inspect build_phase2_bimanual_recovery_center2.log"

write_status "COLLECTING_CENTER3" "Collecting 128 clean recoveries from three completed rings"
collect_recovery center3 "$CENTER3_SOURCE" 7303 \
    collect_phase2_bimanual_recovery_center3.log || \
    fail "FAILED_CENTER3_COLLECTION" "Inspect collect_phase2_bimanual_recovery_center3.log"

write_status "SPLITTING_CENTER3" "Creating disjoint 104/24 center3 splits"
split_recovery "$CENTER3_SOURCE" "$CENTER3_TRAIN" "$CENTER3_VALID" 4 20260924 \
    build_phase2_bimanual_recovery_center3.log || \
    fail "FAILED_CENTER3_SPLIT" "Inspect build_phase2_bimanual_recovery_center3.log"

write_status "MERGING" "Mixing nominal, center2 x2 and center3 x3 supervision"
"$ENV_BIN/python" scripts/merge_bc_datasets.py \
    --dataset "$NOMINAL_TRAIN" 1 \
    --dataset "$CENTER2_TRAIN" 2 \
    --dataset "$CENTER3_TRAIN" 3 \
    --output "$MIXED_TRAIN" > merge_phase2_bimanual_recovery_train.log 2>&1 || \
    fail "FAILED_TRAIN_MERGE" "Inspect merge_phase2_bimanual_recovery_train.log"
"$ENV_BIN/python" scripts/merge_bc_datasets.py \
    --dataset "$NOMINAL_VALID" 1 \
    --dataset "$CENTER2_VALID" 1 \
    --dataset "$CENTER3_VALID" 1 \
    --output "$MIXED_VALID" > merge_phase2_bimanual_recovery_valid.log 2>&1 || \
    fail "FAILED_VALID_MERGE" "Inspect merge_phase2_bimanual_recovery_valid.log"

write_status "TRAINING_BC" "Training recovery-aware bimanual BC for 120 epochs"
"$ISAACLAB_SH" -p scripts/pretrain_ppo.py \
    --dataset_path "$MIXED_TRAIN" --output_path "$BC_STEM" \
    --epochs 120 --batch_size 256 --lr 1e-3 --headless \
    > pretrain_phase2_bimanual_recovery.log 2>&1 || \
    fail "FAILED_BC" "Inspect pretrain_phase2_bimanual_recovery.log"

write_status "CHECKING_OFFLINE" "Checking mixed train and held-out validation accuracy"
"$ENV_BIN/python" scripts/check_bc_policy.py \
    --checkpoint "$BC_CHECKPOINT" --dataset_path "$MIXED_TRAIN" \
    --normalization raw --task_phase phase_2 --device cuda \
    > check_phase2_bimanual_recovery_train.log 2>&1 || \
    fail "FAILED_TRAIN_CHECK" "Inspect check_phase2_bimanual_recovery_train.log"
"$ENV_BIN/python" scripts/check_bc_policy.py \
    --checkpoint "$BC_CHECKPOINT" --dataset_path "$MIXED_VALID" \
    --normalization raw --task_phase phase_2 --device cuda \
    > check_phase2_bimanual_recovery_valid.log 2>&1 || \
    fail "FAILED_VALID_CHECK" "Inspect check_phase2_bimanual_recovery_valid.log"

write_status "EVALUATING_FULL" "Evaluating 50 canonical full-task episodes"
evaluate red2_blue2 50 "$EVAL_ROOT/full_50ep.csv" \
    eval_phase2_bimanual_recovery_full_50ep.log || \
    fail "FAILED_FULL_EVAL" "Inspect eval_phase2_bimanual_recovery_full_50ep.log"

write_status "EVALUATING_CENTER2" "Evaluating 30 held-out two-ring recoveries"
evaluate center2 30 "$EVAL_ROOT/center2_30ep.csv" \
    eval_phase2_bimanual_recovery_center2_30ep.log || \
    fail "FAILED_CENTER2_EVAL" "Inspect eval_phase2_bimanual_recovery_center2_30ep.log"

write_status "EVALUATING_CENTER3" "Evaluating 30 held-out one-ring recoveries"
evaluate center3 30 "$EVAL_ROOT/center3_30ep.csv" \
    eval_phase2_bimanual_recovery_center3_30ep.log || \
    fail "FAILED_CENTER3_EVAL" "Inspect eval_phase2_bimanual_recovery_center3_30ep.log"

"$ENV_BIN/python" - "$EVAL_ROOT" \
    > summarize_phase2_bimanual_recovery.log 2>&1 <<'PY'
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
    summary[path.stem] = {
        "successes": successes,
        "episodes": len(rows),
        "success_rate": successes / len(rows) if rows else 0.0,
        "mean_steps": sum(int(row["episode_steps"]) for row in rows) / len(rows),
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

write_status "COMPLETED" "Recovery-aware bimanual BC trained and evaluated; PPO not started"
