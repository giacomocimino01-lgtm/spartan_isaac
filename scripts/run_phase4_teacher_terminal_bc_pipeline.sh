#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_PYTHON="/home/aiprah/miniconda3/envs/env_isaaclab/bin/python"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"

SOURCE_DATASET="$REPO_ROOT/artifacts/datasets/bc/phase4_phase3_teacher_terminal160_seed20261026_demos.npz"
TRAIN_DATASET="$REPO_ROOT/artifacts/datasets/bc/phase4_phase3_teacher_terminal128_train_seed20261027.npz"
VALID_DATASET="$REPO_ROOT/artifacts/datasets/bc/phase4_phase3_teacher_terminal32_valid_seed20261027.npz"
BC_STEM="$REPO_ROOT/artifacts/checkpoints/bc/phase4_bc_phase3_teacher_terminal128_80ep_seed42"
BC_CHECKPOINT="${BC_STEM}.zip"

JOB_DIR="$REPO_ROOT/artifacts/jobs/phase4_teacher_terminal_bc80_seed42"
EVAL_DIR="$REPO_ROOT/artifacts/evaluations/phase34_true_handoff_teacherterminal_bc80_20ep_seed20261027"
STATUS="$JOB_DIR/status.txt"

mkdir -p "$JOB_DIR" "$EVAL_DIR" "$(dirname "$BC_STEM")"
cd "$REPO_ROOT"
source "$CONDA_SH"
conda activate env_isaaclab
export IsaacLab_PATH="/home/aiprah/Documents/IsaacLab"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

write_status() {
    {
        printf 'updated_at=%s\n' "$(date -Is)"
        printf 'state=%s\n' "$1"
        printf 'detail=%s\n' "$2"
        printf 'source_dataset=%s\n' "$SOURCE_DATASET"
        printf 'train_dataset=%s\n' "$TRAIN_DATASET"
        printf 'valid_dataset=%s\n' "$VALID_DATASET"
        printf 'bc_checkpoint=%s\n' "$BC_CHECKPOINT"
        printf 'evaluation_dir=%s\n' "$EVAL_DIR"
        printf 'initialization=random\n'
        printf 'epochs=80\n'
        printf 'batch_size=256\n'
        printf 'learning_rate=1e-3\n'
        printf 'action_mask_mode=minimal\n'
    } > "$STATUS"
}

fail() {
    write_status "$1" "$2"
    printf '[phase4-teacher-terminal-bc] %s: %s\n' "$1" "$2" >&2
    exit 1
}

[[ -f "$SOURCE_DATASET" ]] || fail "MISSING_INPUT" "$SOURCE_DATASET"

if [[ ! -f "$TRAIN_DATASET" || ! -f "$VALID_DATASET" ]]; then
    write_status "SPLITTING" "Creating episode-disjoint 128/32 train-validation split"
    "$ENV_PYTHON" scripts/split_episode_bc_dataset.py \
        --source "$SOURCE_DATASET" \
        --train_output "$TRAIN_DATASET" \
        --valid_output "$VALID_DATASET" \
        --validation_episodes 32 \
        --split_seed 20261027 \
        > "$JOB_DIR/split.log" 2>&1 || \
        fail "FAILED_SPLIT" "Inspect $JOB_DIR/split.log"
fi

write_status "CHECKING_DATASET" "Checking minimal-mask compatibility"
PYTHONPATH=source "$ENV_PYTHON" scripts/check_bc_dataset_mask.py \
    --dataset "$SOURCE_DATASET" \
    --task_phase phase_4 \
    --action_mask_mode minimal \
    > "$JOB_DIR/check_mask.log" 2>&1 || \
    fail "FAILED_MASK_CHECK" "Expert labels violate the minimal mask"

if [[ ! -f "$BC_CHECKPOINT" ]]; then
    write_status "TRAINING_BC" "Training Phase-4 BC from scratch on 128 teacher-terminal episodes"
    "$ISAACLAB_SH" -p scripts/pretrain_ppo.py \
        --dataset_path "$TRAIN_DATASET" \
        --output_path "$BC_STEM" \
        --task_phase phase_4 \
        --action_mask_mode minimal \
        --epochs 80 \
        --batch_size 256 \
        --lr 1e-3 \
        --seed 42 \
        --headless \
        > "$JOB_DIR/train_bc.log" 2>&1 || \
        fail "FAILED_BC" "Inspect $JOB_DIR/train_bc.log"
fi

write_status "CHECKING_OFFLINE" "Measuring train and held-out exact-action accuracy"
for spec in \
    "train:$TRAIN_DATASET" \
    "valid:$VALID_DATASET"; do
    name="${spec%%:*}"
    dataset="${spec#*:}"
    "$ENV_PYTHON" scripts/check_bc_policy.py \
        --checkpoint "$BC_CHECKPOINT" \
        --dataset_path "$dataset" \
        --normalization raw \
        --task_phase phase_4 \
        --device cuda \
        > "$JOB_DIR/check_${name}.log" 2>&1 || \
        fail "FAILED_OFFLINE_CHECK" "$name"
done

write_status "EVALUATING_TRUE_HANDOFF" "Running 20 deterministic Phase-3 to Phase-4 episodes with videos"
EPISODES=20 \
SEED=20261027 \
PHASE4_CHECKPOINT="$BC_CHECKPOINT" \
RUN_LABEL="teacherterminal_bc80" \
    bash scripts/run_phase34_true_handoff_bc_eval.sh "$EVAL_DIR" \
    > "$JOB_DIR/eval_true_handoff.log" 2>&1 || \
    fail "FAILED_TRUE_HANDOFF_EVAL" "Inspect $JOB_DIR/eval_true_handoff.log"

"$ENV_PYTHON" - "$EVAL_DIR/summary.json" <<'PY' > "$JOB_DIR/evaluation_summary.log"
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    summary = json.load(stream)
print(json.dumps(summary, indent=2))
PY

write_status "COMPLETED" "BC training, offline checks and 20-episode true-handoff evaluation completed"

