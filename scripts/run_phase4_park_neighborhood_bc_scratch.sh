#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_PYTHON="/home/aiprah/miniconda3/envs/env_isaaclab/bin/python"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"
TCC_DATA_ROOT="/mnt/data/aiprah/data/random_sim_dataset_tcc"
TCC_CHECKPOINT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase4_tcc/checkpoints/2600.ckpt"

DATA_DIR="$REPO_ROOT/artifacts/datasets/bc"
JOB_DIR="$REPO_ROOT/artifacts/jobs/phase4_park_neighborhood_bc_scratch80"
EVAL_DIR="$REPO_ROOT/artifacts/evaluations/phase4_park_neighborhood_bc_scratch80"
STATUS="$JOB_DIR/status.txt"

TRAIN_DATASET="$DATA_DIR/phase4_park_neighborhood_bc_adapt.npz"
EXACT_VALID="$DATA_DIR/phase4_park_exact16_valid.npz"
NEAR_VALID="$DATA_DIR/phase4_park_neighborhood16_valid.npz"
EXACT_DECISION_VALID="$DATA_DIR/phase4_park_exact16_decision_valid.npz"
NEAR_DECISION_VALID="$DATA_DIR/phase4_park_neighborhood16_decision_valid.npz"

BC_STEM="$REPO_ROOT/artifacts/checkpoints/bc/phase4_bc_park_neighborhood_scratch80_seed42"
BC_CHECKPOINT="${BC_STEM}.zip"

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
        printf 'initialization=random\n'
        printf 'epochs=80\n'
        printf 'batch_size=256\n'
        printf 'learning_rate=1e-3\n'
        printf 'seed=42\n'
        printf 'train_dataset=%s\n' "$TRAIN_DATASET"
        printf 'bc_checkpoint=%s\n' "$BC_CHECKPOINT"
        printf 'evaluation_dir=%s\n' "$EVAL_DIR"
    } > "$STATUS"
}

fail() {
    write_status "$1" "$2"
    printf '[phase4-park-bc-scratch] %s: %s\n' "$1" "$2" >&2
    exit 1
}

for required in \
    "$TRAIN_DATASET" "$EXACT_VALID" "$NEAR_VALID" \
    "$EXACT_DECISION_VALID" "$NEAR_DECISION_VALID" "$TCC_CHECKPOINT"; do
    [[ -f "$required" ]] || fail "MISSING_INPUT" "$required"
done

write_status "TRAINING_BC" "Training a new Phase-4 actor from random initialization"
"$ISAACLAB_SH" -p scripts/pretrain_ppo.py \
    --dataset_path "$TRAIN_DATASET" --output_path "$BC_STEM" \
    --task_phase phase_4 --action_mask_mode minimal \
    --epochs 80 --batch_size 256 --lr 1e-3 --seed 42 --headless \
    > "$JOB_DIR/train_bc.log" 2>&1 || \
    fail "FAILED_BC" "Inspect $JOB_DIR/train_bc.log"

write_status "CHECKING_OFFLINE" "Checking held-out exact and neighborhood datasets"
for spec in \
    "exact_dense:$EXACT_VALID" \
    "near_dense:$NEAR_VALID" \
    "exact_decision:$EXACT_DECISION_VALID" \
    "near_decision:$NEAR_DECISION_VALID"; do
    name="${spec%%:*}"
    dataset="${spec#*:}"
    "$ENV_PYTHON" scripts/check_bc_policy.py \
        --checkpoint "$BC_CHECKPOINT" --dataset_path "$dataset" \
        --normalization raw --task_phase phase_4 --device cuda \
        > "$JOB_DIR/check_${name}.log" 2>&1 || \
        fail "FAILED_OFFLINE_CHECK" "$name"
done

run_eval() {
    local name="$1"
    shift
    "$ISAACLAB_SH" -p scripts/eval_run.py \
        --checkpoint "$BC_CHECKPOINT" --tcc_checkpoint "$TCC_CHECKPOINT" \
        --dataset_path "$TCC_DATA_ROOT/train/phase_4" --task_phase phase_4 \
        --episodes 30 --seed 42 --metrics_output "$EVAL_DIR/${name}_30ep.csv" \
        --disable_video --quiet_steps --action_masking --action_mask_mode minimal \
        --headless --device cuda:0 "$@" > "$JOB_DIR/eval_${name}.log" 2>&1 || \
        fail "FAILED_EVALUATION" "$name"
}

write_status "EVALUATING_EXACT" "Evaluating the scratch BC on exact parked starts"
run_eval parked_exact --phase_entry_park_reset
write_status "EVALUATING_NEIGHBORHOOD" "Evaluating the scratch BC throughout the parked neighborhood"
run_eval parked_neighborhood --phase_entry_park_reset \
    --phase_entry_park_neighborhood_probability 1.0 \
    --phase_entry_park_neighborhood_scale 1.0
write_status "EVALUATING_CANONICAL" "Measuring out-of-distribution canonical performance"
run_eval canonical

"$ENV_PYTHON" - \
    "$EVAL_DIR/parked_exact_30ep.csv" \
    "$EVAL_DIR/parked_neighborhood_30ep.csv" \
    "$EVAL_DIR/canonical_30ep.csv" <<'PY' > "$JOB_DIR/evaluation_summary.log"
import csv
import sys

def summarize(path):
    rows = list(csv.DictReader(open(path, newline="")))
    successes = sum(row["success"].lower() == "true" for row in rows)
    return successes, len(rows)

for name, path in zip(
    ("parked_exact", "parked_neighborhood", "canonical"),
    sys.argv[1:],
    strict=True,
):
    successes, episodes = summarize(path)
    print(f"{name}={successes}/{episodes} ({successes / episodes:.1%})")
PY

write_status "COMPLETED" "Scratch BC training, offline checks and three online evaluations completed"
