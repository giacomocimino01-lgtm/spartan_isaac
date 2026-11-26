#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_PYTHON="/home/aiprah/miniconda3/envs/env_isaaclab/bin/python"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"
TCC_DATA_ROOT="/mnt/data/aiprah/data/random_sim_dataset_tcc"
TCC_CHECKPOINT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase4_tcc/checkpoints/2600.ckpt"

DATA_DIR="$REPO_ROOT/artifacts/datasets/bc"
JOB_DIR="$REPO_ROOT/artifacts/jobs/phase4_parked_bc_adaptation"
EVAL_DIR="$REPO_ROOT/artifacts/evaluations/phase4_parked_bc_adaptation"
STATUS="$JOB_DIR/status.txt"

CANONICAL_SOURCE="$DATA_DIR/phase4_bimanual_minimal160.npz"
CANONICAL_DENSE_TRAIN="$DATA_DIR/phase4_bimanual_minimal128_train.npz"
CANONICAL_DENSE_VALID="$DATA_DIR/phase4_bimanual_minimal32_valid.npz"
PARKED_SOURCE="$DATA_DIR/phase4_bimanual_minimal_parked80.npz"
PARKED_DENSE_TRAIN="$DATA_DIR/phase4_bimanual_minimal_parked64_train.npz"
PARKED_DENSE_VALID="$DATA_DIR/phase4_bimanual_minimal_parked16_valid.npz"

CANONICAL_DECISION_TRAIN="$DATA_DIR/phase4_bimanual_minimal128_decision_train.npz"
CANONICAL_DECISION_VALID="$DATA_DIR/phase4_bimanual_minimal32_decision_valid.npz"
PARKED_DECISION_TRAIN="$DATA_DIR/phase4_bimanual_minimal_parked64_decision_train.npz"
PARKED_DECISION_VALID="$DATA_DIR/phase4_bimanual_minimal_parked16_decision_valid.npz"
ADAPT_DATASET="$DATA_DIR/phase4_bimanual_minimal_parked_bc_adapt_balanced.npz"

BASE_CHECKPOINT="$REPO_ROOT/artifacts/evaluations/phase4_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/phase4_bimanual_minimal_best_50ep.zip"
ADAPTED_STEM="$REPO_ROOT/artifacts/checkpoints/bc/phase4_bc_parked_adapt_from_canonical98"
ADAPTED_CHECKPOINT="${ADAPTED_STEM}.zip"

mkdir -p "$JOB_DIR" "$EVAL_DIR"
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
        printf 'base_checkpoint=%s\n' "$BASE_CHECKPOINT"
        printf 'adapt_dataset=%s\n' "$ADAPT_DATASET"
        printf 'adapted_checkpoint=%s\n' "$ADAPTED_CHECKPOINT"
        printf 'evaluation_dir=%s\n' "$EVAL_DIR"
    } > "$STATUS"
}

fail() {
    write_status "$1" "$2"
    printf '[phase4-parked-bc] %s: %s\n' "$1" "$2" >&2
    exit 1
}

for required in \
    "$CANONICAL_SOURCE" "$CANONICAL_DENSE_TRAIN" "$CANONICAL_DENSE_VALID" \
    "$PARKED_SOURCE" "$PARKED_DENSE_TRAIN" "$PARKED_DENSE_VALID" \
    "$BASE_CHECKPOINT"; do
    [[ -f "$required" ]] || fail "MISSING_INPUT" "$required"
done

write_status "PREPARING_DECISIONS" "Extracting episode-disjoint semantic decision datasets"
"$ENV_PYTHON" scripts/build_decision_bc_dataset.py \
    --source "$CANONICAL_SOURCE" \
    --train_output "$CANONICAL_DECISION_TRAIN" \
    --valid_output "$CANONICAL_DECISION_VALID" \
    --decisions_per_episode 0 --validation_episodes 32 \
    --split_seed 20261006 > "$JOB_DIR/build_canonical_decisions.log" 2>&1 || \
    fail "FAILED_CANONICAL_DECISIONS" "Inspect build_canonical_decisions.log"

"$ENV_PYTHON" scripts/build_decision_bc_dataset.py \
    --source "$PARKED_SOURCE" \
    --train_output "$PARKED_DECISION_TRAIN" \
    --valid_output "$PARKED_DECISION_VALID" \
    --decisions_per_episode 0 --validation_episodes 16 \
    --split_seed 20261007 > "$JOB_DIR/build_parked_decisions.log" 2>&1 || \
    fail "FAILED_PARKED_DECISIONS" "Inspect build_parked_decisions.log"

# Equalise the two physical start distributions and strongly expose semantic
# decision boundaries that are otherwise only ~4% of dense low-level rows.
"$ENV_PYTHON" scripts/merge_bc_datasets.py \
    --dataset "$CANONICAL_DENSE_TRAIN" 1 \
    --dataset "$PARKED_DENSE_TRAIN" 2 \
    --dataset "$CANONICAL_DECISION_TRAIN" 10 \
    --dataset "$PARKED_DECISION_TRAIN" 20 \
    --output "$ADAPT_DATASET" > "$JOB_DIR/merge_adapt.log" 2>&1 || \
    fail "FAILED_MERGE" "Inspect merge_adapt.log"

PYTHONPATH=source "$ENV_PYTHON" scripts/check_bc_dataset_mask.py \
    --dataset "$ADAPT_DATASET" --task_phase phase_4 \
    --action_mask_mode minimal > "$JOB_DIR/check_adapt_mask.log" 2>&1 || \
    fail "FAILED_MASK_CHECK" "Adaptation labels violate the minimal mask"

write_status "TRAINING_BC" "Fine-tuning canonical 98% actor on balanced canonical/parked supervision"
"$ISAACLAB_SH" -p scripts/pretrain_ppo.py \
    --dataset_path "$ADAPT_DATASET" --output_path "$ADAPTED_STEM" \
    --initial_checkpoint "$BASE_CHECKPOINT" \
    --task_phase phase_4 --action_mask_mode minimal \
    --epochs 10 --batch_size 256 --lr 1e-5 --seed 42 --headless \
    > "$JOB_DIR/train_bc.log" 2>&1 || \
    fail "FAILED_BC" "Inspect train_bc.log"

write_status "CHECKING_OFFLINE" "Measuring dense and decision-only held-out accuracy"
for spec in \
    "canonical_dense:$CANONICAL_DENSE_VALID" \
    "parked_dense:$PARKED_DENSE_VALID" \
    "canonical_decision:$CANONICAL_DECISION_VALID" \
    "parked_decision:$PARKED_DECISION_VALID"; do
    name="${spec%%:*}"
    dataset="${spec#*:}"
    "$ENV_PYTHON" scripts/check_bc_policy.py \
        --checkpoint "$ADAPTED_CHECKPOINT" --dataset_path "$dataset" \
        --normalization raw --task_phase phase_4 --device cuda \
        > "$JOB_DIR/check_${name}.log" 2>&1 || \
        fail "FAILED_OFFLINE_CHECK" "$name"
done

write_status "EVALUATING_PARKED_GATE" "Running 20 deterministic parked episodes"
"$ISAACLAB_SH" -p scripts/eval_run.py \
    --checkpoint "$ADAPTED_CHECKPOINT" --tcc_checkpoint "$TCC_CHECKPOINT" \
    --dataset_path "$TCC_DATA_ROOT/train/phase_4" --task_phase phase_4 \
    --phase_entry_park_reset --episodes 20 --seed 42 \
    --metrics_output "$EVAL_DIR/parked_gate_20ep.csv" \
    --disable_video --quiet_steps --action_masking --action_mask_mode minimal \
    --headless --device cuda:0 > "$JOB_DIR/eval_parked_gate.log" 2>&1 || \
    fail "FAILED_PARKED_GATE_EVAL" "Inspect eval_parked_gate.log"

write_status "EVALUATING_CANONICAL_GATE" "Running 20 deterministic canonical episodes"
"$ISAACLAB_SH" -p scripts/eval_run.py \
    --checkpoint "$ADAPTED_CHECKPOINT" --tcc_checkpoint "$TCC_CHECKPOINT" \
    --dataset_path "$TCC_DATA_ROOT/train/phase_4" --task_phase phase_4 \
    --episodes 20 --seed 42 \
    --metrics_output "$EVAL_DIR/canonical_gate_20ep.csv" \
    --disable_video --quiet_steps --action_masking --action_mask_mode minimal \
    --headless --device cuda:0 > "$JOB_DIR/eval_canonical_gate.log" 2>&1 || \
    fail "FAILED_CANONICAL_GATE_EVAL" "Inspect eval_canonical_gate.log"

"$ENV_PYTHON" - "$EVAL_DIR/parked_gate_20ep.csv" "$EVAL_DIR/canonical_gate_20ep.csv" <<'PY' \
    > "$JOB_DIR/gate_summary.log" 2>&1 || \
    fail "HALTED_BC_GATE" "BC must reach >=80% parked and >=90% canonical before PPO"
import csv
import sys

def rate(path):
    rows = list(csv.DictReader(open(path, newline="")))
    return sum(row["success"].lower() == "true" for row in rows) / len(rows), len(rows)

parked, parked_n = rate(sys.argv[1])
canonical, canonical_n = rate(sys.argv[2])
print(f"parked={parked:.1%} ({parked_n} episodes)")
print(f"canonical={canonical:.1%} ({canonical_n} episodes)")
if parked_n != 20 or canonical_n != 20 or parked < 0.80 or canonical < 0.90:
    raise SystemExit(1)
PY

write_status "COMPLETED_BC_GATE" "BC passed parked and canonical gates; PPO intentionally not started"
