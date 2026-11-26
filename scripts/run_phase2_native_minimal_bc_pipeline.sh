#!/usr/bin/env bash
set -uo pipefail

# Build and validate a Phase-2 BC policy using observations collected with the
# exact fully-minimal semantics used at inference.  PPO is deliberately not
# started here: the deterministic BC gate must pass first.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
DATASET="artifacts/datasets/bc/dataset_supervisionato_phase2_native_minimal_fixed160.npz"
BC_STEM="artifacts/checkpoints/bc/phase2_bc_native_minimal_fixed160"
BC_CHECKPOINT="${BC_STEM}.zip"
EVAL_DIR="artifacts/evaluations/phase2_bc_native_minimal_fixed160_50ep"
EVAL_CSV="$EVAL_DIR/episodes.csv"
JOB_DIR="artifacts/jobs/phase2_native_minimal_bc_pipeline"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/job.lock"
COLLECT_LOG="$REPO_ROOT/collect_phase2_native_minimal_fixed160.log"
BC_LOG="$REPO_ROOT/pretrain_phase2_native_minimal_fixed160.log"
CHECK_LOG="$REPO_ROOT/check_phase2_native_minimal_fixed160.log"
EVAL_LOG="$REPO_ROOT/eval_phase2_native_minimal_fixed160_50ep.log"

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
    echo "The native-minimal Phase-2 BC pipeline is already running." >&2
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
        printf 'dataset=%s\n' "$DATASET"
        printf 'bc_checkpoint=%s\n' "$BC_CHECKPOINT"
        printf 'evaluation_csv=%s\n' "$EVAL_CSV"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

write_status "COLLECTING" "Collecting 160 deterministic native-minimal Phase-2 demonstrations"
if ! "$ISAACLAB_SH" -p scripts/collect_ppo_demos.py \
    --num_envs 8 \
    --num_episodes 160 \
    --output_path "$DATASET" \
    --task_phase phase_2 \
    --action_mask_mode minimal \
    --manipulability_scale 1 \
    --ring_order fixed \
    --arm_mode left \
    --seed 4242 \
    --headless > "$COLLECT_LOG" 2>&1; then
    write_status "FAILED_COLLECTION" "Inspect $COLLECT_LOG"
    exit 1
fi

write_status "CHECKING_DATASET" "Validating native-minimal metadata and tensor shapes"
if ! "$ENV_BIN/python" - "$DATASET" >> "$COLLECT_LOG" 2>&1 <<'PY'
import json
import sys
import numpy as np

path = sys.argv[1]
data = np.load(path, allow_pickle=True)
obs = data["obs"]
actions = data["actions"]
metadata = data["metadata"].item()
print(f"[dataset] observations={obs.shape} actions={actions.shape}")
print(f"[dataset] metadata={json.dumps(metadata, sort_keys=True)}")
if obs.ndim != 2 or obs.shape[1] != 244 or actions.shape != (len(obs), 4):
    raise SystemExit("Invalid BC dataset shapes")
if len(obs) < 2500:
    raise SystemExit("Too few expert decision transitions")
expected = {
    "task_phase": "phase_2",
    "action_mask_mode": "minimal",
    "ring_order": "fixed",
    "arm_mode": "left",
}
for key, value in expected.items():
    if metadata.get(key) != value:
        raise SystemExit(f"Dataset metadata mismatch for {key}: {metadata.get(key)!r}")
if float(metadata.get("manipulability_scale", -1)) != 1.0:
    raise SystemExit("Dataset manipulability scale is not one")
PY
then
    write_status "FAILED_DATASET_GATE" "Collected dataset failed metadata/shape validation"
    exit 1
fi

write_status "TRAINING_BC" "Training deterministic Phase-2 BC for 80 epochs"
if ! "$ISAACLAB_SH" -p scripts/pretrain_ppo.py \
    --dataset_path "$DATASET" \
    --output_path "$BC_STEM" \
    --epochs 80 \
    --batch_size 256 \
    --lr 1e-3 \
    --headless > "$BC_LOG" 2>&1; then
    write_status "FAILED_BC" "Inspect $BC_LOG"
    exit 1
fi

write_status "CHECKING_BC_OFFLINE" "Requiring at least 99.5% exact joint-action accuracy"
if ! "$ENV_BIN/python" scripts/check_bc_policy.py \
    --checkpoint "$BC_CHECKPOINT" \
    --dataset_path "$DATASET" \
    --normalization raw \
    --task_phase phase_2 \
    --device cuda > "$CHECK_LOG" 2>&1; then
    write_status "FAILED_BC_OFFLINE" "Inspect $CHECK_LOG"
    exit 1
fi

if ! "$ENV_BIN/python" - "$CHECK_LOG" >> "$CHECK_LOG" 2>&1 <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
matches = re.findall(r"exact action accuracy=([0-9.]+)%", text)
if not matches:
    raise SystemExit("Exact-action accuracy was not found in the BC report")
accuracy = float(matches[-1])
print(f"[gate] exact_joint_action_accuracy={accuracy:.3f}%")
if accuracy < 99.5:
    raise SystemExit("BC exact-action accuracy is below 99.5%")
PY
then
    write_status "HALTED_BC_ACCURACY" "BC stayed below 99.5%; simulation evaluation was not started"
    exit 1
fi

write_status "EVALUATING_BC" "Running 50 deterministic episodes with a fully-minimal zero-penalty mask"
if ! "$ISAACLAB_SH" -p scripts/eval_run.py \
    --checkpoint "$BC_CHECKPOINT" \
    --tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase2_tcc/checkpoints/4000.ckpt \
    --dataset_path /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_2 \
    --task_phase phase_2 \
    --episodes 50 \
    --seed 42 \
    --metrics_output "$EVAL_CSV" \
    --disable_video \
    --quiet_steps \
    --action_masking \
    --action_mask_mode minimal \
    --headless > "$EVAL_LOG" 2>&1; then
    write_status "FAILED_BC_EVAL" "Inspect $EVAL_LOG"
    exit 1
fi

if ! "$ENV_BIN/python" - "$EVAL_CSV" >> "$EVAL_LOG" 2>&1 <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream))
successes = sum(str(row["success"]).lower() in {"true", "1", "yes"} for row in rows)
rate = successes / len(rows) if rows else 0.0
mean_steps = (
    sum(int(row["episode_steps"]) for row in rows if str(row["success"]).lower() in {"true", "1", "yes"})
    / successes
    if successes else float("nan")
)
print(f"[BC eval] success={successes}/{len(rows)} ({rate:.1%}) mean_success_steps={mean_steps:.1f}")
if len(rows) < 50 or rate < 0.90:
    raise SystemExit("BC deterministic simulation gate failed")
PY
then
    write_status "HALTED_BC_EVAL" "BC did not reach 90% deterministic success; PPO was not started"
    exit 1
fi

write_status "COMPLETED" "Native-minimal Phase-2 BC passed offline and 50-episode simulation gates"
