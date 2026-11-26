#!/usr/bin/env bash
set -uo pipefail

# Phase-1 pipeline whose BC dataset is collected natively with fully-minimal
# observation semantics. Each expensive stage is protected by a health gate.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
DATASET="artifacts/datasets/bc/dataset_supervisionato_phase1_native_minimal_160.npz"
BC_STEM="artifacts/checkpoints/bc/phase1_bc_native_minimal_160"
BC_CHECKPOINT="${BC_STEM}.zip"
EVAL_DIR="artifacts/evaluations/phase1_bc_native_minimal_20ep"
EVAL_CSV="$EVAL_DIR/episodes.csv"
PPO_SUFFIX="native_minimal_bc_finetune_5m"
PPO_LOG_DIR="artifacts/logs/random_sb3_log_sim_phase_1_${PPO_SUFFIX}"
JOB_DIR="artifacts/jobs/phase1_native_minimal_pipeline"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/job.lock"

mkdir -p "$JOB_DIR" "$EVAL_DIR"
cd "$REPO_ROOT"
source /home/aiprah/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PATH="$ENV_BIN:$PATH"
export TERM=xterm-256color
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "The native-minimal Phase-1 pipeline is already running." >&2
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
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

write_status "COLLECTING" "Collecting 160 native-minimal successful demonstrations"
if ! "$ISAACLAB_SH" -p scripts/collect_ppo_demos.py \
    --num_envs 8 \
    --num_episodes 160 \
    --output_path "$DATASET" \
    --task_phase phase_1 \
    --action_mask_mode minimal \
    --manipulability_scale 1 \
    --ring_order random \
    --arm_mode random \
    --seed 4242 \
    --headless > collect_phase1_native_minimal_160.log 2>&1; then
    write_status "FAILED_COLLECTION" "Inspect collect_phase1_native_minimal_160.log"
    exit 1
fi

if ! "$ENV_BIN/python" - "$DATASET" <<'PY'
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
if obs.ndim != 2 or obs.shape[1] != 244 or len(obs) != len(actions):
    raise SystemExit("Invalid BC dataset shapes")
if metadata.get("task_phase") != "phase_1":
    raise SystemExit("Dataset phase metadata mismatch")
if metadata.get("action_mask_mode") != "minimal":
    raise SystemExit("Dataset was not collected with minimal observations")
if float(metadata.get("manipulability_scale", -1)) != 1.0:
    raise SystemExit("Dataset manipulability scale is not one")
PY
then
    write_status "FAILED_DATASET_GATE" "Collected dataset failed its metadata/shape gate"
    exit 1
fi

write_status "TRAINING_BC" "Training native-minimal behavior-cloning checkpoint"
if ! "$ISAACLAB_SH" -p scripts/pretrain_ppo.py \
    --dataset_path "$DATASET" \
    --output_path "$BC_STEM" \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-3 \
    --headless > pretrain_phase1_native_minimal_160.log 2>&1; then
    write_status "FAILED_BC" "Inspect pretrain_phase1_native_minimal_160.log"
    exit 1
fi

write_status "CHECKING_BC_OFFLINE" "Computing BC label accuracy"
if ! "$ENV_BIN/python" scripts/check_bc_policy.py \
    --checkpoint "$BC_CHECKPOINT" \
    --dataset_path "$DATASET" \
    --normalization raw \
    --task_phase phase_1 \
    --device cuda > check_phase1_native_minimal_bc.log 2>&1; then
    write_status "FAILED_BC_OFFLINE" "Inspect check_phase1_native_minimal_bc.log"
    exit 1
fi

write_status "EVALUATING_BC" "Running 20 deterministic minimal-mask episodes"
if ! "$ISAACLAB_SH" -p scripts/eval_run.py \
    --checkpoint "$BC_CHECKPOINT" \
    --tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase1_tcc/checkpoints/31000.ckpt \
    --dataset_path /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_1 \
    --task_phase phase_1 \
    --episodes 20 \
    --seed 42 \
    --metrics_output "$EVAL_CSV" \
    --disable_video \
    --quiet_steps \
    --action_masking \
    --action_mask_mode minimal \
    --headless > eval_phase1_native_minimal_bc_20ep.log 2>&1; then
    write_status "FAILED_BC_EVAL" "Inspect eval_phase1_native_minimal_bc_20ep.log"
    exit 1
fi

if ! "$ENV_BIN/python" - "$EVAL_CSV" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream))
successes = sum(str(row["success"]).lower() in {"true", "1", "yes"} for row in rows)
rate = successes / len(rows) if rows else 0.0
print(f"[BC eval] success={successes}/{len(rows)} ({rate:.1%})")
if len(rows) < 20 or rate < 0.60:
    raise SystemExit("BC simulation gate failed")
PY
then
    write_status "HALTED_BC_EVAL" "BC did not reach 60% deterministic success; PPO not started"
    exit 1
fi

write_status "TRAINING_PPO" "Training 5M PPO steps with native-minimal observations and mask"
if ! "$ISAACLAB_SH" -p scripts/parallel_run.py \
    --num_envs 64 \
    --task_phase phase_1 \
    --goal_dataset_root /mnt/data/aiprah/data/random_sim_dataset_tcc \
    --pretrained_checkpoint "$BC_CHECKPOINT" \
    --disable_obs_normalization \
    --action_masking \
    --action_mask_mode minimal \
    --freeze_policy_timesteps 500000 \
    --learning_rate 3e-5 \
    --lr_schedule low_high_cosine \
    --freeze_learning_rate 1e-5 \
    --lr_ramp_timesteps 250000 \
    --lr_decay_start_fraction 0.70 \
    --final_learning_rate 3e-6 \
    --ent_coef 0.001 \
    --ent_decay_start_fraction 0.70 \
    --final_ent_coef 0 \
    --total_timesteps 5000000 \
    --log_suffix "$PPO_SUFFIX" \
    --headless > train_phase1_${PPO_SUFFIX}.log 2>&1; then
    write_status "FAILED_PPO" "Inspect train_phase1_${PPO_SUFFIX}.log"
    exit 1
fi

write_status "REPORTING" "Generating native-minimal PPO report"
if ! "$ENV_BIN/python" evaluation/training_analysis.py --log_dir "$PPO_LOG_DIR" \
    >> train_phase1_${PPO_SUFFIX}.log 2>&1; then
    write_status "FAILED_REPORT" "PPO completed but report generation failed"
    exit 1
fi

if ! "$ENV_BIN/python" - "$PPO_LOG_DIR/episode_progress.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream))
def truth(value):
    return str(value).strip().lower() in {"true", "1", "yes"}
if len(rows) < 500:
    raise SystemExit("Insufficient PPO episodes")
last500 = rows[-500:]
last100 = rows[-100:]
success500 = sum(truth(row["is_success"]) for row in last500) / 500
success100 = sum(truth(row["is_success"]) for row in last100) / 100
timeout100 = sum(truth(row["time_limit_truncated"]) for row in last100) / 100
print(
    f"[PPO health] last500_success={success500:.3f} "
    f"last100_success={success100:.3f} last100_timeout={timeout100:.3f}"
)
if success500 < 0.75 or success100 < 0.70:
    raise SystemExit("Native-minimal PPO health gate failed")
PY
then
    write_status "HALTED_PPO_HEALTH" "PPO completed but robust-success gate failed"
    exit 1
fi

write_status "COMPLETED" "Native-minimal BC, evaluation, PPO, and report passed"
