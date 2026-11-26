#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
PPO_CHECKPOINT="artifacts/checkpoints/ppo/dvrk_ppo_phase_1_native_minimal_bc_finetune_5m_384000_steps.zip"
VECNORMALIZE_CHECKPOINT="artifacts/checkpoints/ppo/dvrk_ppo_phase_1_native_minimal_bc_finetune_5m_vecnormalize_384000_steps.pkl"
SUFFIX="native_minimal_conservative_resume384k_3m"
LOG_DIR="artifacts/logs/random_sb3_log_sim_phase_1_${SUFFIX}"
TRAIN_LOG="$REPO_ROOT/train_phase1_${SUFFIX}.log"
JOB_DIR="$REPO_ROOT/artifacts/jobs/phase1_native_minimal_ppo_conservative"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/job.lock"

mkdir -p "$JOB_DIR"
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
    echo "The conservative native-minimal Phase-1 PPO job is already running." >&2
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
        printf 'training_log=%s\n' "$TRAIN_LOG"
        printf 'checkpoint=%s\n' "$PPO_CHECKPOINT"
        printf 'vecnormalize=%s\n' "$VECNORMALIZE_CHECKPOINT"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

for required_file in "$PPO_CHECKPOINT" "$VECNORMALIZE_CHECKPOINT"; do
    if [[ ! -f "$required_file" ]]; then
        write_status "FAILED" "Missing resume artifact: $required_file"
        exit 2
    fi
done

write_status "TRAINING" "3M conservative PPO steps from the pre-unfreeze 384k checkpoint"
if ! "$ISAACLAB_SH" -p scripts/parallel_run.py \
    --num_envs 64 \
    --task_phase phase_1 \
    --goal_dataset_root /mnt/data/aiprah/data/random_sim_dataset_tcc \
    --pretrained_checkpoint "$PPO_CHECKPOINT" \
    --pretrained_vecnormalize "$VECNORMALIZE_CHECKPOINT" \
    --disable_obs_normalization \
    --action_masking \
    --action_mask_mode minimal \
    --freeze_policy_timesteps 0 \
    --learning_rate 5e-6 \
    --lr_schedule low_high_cosine \
    --freeze_learning_rate 1e-6 \
    --lr_ramp_timesteps 500000 \
    --lr_decay_start_fraction 0.60 \
    --final_learning_rate 5e-7 \
    --ent_coef 0.0001 \
    --ent_decay_start_fraction 0.60 \
    --final_ent_coef 0 \
    --total_timesteps 3000000 \
    --log_suffix "$SUFFIX" \
    --headless > "$TRAIN_LOG" 2>&1; then
    write_status "FAILED_TRAINING" "Inspect $TRAIN_LOG"
    exit 1
fi

write_status "REPORTING" "Generating conservative native-minimal PPO report"
if ! "$ENV_BIN/python" evaluation/training_analysis.py --log_dir "$LOG_DIR" \
    >> "$TRAIN_LOG" 2>&1; then
    write_status "FAILED_REPORT" "PPO completed but report generation failed"
    exit 1
fi

if ! "$ENV_BIN/python" - "$LOG_DIR/episode_progress.csv" >> "$TRAIN_LOG" 2>&1 <<'PY'
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
if success500 < 0.65 or success100 < 0.60:
    raise SystemExit("Conservative native-minimal PPO health gate failed")
PY
then
    write_status "HALTED_HEALTH" "PPO completed but robust-success gate failed"
    exit 1
fi

write_status "COMPLETED" "Conservative native-minimal PPO and report passed the health gate"
