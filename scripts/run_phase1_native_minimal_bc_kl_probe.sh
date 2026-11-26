#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
BC_CHECKPOINT="artifacts/checkpoints/bc/phase1_bc_native_minimal_160.zip"
SUFFIX="native_minimal_bc_kl01_gamma0999_gae0999_probe_700k"
LOG_DIR="artifacts/logs/random_sb3_log_sim_phase_1_${SUFFIX}"
TRAIN_LOG="$REPO_ROOT/train_phase1_${SUFFIX}.log"
JOB_DIR="$REPO_ROOT/artifacts/jobs/phase1_native_minimal_bc_kl_probe"
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
    echo "The Phase-1 BC-KL probe is already running." >&2
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
        printf 'pretrained_checkpoint=%s\n' "$BC_CHECKPOINT"
        printf 'kl_anchor_checkpoint=%s\n' "$BC_CHECKPOINT"
        printf 'kl_anchor_coefficient=0.1\n'
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

if [[ ! -f "$BC_CHECKPOINT" ]]; then
    write_status "FAILED" "Missing BC checkpoint: $BC_CHECKPOINT"
    exit 2
fi

write_status "TRAINING" "700k BC-anchored probe; actor frozen through 150k"
if ! "$ISAACLAB_SH" -p scripts/parallel_run.py \
    --num_envs 64 \
    --task_phase phase_1 \
    --goal_dataset_root /mnt/data/aiprah/data/random_sim_dataset_tcc \
    --pretrained_checkpoint "$BC_CHECKPOINT" \
    --disable_obs_normalization \
    --action_masking \
    --action_mask_mode minimal \
    --randomize_rings \
    --ring_randomization_scale 1.0 \
    --kl_anchor_checkpoint "$BC_CHECKPOINT" \
    --kl_anchor_coef 0.1 \
    --gamma 0.999 \
    --gae_lambda 0.999 \
    --freeze_policy_timesteps 150000 \
    --learning_rate 2e-6 \
    --lr_schedule low_high_cosine \
    --freeze_learning_rate 5e-6 \
    --lr_ramp_timesteps 0 \
    --lr_decay_start_fraction 0.70 \
    --final_learning_rate 2e-7 \
    --ent_coef 0.0001 \
    --ent_decay_start_fraction 0.70 \
    --final_ent_coef 0 \
    --total_timesteps 700000 \
    --log_suffix "$SUFFIX" \
    --headless > "$TRAIN_LOG" 2>&1; then
    write_status "FAILED_TRAINING" "Inspect $TRAIN_LOG"
    exit 1
fi

write_status "REPORTING" "Generating BC-KL probe report"
if ! "$ENV_BIN/python" evaluation/training_analysis.py --log_dir "$LOG_DIR" \
    >> "$TRAIN_LOG" 2>&1; then
    write_status "FAILED_REPORT" "Probe completed but report generation failed"
    exit 1
fi

if ! "$ENV_BIN/python" - "$LOG_DIR/episode_progress.csv" >> "$TRAIN_LOG" 2>&1 <<'PY'
import csv
import statistics
import sys

with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream))

def truth(value):
    return str(value).strip().lower() in {"true", "1", "yes"}

if len(rows) < 500:
    raise SystemExit("Insufficient probe episodes")
last500 = rows[-500:]
last100 = rows[-100:]
success500 = sum(truth(row["is_success"]) for row in last500) / 500
success100 = sum(truth(row["is_success"]) for row in last100) / 100
timeout100 = sum(truth(row["time_limit_truncated"]) for row in last100) / 100
successful_returns = [
    float(row["discounted_episode_return"])
    for row in rows
    if truth(row["is_success"])
]
failed_returns = [
    float(row["discounted_episode_return"])
    for row in rows
    if not truth(row["is_success"])
]
print(
    f"[BC-KL probe] last500_success={success500:.3f} "
    f"last100_success={success100:.3f} last100_timeout={timeout100:.3f} "
    f"mean_discounted_success={statistics.fmean(successful_returns):.3f} "
    f"mean_discounted_failure={statistics.fmean(failed_returns):.3f}"
)
if success500 < 0.40 or success100 < 0.35:
    raise SystemExit("BC-KL probe stability gate failed")
PY
then
    write_status "HALTED_HEALTH" "BC-KL probe completed but stability gate failed"
    exit 1
fi

write_status "COMPLETED" "BC-KL probe and report passed the stability gate"
