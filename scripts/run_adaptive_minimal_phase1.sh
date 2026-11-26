#!/usr/bin/env bash
set -uo pipefail

# Success-gated transition from the high-performing 768k Phase-1 checkpoint
# to a fully minimal action mask.  Observation semantics open first; only
# after scale=1 does the soft-mask logit penalty begin to decrease.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
CHECKPOINT="artifacts/checkpoints/ppo/dvrk_ppo_phase_1_softmask_to_minimal_finetune_2m_768000_steps.zip"
SUFFIX="adaptive_minimalmask_finetune_3m"
LOG_DIR="artifacts/logs/random_sb3_log_sim_phase_1_${SUFFIX}"
TRAIN_LOG="$REPO_ROOT/train_phase1_${SUFFIX}.log"
JOB_DIR="$REPO_ROOT/artifacts/jobs/phase1_adaptive_minimalmask"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/job.lock"

mkdir -p "$JOB_DIR"
cd "$REPO_ROOT"
export PATH="$ENV_BIN:$PATH"
export TERM=xterm-256color
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "The adaptive Phase-1 job is already running." >&2
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
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

if [[ ! -f "$CHECKPOINT" ]]; then
    write_status "FAILED" "Missing checkpoint: $CHECKPOINT"
    exit 2
fi

write_status "RUNNING" "Adaptive Phase-1 minimal-mask curriculum"
if ! "$ISAACLAB_SH" -p scripts/parallel_run.py \
    --num_envs 64 \
    --task_phase phase_1 \
    --goal_dataset_root /mnt/data/aiprah/data/random_sim_dataset_tcc \
    --pretrained_checkpoint "$CHECKPOINT" \
    --disable_obs_normalization \
    --action_masking \
    --action_mask_mode minimal \
    --soft_mask_initial_penalty 8 \
    --soft_mask_final_penalty 0 \
    --adaptive_minimal_mask_curriculum \
    --adaptive_mask_start_scale 0.55 \
    --adaptive_mask_scale_step 0.05 \
    --adaptive_mask_penalty_step 1 \
    --adaptive_mask_success_threshold 0.85 \
    --adaptive_mask_window_episodes 256 \
    --adaptive_mask_cooldown_timesteps 32768 \
    --freeze_policy_timesteps 0 \
    --learning_rate 3e-6 \
    --lr_schedule constant \
    --ent_coef 0.001 \
    --ent_decay_start_fraction 0.80 \
    --final_ent_coef 0.0001 \
    --total_timesteps 3000000 \
    --log_suffix "$SUFFIX" \
    --headless > "$TRAIN_LOG" 2>&1; then
    write_status "FAILED_TRAINING" "Inspect $TRAIN_LOG"
    exit 1
fi

write_status "REPORTING" "Generating Phase-1 report"
if ! "$ENV_BIN/python" evaluation/training_analysis.py --log_dir "$LOG_DIR" \
    >> "$TRAIN_LOG" 2>&1; then
    write_status "FAILED_REPORT" "Training completed but report generation failed"
    exit 1
fi

if ! "$ENV_BIN/python" - "$LOG_DIR" >> "$TRAIN_LOG" 2>&1 <<'PY'
import csv
import sys

root = sys.argv[1]
with open(f"{root}/progress.csv", newline="") as stream:
    progress = list(csv.DictReader(stream))
with open(f"{root}/episode_progress.csv", newline="") as stream:
    episodes = list(csv.DictReader(stream))

def truth(value):
    return str(value).strip().lower() in {"true", "1", "yes"}

if not progress or len(episodes) < 500:
    print("[health] FAIL: insufficient completed data")
    raise SystemExit(1)

last = progress[-1]
window = episodes[-500:]
success = sum(truth(row["is_success"]) for row in window) / len(window)
scale = float(last.get("curriculum/manipulability_scale", "nan"))
penalty = float(last.get("curriculum/soft_mask_penalty", "nan"))
complete = float(last.get("curriculum/minimal_complete", "0")) >= 0.5
print(
    f"[health] last500_success={success:.3f} scale={scale:.3f} "
    f"penalty={penalty:.3f} minimal_complete={complete}"
)
if success < 0.75 or scale < 1.0 or penalty > 0.0 or not complete:
    print("[health] FAIL: robust fully-minimal policy not reached")
    raise SystemExit(1)
print("[health] PASS")
PY
then
    write_status "HALTED_HEALTH" "Fully-minimal success gate was not reached"
    exit 1
fi

write_status "COMPLETED" "Adaptive Phase-1 training and report passed the health gate"
