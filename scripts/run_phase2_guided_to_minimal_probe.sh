#!/usr/bin/env bash
set -uo pipefail

# Validate the historically robust guided Phase-2 BC, then start a short
# success-gated transition toward minimal semantics.  This is deliberately a
# 400k probe: it measures how far the curriculum can open before committing to
# a multi-million-step run.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
BC_CHECKPOINT="artifacts/checkpoints/bc/phase2_bc_decisions.zip"
BC_REPLAY_DATASET="artifacts/datasets/bc/dataset_supervisionato_phase2_decisions.npz"
GUIDED_EVAL_DIR="artifacts/evaluations/phase2_bc_decisions_guided_50ep"
GUIDED_EVAL_CSV="$GUIDED_EVAL_DIR/episodes.csv"
GUIDED_EVAL_LOG="$REPO_ROOT/eval_phase2_bc_decisions_guided_50ep.log"
SUFFIX="guided_bc_adaptive_minimal_replay01_probe_400k"
TRAIN_LOG="$REPO_ROOT/train_phase2_${SUFFIX}.log"
LOG_DIR="artifacts/logs/random_sb3_log_sim_phase_2_${SUFFIX}"
JOB_DIR="artifacts/jobs/phase2_guided_to_minimal_probe"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/job.lock"

mkdir -p "$JOB_DIR" "$GUIDED_EVAL_DIR"
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
    echo "The Phase-2 guided-to-minimal probe is already running." >&2
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
        printf 'bc_checkpoint=%s\n' "$BC_CHECKPOINT"
        printf 'guided_evaluation_csv=%s\n' "$GUIDED_EVAL_CSV"
        printf 'training_log=%s\n' "$TRAIN_LOG"
        printf 'ppo_log_dir=%s\n' "$LOG_DIR"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

for required in "$BC_CHECKPOINT" "$BC_REPLAY_DATASET"; do
    if [[ ! -f "$required" ]]; then
        write_status "FAILED_SETUP" "Missing required file: $required"
        exit 2
    fi
done

write_status "EVALUATING_GUIDED_BC" "Validating the historical BC over 50 deterministic guided-mask episodes"
if ! "$ISAACLAB_SH" -p scripts/eval_run.py \
    --checkpoint "$BC_CHECKPOINT" \
    --tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase2_tcc/checkpoints/4000.ckpt \
    --dataset_path /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_2 \
    --task_phase phase_2 --episodes 50 --seed 42 \
    --metrics_output "$GUIDED_EVAL_CSV" \
    --disable_video --quiet_steps \
    --action_masking --action_mask_mode guided --headless > "$GUIDED_EVAL_LOG" 2>&1; then
    write_status "FAILED_GUIDED_EVAL" "Inspect $GUIDED_EVAL_LOG"
    exit 1
fi

if ! "$ENV_BIN/python" - "$GUIDED_EVAL_CSV" >> "$GUIDED_EVAL_LOG" 2>&1 <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream))
successes = sum(str(row["success"]).lower() in {"true", "1", "yes"} for row in rows)
rate = successes / len(rows) if rows else 0.0
print(f"[guided BC gate] success={successes}/{len(rows)} ({rate:.1%})")
if len(rows) != 50 or rate < 0.90:
    raise SystemExit("Guided BC did not reach the 90% rollout gate")
PY
then
    write_status "HALTED_GUIDED_GATE" "Guided BC stayed below 90%; PPO probe was not started"
    exit 1
fi

write_status "TRAINING_PROBE" "400k PPO probe with success-gated guided-to-minimal curriculum and BC replay"
if ! "$ISAACLAB_SH" -p scripts/parallel_run.py \
    --num_envs 64 \
    --task_phase phase_2 \
    --goal_dataset_root /mnt/data/aiprah/data/random_sim_dataset_tcc \
    --pretrained_checkpoint "$BC_CHECKPOINT" \
    --disable_obs_normalization \
    --action_masking \
    --action_mask_mode minimal \
    --soft_mask_initial_penalty 8 \
    --soft_mask_final_penalty 0 \
    --adaptive_minimal_mask_curriculum \
    --adaptive_mask_start_scale 0 \
    --adaptive_mask_scale_step 0.1 \
    --adaptive_mask_penalty_step 1 \
    --adaptive_mask_success_threshold 0.85 \
    --adaptive_mask_window_episodes 128 \
    --adaptive_mask_cooldown_timesteps 32768 \
    --bc_replay_dataset "$BC_REPLAY_DATASET" \
    --bc_replay_coef 0.1 \
    --bc_replay_batch_size 256 \
    --gamma 0.999 \
    --gae_lambda 0.999 \
    --freeze_policy_timesteps 0 \
    --learning_rate 2e-6 \
    --lr_schedule constant \
    --ent_coef 0.0001 \
    --ent_decay_start_fraction 0.75 \
    --final_ent_coef 0 \
    --total_timesteps 400000 \
    --log_suffix "$SUFFIX" \
    --headless > "$TRAIN_LOG" 2>&1; then
    write_status "FAILED_TRAINING" "Inspect $TRAIN_LOG"
    exit 1
fi

write_status "REPORTING" "Generating the 400k curriculum report"
if ! "$ENV_BIN/python" evaluation/training_analysis.py --log_dir "$LOG_DIR" \
    >> "$TRAIN_LOG" 2>&1; then
    write_status "FAILED_REPORT" "Probe completed but report generation failed"
    exit 1
fi

"$ENV_BIN/python" - "$LOG_DIR" >> "$TRAIN_LOG" 2>&1 <<'PY'
import csv
import math
import sys

root = sys.argv[1]
with open(f"{root}/progress.csv", newline="") as stream:
    progress = list(csv.DictReader(stream))
with open(f"{root}/episode_progress.csv", newline="") as stream:
    episodes = list(csv.DictReader(stream))

def truth(value):
    return str(value).strip().lower() in {"true", "1", "yes"}

last = progress[-1]
window = episodes[-min(500, len(episodes)):]
success = sum(truth(row["is_success"]) for row in window) / len(window)
scale = float(last.get("curriculum/manipulability_scale", "nan"))
penalty = float(last.get("curriculum/soft_mask_penalty", "nan"))
print(
    f"[probe summary] episodes={len(episodes)} last_window_success={success:.3f} "
    f"scale={scale:.3f} penalty={penalty:.3f}"
)
PY

write_status "COMPLETED" "Guided BC gate and 400k adaptive-minimal probe completed; rank checkpoints next"
