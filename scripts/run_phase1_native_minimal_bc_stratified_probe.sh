#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
BC_CHECKPOINT="artifacts/checkpoints/bc/phase1_bc_native_minimal_160.zip"
BC_DATASET="artifacts/datasets/bc/dataset_supervisionato_phase1_native_minimal_160.npz"
SUFFIX="native_minimal_bc_stratified50_gamma0999_gae0999_probe_400k"
LOG_DIR="artifacts/logs/random_sb3_log_sim_phase_1_${SUFFIX}"
TRAIN_LOG="$REPO_ROOT/train_phase1_${SUFFIX}.log"
RANK_LOG="$REPO_ROOT/rank_phase1_${SUFFIX}_50ep.log"
RANK_DIR="artifacts/evaluations/phase1_${SUFFIX}_rank_50ep"
FINAL_CHECKPOINT="artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_1_${SUFFIX}.zip"
STEP_CHECKPOINT="artifacts/checkpoints/ppo/dvrk_ppo_phase_1_${SUFFIX}_384000_steps.zip"
BEST_DISTANCE_CHECKPOINT="artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_1_${SUFFIX}.zip"
JOB_DIR="$REPO_ROOT/artifacts/jobs/phase1_native_minimal_bc_stratified_probe"
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
export IsaacLab_PATH=/home/aiprah/Documents/IsaacLab

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "The Phase-1 stratified BC-replay probe is already running." >&2
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
        printf 'ranking_log=%s\n' "$RANK_LOG"
        printf 'pretrained_checkpoint=%s\n' "$BC_CHECKPOINT"
        printf 'bc_replay_dataset=%s\n' "$BC_DATASET"
        printf 'bc_replay_coefficient=0.1\n'
        printf 'bc_replay_batch_size=256\n'
        printf 'bc_replay_decision_fraction=0.5\n'
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

for required_file in "$BC_CHECKPOINT" "$BC_DATASET"; do
    if [[ ! -f "$required_file" ]]; then
        write_status "FAILED" "Missing BC artifact: $required_file"
        exit 2
    fi
done

write_status "TRAINING" "400k stratified BC-replay probe; 50% decision rows; actor frozen through 150k"
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
    --bc_replay_dataset "$BC_DATASET" \
    --bc_replay_coef 0.1 \
    --bc_replay_batch_size 256 \
    --bc_replay_decision_fraction 0.5 \
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
    --total_timesteps 400000 \
    --log_suffix "$SUFFIX" \
    --headless > "$TRAIN_LOG" 2>&1; then
    write_status "FAILED_TRAINING" "Inspect $TRAIN_LOG"
    exit 1
fi

write_status "REPORTING" "Generating stratified replay training report"
if ! "$ENV_BIN/python" evaluation/training_analysis.py --log_dir "$LOG_DIR" \
    >> "$TRAIN_LOG" 2>&1; then
    write_status "FAILED_REPORT" "Probe completed but report generation failed"
    exit 1
fi

health_state="PASSED"
if ! "$ENV_BIN/python" - "$LOG_DIR/episode_progress.csv" >> "$TRAIN_LOG" 2>&1 <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream))

def truth(value):
    return str(value).strip().lower() in {"true", "1", "yes"}

if len(rows) < 300:
    raise SystemExit("Insufficient probe episodes")
last300 = rows[-300:]
last100 = rows[-100:]
success300 = sum(truth(row["is_success"]) for row in last300) / 300
success100 = sum(truth(row["is_success"]) for row in last100) / 100
timeout100 = sum(truth(row["time_limit_truncated"]) for row in last100) / 100
print(
    f"[Stratified BC-replay probe] last300_success={success300:.3f} "
    f"last100_success={success100:.3f} last100_timeout={timeout100:.3f}"
)
if success300 < 0.45 or success100 < 0.40:
    raise SystemExit("Stratified BC-replay probe stability gate failed")
PY
then
    health_state="FAILED"
fi

for checkpoint in "$FINAL_CHECKPOINT" "$STEP_CHECKPOINT" "$BEST_DISTANCE_CHECKPOINT"; do
    if [[ ! -f "$checkpoint" ]]; then
        write_status "FAILED_RANKING" "Missing trained checkpoint: $checkpoint"
        exit 1
    fi
done

write_status "RANKING" "Training finished; deterministic 50-episode ranking in progress"
if ! "$ENV_BIN/python" scripts/rank_checkpoints.py \
    --checkpoints "$BC_CHECKPOINT" "$BEST_DISTANCE_CHECKPOINT" "$STEP_CHECKPOINT" "$FINAL_CHECKPOINT" \
    --episodes 50 \
    --seed 42 \
    --task_phase phase_1 \
    --tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase1_tcc/checkpoints/31000.ckpt \
    --dataset_path /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_1 \
    --action_masking \
    --action_mask_mode minimal \
    --output_dir "$RANK_DIR" \
    --best_name phase1_bc_stratified50_best_50ep \
    --device cuda:0 > "$RANK_LOG" 2>&1; then
    write_status "FAILED_RANKING" "Inspect $RANK_LOG"
    exit 1
fi

if [[ "$health_state" == "FAILED" ]]; then
    write_status "HALTED_HEALTH" "Training and ranking completed; stochastic stability gate failed"
    exit 1
fi
write_status "COMPLETED" "Training report and deterministic ranking completed"
