#!/usr/bin/env bash
set -uo pipefail

# Conservative guided-mask PPO probe from the recovery-aware bimanual BC.
# The run keeps the demonstrated strategy alive with supervised replay, warms
# up the critic before unfreezing the actor, then ranks every useful candidate
# against the original BC on the same deterministic 30-episode stream.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
BC_CHECKPOINT="artifacts/checkpoints/bc/phase2_bc_native_guided_bimanual_recovery.zip"
BC_REPLAY_DATASET="artifacts/datasets/bc/phase2_bimanual_recovery_mixed_train.npz"
SUFFIX="bimanual_recovery_guided_replay01_probe_400k"
LOG_DIR="artifacts/logs/random_sb3_log_sim_phase_2_${SUFFIX}"
TRAIN_LOG="$REPO_ROOT/train_phase2_${SUFFIX}.log"
FINAL_CHECKPOINT="artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_2_${SUFFIX}.zip"
BEST_DISTANCE="artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_2_${SUFFIX}.zip"
PERIODIC_GLOB="artifacts/checkpoints/ppo/dvrk_ppo_phase_2_${SUFFIX}_*_steps.zip"
RANK_DIR="artifacts/evaluations/phase2_bimanual_recovery_guided_ppo_probe_30ep"
JOB_DIR="artifacts/jobs/phase2_bimanual_recovery_ppo_probe"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/job.lock"

mkdir -p "$JOB_DIR" "$RANK_DIR"
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
    echo "The Phase-2 bimanual recovery PPO probe is already running." >&2
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
        printf 'bc_replay_dataset=%s\n' "$BC_REPLAY_DATASET"
        printf 'training_log=%s\n' "$TRAIN_LOG"
        printf 'ppo_log_dir=%s\n' "$LOG_DIR"
        printf 'final_checkpoint=%s\n' "$FINAL_CHECKPOINT"
        printf 'ranking_dir=%s\n' "$RANK_DIR"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

fail() {
    write_status "$1" "$2"
    exit 1
}

for required in "$BC_CHECKPOINT" "$BC_REPLAY_DATASET"; do
    [[ -f "$required" ]] || fail "FAILED_SETUP" "Missing required artifact: $required"
done

write_status "TRAINING" "400k guided PPO probe; actor frozen for 100k with recovery BC replay"
"$ISAACLAB_SH" -p scripts/parallel_run.py \
    --num_envs 64 \
    --task_phase phase_2 \
    --goal_dataset_root /mnt/data/aiprah/data/random_sim_dataset_tcc \
    --pretrained_checkpoint "$BC_CHECKPOINT" \
    --disable_obs_normalization \
    --action_masking \
    --action_mask_mode guided \
    --bc_replay_dataset "$BC_REPLAY_DATASET" \
    --bc_replay_coef 0.1 \
    --bc_replay_batch_size 256 \
    --gamma 0.999 \
    --gae_lambda 0.999 \
    --freeze_policy_timesteps 100000 \
    --learning_rate 2e-6 \
    --lr_schedule low_high_cosine \
    --freeze_learning_rate 1e-6 \
    --lr_ramp_timesteps 100000 \
    --lr_decay_start_fraction 0.70 \
    --final_learning_rate 2e-7 \
    --ent_coef 0.0001 \
    --ent_decay_start_fraction 0.70 \
    --final_ent_coef 0 \
    --total_timesteps 400000 \
    --log_suffix "$SUFFIX" \
    --headless > "$TRAIN_LOG" 2>&1 || \
    fail "FAILED_TRAINING" "Inspect $TRAIN_LOG"

write_status "REPORTING" "Generating the PPO training report"
"$ENV_BIN/python" evaluation/training_analysis.py --log_dir "$LOG_DIR" \
    >> "$TRAIN_LOG" 2>&1 || \
    fail "FAILED_REPORT" "Training completed but report generation failed"

[[ -f "$FINAL_CHECKPOINT" ]] || \
    fail "FAILED_FINAL_CHECKPOINT" "PPO completed without its final checkpoint"

write_status "RANKING" "Comparing BC, best-distance, periodic and final checkpoints over 30 episodes"
"$ENV_BIN/python" scripts/rank_checkpoints.py \
    --checkpoints \
        "$BC_CHECKPOINT" \
        "$BEST_DISTANCE" \
        "$PERIODIC_GLOB" \
        "$FINAL_CHECKPOINT" \
    --episodes 30 \
    --seed 42 \
    --task_phase phase_2 \
    --tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase2_tcc/checkpoints/4000.ckpt \
    --dataset_path /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_2 \
    --fixed_rings \
    --action_masking \
    --action_mask_mode guided \
    --output_dir "$RANK_DIR" \
    --best_name phase2_bimanual_recovery_guided_probe_best \
    --device cuda:0 > rank_phase2_bimanual_recovery_guided_probe_30ep.log 2>&1 || \
    fail "FAILED_RANKING" "Inspect rank_phase2_bimanual_recovery_guided_probe_30ep.log"

write_status "COMPLETED" "400k PPO probe, report and deterministic ranking completed"
