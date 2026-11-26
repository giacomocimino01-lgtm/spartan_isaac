#!/usr/bin/env bash
set -uo pipefail

# Conservative transition from the asynchronous guided BC to a genuinely
# minimal-mask PPO policy.  Replay labels are first revalidated against the
# minimal mask; deterministic ranking is also performed with the minimal mask.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
BC_CHECKPOINT="artifacts/checkpoints/bc/phase2_bc_async_bimanual_recovery.zip"
GUIDED_REPLAY="artifacts/datasets/bc/phase2_async_bimanual_recovery_mixed_train.npz"
MINIMAL_REPLAY="artifacts/datasets/bc/phase2_async_bimanual_recovery_mixed_train_minimal_compatible.npz"
SUFFIX="async_bimanual_recovery_adaptive_minimal_replay01_probe_400k"
LOG_DIR="artifacts/logs/random_sb3_log_sim_phase_2_${SUFFIX}"
TRAIN_LOG="$REPO_ROOT/train_phase2_${SUFFIX}.log"
FINAL_CHECKPOINT="artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_2_${SUFFIX}.zip"
BEST_DISTANCE="artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_2_${SUFFIX}.zip"
PERIODIC_GLOB="artifacts/checkpoints/ppo/dvrk_ppo_phase_2_${SUFFIX}_*_steps.zip"
RANK_DIR="artifacts/evaluations/phase2_async_bimanual_minimal_ppo_probe_30ep"
RANK_LOG="$REPO_ROOT/rank_phase2_async_bimanual_minimal_ppo_probe_30ep.log"
JOB_DIR="artifacts/jobs/phase2_async_bimanual_minimal_ppo_probe"
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
    echo "The Phase-2 asynchronous minimal PPO probe is already running." >&2
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
        printf 'bc_replay_dataset=%s\n' "$MINIMAL_REPLAY"
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

for required in "$BC_CHECKPOINT" "$GUIDED_REPLAY"; do
    [[ -f "$required" ]] || fail "FAILED_SETUP" "Missing required artifact: $required"
done

write_status "VALIDATING_REPLAY" "Proving guided demonstrations are legal under the minimal mask"
PYTHONPATH=source "$ENV_BIN/python" scripts/relabel_bc_dataset_mask_mode.py \
    --source "$GUIDED_REPLAY" \
    --output "$MINIMAL_REPLAY" \
    --task_phase phase_2 \
    --action_mask_mode minimal > relabel_phase2_async_bimanual_replay_minimal.log 2>&1 || \
    fail "FAILED_REPLAY_VALIDATION" "Guided replay contains labels illegal under minimal masking"

write_status "TRAINING" "400k adaptive-minimal PPO probe with supervised replay from the 82% guided BC"
"$ISAACLAB_SH" -p scripts/parallel_run.py \
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
    --adaptive_mask_success_threshold 0.80 \
    --adaptive_mask_window_episodes 128 \
    --adaptive_mask_cooldown_timesteps 32768 \
    --bc_replay_dataset "$MINIMAL_REPLAY" \
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
    --headless > "$TRAIN_LOG" 2>&1 || \
    fail "FAILED_TRAINING" "Inspect $TRAIN_LOG"

write_status "REPORTING" "Generating the adaptive-minimal PPO report"
"$ENV_BIN/python" evaluation/training_analysis.py --log_dir "$LOG_DIR" \
    >> "$TRAIN_LOG" 2>&1 || \
    fail "FAILED_REPORT" "Training completed but report generation failed"

[[ -f "$FINAL_CHECKPOINT" ]] || \
    fail "FAILED_FINAL_CHECKPOINT" "PPO completed without its final checkpoint"

write_status "RANKING" "Comparing BC, best-distance, periodic and final checkpoints under the minimal mask"
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
    --action_mask_mode minimal \
    --output_dir "$RANK_DIR" \
    --best_name phase2_async_bimanual_minimal_probe_best \
    --device cuda:0 > "$RANK_LOG" 2>&1 || \
    fail "FAILED_RANKING" "Inspect $RANK_LOG"

write_status "COMPLETED" "400k adaptive-minimal PPO probe, report and deterministic ranking completed"
