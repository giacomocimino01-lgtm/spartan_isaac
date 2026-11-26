#!/usr/bin/env bash
set -uo pipefail

# Resume the asynchronous Phase-2 PPO policy from the point reached by the
# 400k probe.  VecNormalize reward statistics are restored and the adaptive
# curriculum resumes explicitly at scale=0.5, penalty=8.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
BC_CHECKPOINT="artifacts/checkpoints/bc/phase2_bc_async_bimanual_recovery.zip"
START_CHECKPOINT="artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_2_async_bimanual_recovery_adaptive_minimal_replay01_probe_400k.zip"
START_VECNORMALIZE="artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_2_async_bimanual_recovery_adaptive_minimal_replay01_probe_400k_vecnormalize.pkl"
REPLAY_DATASET="artifacts/datasets/bc/phase2_async_bimanual_recovery_mixed_train_minimal_compatible.npz"
SUFFIX="async_bimanual_recovery_adaptive_minimal_resume05_replay01_1p5m"
LOG_DIR="artifacts/logs/random_sb3_log_sim_phase_2_${SUFFIX}"
TRAIN_LOG="$REPO_ROOT/train_phase2_${SUFFIX}.log"
FINAL_CHECKPOINT="artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_2_${SUFFIX}.zip"
BEST_DISTANCE="artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_2_${SUFFIX}.zip"
PERIODIC_GLOB="artifacts/checkpoints/ppo/dvrk_ppo_phase_2_${SUFFIX}_*_steps.zip"
RANK_DIR="artifacts/evaluations/phase2_async_bimanual_minimal_resume_1p5m_30ep"
RANK_LOG="$REPO_ROOT/rank_phase2_async_bimanual_minimal_resume_1p5m_30ep.log"
JOB_DIR="artifacts/jobs/phase2_async_bimanual_minimal_resume_1p5m"
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
    echo "The Phase-2 1.5M minimal-mask continuation is already running." >&2
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
        printf 'start_checkpoint=%s\n' "$START_CHECKPOINT"
        printf 'start_vecnormalize=%s\n' "$START_VECNORMALIZE"
        printf 'bc_replay_dataset=%s\n' "$REPLAY_DATASET"
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

for required in "$BC_CHECKPOINT" "$START_CHECKPOINT" "$START_VECNORMALIZE" "$REPLAY_DATASET"; do
    [[ -f "$required" ]] || fail "FAILED_SETUP" "Missing required artifact: $required"
done

write_status "TRAINING" "1.5M continuation from scale=0.5 penalty=8 toward fully minimal semantics"
"$ISAACLAB_SH" -p scripts/parallel_run.py \
    --num_envs 64 \
    --task_phase phase_2 \
    --goal_dataset_root /mnt/data/aiprah/data/random_sim_dataset_tcc \
    --pretrained_checkpoint "$START_CHECKPOINT" \
    --pretrained_vecnormalize "$START_VECNORMALIZE" \
    --disable_obs_normalization \
    --action_masking \
    --action_mask_mode minimal \
    --soft_mask_initial_penalty 8 \
    --soft_mask_final_penalty 0 \
    --adaptive_minimal_mask_curriculum \
    --adaptive_mask_start_scale 0.5 \
    --adaptive_mask_scale_step 0.1 \
    --adaptive_mask_penalty_step 1 \
    --adaptive_mask_success_threshold 0.80 \
    --adaptive_mask_window_episodes 128 \
    --adaptive_mask_cooldown_timesteps 32768 \
    --bc_replay_dataset "$REPLAY_DATASET" \
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
    --total_timesteps 1500000 \
    --log_suffix "$SUFFIX" \
    --headless > "$TRAIN_LOG" 2>&1 || \
    fail "FAILED_TRAINING" "Inspect $TRAIN_LOG"

write_status "REPORTING" "Generating the 1.5M continuation report"
"$ENV_BIN/python" evaluation/training_analysis.py --log_dir "$LOG_DIR" \
    >> "$TRAIN_LOG" 2>&1 || \
    fail "FAILED_REPORT" "Training completed but report generation failed"

[[ -f "$FINAL_CHECKPOINT" ]] || \
    fail "FAILED_FINAL_CHECKPOINT" "PPO completed without its final checkpoint"

write_status "RANKING" "Comparing BC, 400k start, 1.5M best, periodic and final checkpoints under fully minimal masking"
"$ENV_BIN/python" scripts/rank_checkpoints.py \
    --checkpoints \
        "$BC_CHECKPOINT" \
        "$START_CHECKPOINT" \
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
    --best_name phase2_async_bimanual_minimal_resume_1p5m_best \
    --device cuda:0 > "$RANK_LOG" 2>&1 || \
    fail "FAILED_RANKING" "Inspect $RANK_LOG"

write_status "COMPLETED" "1.5M continuation, report and fully-minimal deterministic ranking completed"
