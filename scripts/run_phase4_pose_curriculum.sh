#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_PYTHON="/home/aiprah/miniconda3/envs/env_isaaclab/bin/python"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"
TCC_DATA_ROOT="/mnt/data/aiprah/data/random_sim_dataset_tcc"
TCC_CHECKPOINT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase4_tcc/checkpoints/2600.ckpt"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-800000}"
SUFFIX="pose_curriculum_replay_${TOTAL_TIMESTEPS}"
JOB_DIR="$REPO_ROOT/artifacts/jobs/phase4_${SUFFIX}"
EVAL_DIR="$REPO_ROOT/artifacts/evaluations/phase4_${SUFFIX}"
LOG_DIR="$REPO_ROOT/artifacts/logs/random_sb3_log_sim_phase_4_${SUFFIX}"
REPORT_DIR="$REPO_ROOT/artifacts/reports/random_sb3_log_sim_phase_4_${SUFFIX}"
STATUS="$JOB_DIR/status.txt"

BASE_CHECKPOINT="$REPO_ROOT/artifacts/evaluations/phase4_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/phase4_bimanual_minimal_best_50ep.zip"
REPLAY_DATASET="$REPO_ROOT/artifacts/datasets/bc/phase4_bimanual_minimal128_train.npz"
PPO_BEST="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_4_${SUFFIX}.zip"
PPO_FINAL="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_4_${SUFFIX}.zip"

mkdir -p "$JOB_DIR" "$EVAL_DIR" "$REPORT_DIR"
cd "$REPO_ROOT"
source "$CONDA_SH"
conda activate env_isaaclab
export IsaacLab_PATH="/home/aiprah/Documents/IsaacLab"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

write_status() {
    {
        printf 'updated_at=%s\n' "$(date -Is)"
        printf 'state=%s\n' "$1"
        printf 'detail=%s\n' "$2"
        printf 'curriculum=canonical_to_full_task_parked_pose\n'
        printf 'pose_scales=0.00,0.25,0.50,0.75,1.00\n'
        printf 'advance_threshold=0.85\n'
        printf 'window_episodes=128\n'
        printf 'base_checkpoint=%s\n' "$BASE_CHECKPOINT"
        printf 'replay_dataset=%s\n' "$REPLAY_DATASET"
        printf 'ppo_best=%s\n' "$PPO_BEST"
        printf 'ppo_final=%s\n' "$PPO_FINAL"
        printf 'evaluation_dir=%s\n' "$EVAL_DIR"
    } > "$STATUS"
}

fail() {
    write_status "$1" "$2"
    printf '[phase4-pose-curriculum] %s: %s\n' "$1" "$2" >&2
    exit 1
}

[[ -f "$BASE_CHECKPOINT" ]] || fail "MISSING_BASE_CHECKPOINT" "$BASE_CHECKPOINT"
[[ -f "$REPLAY_DATASET" ]] || fail "MISSING_REPLAY_DATASET" "$REPLAY_DATASET"

write_status "TRAINING_PPO" "Adaptive success-gated pose curriculum is running"
"$ISAACLAB_SH" -p scripts/parallel_run.py \
    --num_envs 64 --seed 42 --task_phase phase_4 \
    --goal_dataset_root "$TCC_DATA_ROOT" \
    --pretrained_checkpoint "$BASE_CHECKPOINT" \
    --adaptive_phase_entry_pose_curriculum \
    --adaptive_phase_entry_pose_start_scale 0 \
    --adaptive_phase_entry_pose_scale_step 0.25 \
    --adaptive_phase_entry_pose_success_threshold 0.85 \
    --adaptive_phase_entry_pose_window_episodes 128 \
    --disable_obs_normalization --action_masking --action_mask_mode minimal \
    --bc_replay_dataset "$REPLAY_DATASET" --bc_replay_coef 0.1 \
    --bc_replay_batch_size 256 --bc_replay_decision_fraction 0.5 \
    --gamma 0.999 --gae_lambda 0.999 \
    --freeze_policy_timesteps 0 --learning_rate 2e-6 \
    --lr_schedule constant --final_learning_rate 2e-6 \
    --ent_coef 0.0001 --ent_decay_start_fraction 0.70 --final_ent_coef 0 \
    --total_timesteps "$TOTAL_TIMESTEPS" --log_suffix "$SUFFIX" \
    --headless --device cuda:0 > "$JOB_DIR/train_ppo.log" 2>&1 || \
    fail "FAILED_PPO" "Inspect $JOB_DIR/train_ppo.log"

write_status "REPORTING" "Generating curriculum-aware Phase-4 training report"
"$ENV_PYTHON" evaluation/training_analysis.py \
    --log_dir "$LOG_DIR" > "$JOB_DIR/report.log" 2>&1 || \
    fail "FAILED_REPORT" "Inspect $JOB_DIR/report.log"

for required in "$PPO_BEST" "$PPO_FINAL"; do
    [[ -f "$required" ]] || fail "MISSING_CHECKPOINT" "$required"
done

write_status "EVALUATING_PARKED" "Ranking base and curriculum checkpoints on exact parked resets"
"$ENV_PYTHON" scripts/rank_checkpoints.py \
    --checkpoints "$BASE_CHECKPOINT" "$PPO_BEST" "$PPO_FINAL" \
    --episodes 50 --seed 42 --task_phase phase_4 \
    --tcc_checkpoint "$TCC_CHECKPOINT" \
    --dataset_path "$TCC_DATA_ROOT/train/phase_4" --fixed_rings \
    --phase_entry_park_reset --action_masking --action_mask_mode minimal \
    --output_dir "$EVAL_DIR/parked_rank_50ep" \
    --best_name phase4_pose_curriculum_parked_best_50ep --device cuda:0 \
    > "$JOB_DIR/rank_parked.log" 2>&1 || \
    fail "FAILED_PARKED_RANK" "Inspect $JOB_DIR/rank_parked.log"

write_status "EVALUATING_CANONICAL" "Checking retention on canonical Phase-4 resets"
"$ENV_PYTHON" scripts/rank_checkpoints.py \
    --checkpoints "$BASE_CHECKPOINT" "$PPO_BEST" "$PPO_FINAL" \
    --episodes 50 --seed 42 --task_phase phase_4 \
    --tcc_checkpoint "$TCC_CHECKPOINT" \
    --dataset_path "$TCC_DATA_ROOT/train/phase_4" --fixed_rings \
    --action_masking --action_mask_mode minimal \
    --output_dir "$EVAL_DIR/canonical_rank_50ep" \
    --best_name phase4_pose_curriculum_canonical_best_50ep --device cuda:0 \
    > "$JOB_DIR/rank_canonical.log" 2>&1 || \
    fail "FAILED_CANONICAL_RANK" "Inspect $JOB_DIR/rank_canonical.log"

write_status "COMPLETED" "Training report and parked/canonical rankings completed"
