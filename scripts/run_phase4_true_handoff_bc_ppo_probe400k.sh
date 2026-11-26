#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_PYTHON="/home/aiprah/miniconda3/envs/env_isaaclab/bin/python"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"
TCC_DATA_ROOT="/mnt/data/aiprah/data/random_sim_dataset_tcc"

SUFFIX="true_handoff_bc_replay_probe_400k"
JOB_DIR="$REPO_ROOT/artifacts/jobs/phase4_${SUFFIX}"
LOG_DIR="$REPO_ROOT/artifacts/logs/random_sb3_log_sim_phase_4_${SUFFIX}"
REPORT_DIR="$REPO_ROOT/artifacts/reports/random_sb3_log_sim_phase_4_${SUFFIX}"
STATUS="$JOB_DIR/status.txt"

BC_CHECKPOINT="$REPO_ROOT/artifacts/checkpoints/bc/phase4_bc_true_handoff_adapt10_seed42.zip"
REPLAY_DATASET="$REPO_ROOT/artifacts/datasets/bc/phase4_park_and_true_handoff_bc_adapt.npz"
PPO_BEST="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_4_${SUFFIX}.zip"
PPO_STEP="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_phase_4_${SUFFIX}_384000_steps.zip"
PPO_FINAL="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_4_${SUFFIX}.zip"

mkdir -p "$JOB_DIR" "$REPORT_DIR"
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
        printf 'total_timesteps=400000\n'
        printf 'actor_freeze_timesteps=100000\n'
        printf 'parked_exact_probability=0.60\n'
        printf 'parked_neighborhood_probability=0.40\n'
        printf 'bc_checkpoint=%s\n' "$BC_CHECKPOINT"
        printf 'replay_dataset=%s\n' "$REPLAY_DATASET"
        printf 'ppo_best=%s\n' "$PPO_BEST"
        printf 'ppo_step=%s\n' "$PPO_STEP"
        printf 'ppo_final=%s\n' "$PPO_FINAL"
        printf 'report_dir=%s\n' "$REPORT_DIR"
    } > "$STATUS"
}

fail() {
    write_status "$1" "$2"
    printf '[phase4-true-handoff-ppo] %s: %s\n' "$1" "$2" >&2
    exit 1
}

for required in "$BC_CHECKPOINT" "$REPLAY_DATASET"; do
    [[ -f "$required" ]] || fail "MISSING_INPUT" "$required"
done

write_status "TRAINING_PPO" "Training 400k PPO probe from true-handoff-adapted BC"
"$ISAACLAB_SH" -p scripts/parallel_run.py \
    --num_envs 64 --seed 42 --task_phase phase_4 \
    --goal_dataset_root "$TCC_DATA_ROOT" \
    --pretrained_checkpoint "$BC_CHECKPOINT" --disable_obs_normalization \
    --phase_entry_park_reset \
    --phase_entry_park_neighborhood_probability 0.40 \
    --phase_entry_park_neighborhood_scale 1.0 \
    --action_masking --action_mask_mode minimal \
    --bc_replay_dataset "$REPLAY_DATASET" --bc_replay_coef 0.10 \
    --bc_replay_batch_size 256 --bc_replay_decision_fraction 0.50 \
    --gamma 0.999 --gae_lambda 0.999 \
    --freeze_policy_timesteps 100000 --learning_rate 3e-6 \
    --lr_schedule low_high_cosine --freeze_learning_rate 2e-6 \
    --lr_ramp_timesteps 100000 --lr_decay_start_fraction 0.75 \
    --final_learning_rate 3e-7 --ent_coef 0.0001 \
    --ent_decay_start_fraction 0.75 --final_ent_coef 0 \
    --total_timesteps 400000 --log_suffix "$SUFFIX" \
    --headless --device cuda:0 > "$JOB_DIR/train_ppo.log" 2>&1 || \
    fail "FAILED_PPO" "Inspect $JOB_DIR/train_ppo.log"

for required in "$PPO_BEST" "$PPO_STEP" "$PPO_FINAL"; do
    [[ -f "$required" ]] || fail "MISSING_CHECKPOINT" "$required"
done

write_status "REPORTING" "Generating PPO training report"
"$ENV_PYTHON" evaluation/training_analysis.py --log_dir "$LOG_DIR" \
    > "$JOB_DIR/report.log" 2>&1 || \
    fail "FAILED_REPORT" "Inspect $JOB_DIR/report.log"

write_status "COMPLETED" "400k PPO probe and report completed; closed-loop ranking pending"
