#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_PYTHON="/home/aiprah/miniconda3/envs/env_isaaclab/bin/python"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"
TCC_DATA_ROOT="/mnt/data/aiprah/data/random_sim_dataset_tcc"
TCC_CHECKPOINT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase4_tcc/checkpoints/2600.ckpt"

DATA_DIR="$REPO_ROOT/artifacts/datasets/bc"
JOB_DIR="$REPO_ROOT/artifacts/jobs/phase4_parked_interface_retrain_400k"
EVAL_DIR="$REPO_ROOT/artifacts/evaluations/phase4_parked_interface_retrain_400k"
REPORT_DIR="$REPO_ROOT/artifacts/reports/random_sb3_log_sim_phase_4_parked_interface_replay_400k"
STATUS="$JOB_DIR/status.txt"

PARKED_SOURCE="$DATA_DIR/phase4_bimanual_minimal_parked80.npz"
PARKED_TRAIN="$DATA_DIR/phase4_bimanual_minimal_parked64_train.npz"
PARKED_VALID="$DATA_DIR/phase4_bimanual_minimal_parked16_valid.npz"
CANONICAL_TRAIN="$DATA_DIR/phase4_bimanual_minimal128_train.npz"
MIXED_REPLAY="$DATA_DIR/phase4_bimanual_minimal_canonical128_parked64_replay.npz"

BASE_CHECKPOINT="$REPO_ROOT/artifacts/evaluations/phase4_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/phase4_bimanual_minimal_best_50ep.zip"
SUFFIX="parked_interface_replay_400k"
PPO_BEST="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_4_${SUFFIX}.zip"
PPO_STEP="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_phase_4_${SUFFIX}_384000_steps.zip"
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
        printf 'base_checkpoint=%s\n' "$BASE_CHECKPOINT"
        printf 'mixed_replay=%s\n' "$MIXED_REPLAY"
        printf 'ppo_final=%s\n' "$PPO_FINAL"
        printf 'evaluation_dir=%s\n' "$EVAL_DIR"
    } > "$STATUS"
}

fail() {
    write_status "$1" "$2"
    printf '[phase4-parked-retrain] %s: %s\n' "$1" "$2" >&2
    exit 1
}

[[ -f "$PARKED_SOURCE" ]] || fail "MISSING_PARKED_DATA" "$PARKED_SOURCE"
[[ -f "$CANONICAL_TRAIN" ]] || fail "MISSING_CANONICAL_DATA" "$CANONICAL_TRAIN"
[[ -f "$BASE_CHECKPOINT" ]] || fail "MISSING_BASE_CHECKPOINT" "$BASE_CHECKPOINT"

write_status "PREPARING_DATA" "Splitting parked demonstrations and building mixed replay"
"$ENV_PYTHON" scripts/split_episode_bc_dataset.py \
    --source "$PARKED_SOURCE" \
    --train_output "$PARKED_TRAIN" \
    --valid_output "$PARKED_VALID" \
    --validation_episodes 16 --split_seed 20261007 \
    > "$JOB_DIR/split_parked.log" 2>&1 || \
    fail "FAILED_SPLIT" "Inspect $JOB_DIR/split_parked.log"

"$ENV_PYTHON" scripts/merge_bc_datasets.py \
    --dataset "$CANONICAL_TRAIN" 1 \
    --dataset "$PARKED_TRAIN" 1 \
    --output "$MIXED_REPLAY" \
    > "$JOB_DIR/merge_replay.log" 2>&1 || \
    fail "FAILED_MERGE" "Inspect $JOB_DIR/merge_replay.log"

PYTHONPATH=source "$ENV_PYTHON" scripts/check_bc_dataset_mask.py \
    --dataset "$MIXED_REPLAY" --task_phase phase_4 \
    --action_mask_mode minimal > "$JOB_DIR/check_replay_mask.log" 2>&1 || \
    fail "FAILED_MASK_CHECK" "Mixed replay contains mask-incompatible labels"

write_status "TRAINING_PPO" "Fine-tuning from the 98% canonical policy on parked interface resets"
"$ISAACLAB_SH" -p scripts/parallel_run.py \
    --num_envs 64 --seed 42 --task_phase phase_4 \
    --goal_dataset_root "$TCC_DATA_ROOT" \
    --pretrained_checkpoint "$BASE_CHECKPOINT" \
    --phase_entry_park_reset --disable_obs_normalization \
    --action_masking --action_mask_mode minimal \
    --bc_replay_dataset "$MIXED_REPLAY" --bc_replay_coef 0.2 \
    --bc_replay_batch_size 256 --bc_replay_decision_fraction 0.5 \
    --gamma 0.999 --gae_lambda 0.999 \
    --freeze_policy_timesteps 0 --learning_rate 2e-6 \
    --lr_schedule constant --final_learning_rate 2e-6 \
    --ent_coef 0.0001 --ent_decay_start_fraction 0.70 --final_ent_coef 0 \
    --total_timesteps 400000 --log_suffix "$SUFFIX" \
    --headless --device cuda:0 > "$JOB_DIR/train_ppo.log" 2>&1 || \
    fail "FAILED_PPO" "Inspect $JOB_DIR/train_ppo.log"

write_status "REPORTING" "Generating PPO training report"
"$ENV_PYTHON" evaluation/training_analysis.py \
    --log_dir "$REPO_ROOT/artifacts/logs/random_sb3_log_sim_phase_4_${SUFFIX}" \
    > "$JOB_DIR/report.log" 2>&1 || \
    fail "FAILED_REPORT" "Inspect $JOB_DIR/report.log"

for required in "$PPO_BEST" "$PPO_STEP" "$PPO_FINAL"; do
    [[ -f "$required" ]] || fail "MISSING_CHECKPOINT" "$required"
done

write_status "EVALUATING_PARKED" "Ranking base and fine-tuned checkpoints on 50 parked episodes"
"$ENV_PYTHON" scripts/rank_checkpoints.py \
    --checkpoints "$BASE_CHECKPOINT" "$PPO_BEST" "$PPO_STEP" "$PPO_FINAL" \
    --episodes 50 --seed 42 --task_phase phase_4 \
    --tcc_checkpoint "$TCC_CHECKPOINT" \
    --dataset_path "$TCC_DATA_ROOT/train/phase_4" --fixed_rings \
    --phase_entry_park_reset --action_masking --action_mask_mode minimal \
    --output_dir "$EVAL_DIR/parked_rank_50ep" \
    --best_name phase4_parked_interface_best_50ep --device cuda:0 \
    > "$JOB_DIR/rank_parked.log" 2>&1 || \
    fail "FAILED_PARKED_RANK" "Inspect $JOB_DIR/rank_parked.log"

write_status "EVALUATING_CANONICAL" "Checking retention on 50 canonical Phase-4 episodes"
"$ENV_PYTHON" scripts/rank_checkpoints.py \
    --checkpoints "$BASE_CHECKPOINT" "$PPO_BEST" "$PPO_STEP" "$PPO_FINAL" \
    --episodes 50 --seed 42 --task_phase phase_4 \
    --tcc_checkpoint "$TCC_CHECKPOINT" \
    --dataset_path "$TCC_DATA_ROOT/train/phase_4" --fixed_rings \
    --action_masking --action_mask_mode minimal \
    --output_dir "$EVAL_DIR/canonical_rank_50ep" \
    --best_name phase4_canonical_retention_best_50ep --device cuda:0 \
    > "$JOB_DIR/rank_canonical.log" 2>&1 || \
    fail "FAILED_CANONICAL_RANK" "Inspect $JOB_DIR/rank_canonical.log"

write_status "COMPLETED" "Parked-interface fine-tuning, report and dual evaluation completed"
