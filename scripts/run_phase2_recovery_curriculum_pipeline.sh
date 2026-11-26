#!/usr/bin/env bash
set -euo pipefail

# Phase-2 RL recovery experiment:
# 1) collect real timeout/correction physical states from the deterministic BC;
# 2) mix those resets 50/50 with the nominal parked distribution;
# 3) adapt with PPO reward, using only weak BC replay and a weak finite KL anchor;
# 4) rank on the recovery bank, then verify the winner on nominal parked resets.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_PYTHON="/home/aiprah/miniconda3/envs/env_isaaclab/bin/python"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"
TCC_CHECKPOINT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase2_tcc/checkpoints/4000.ckpt"
TCC_DATASET="/mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_2"

BC_CHECKPOINT="$REPO_ROOT/artifacts/checkpoints/bc/phase2_bc_phase1_teacher_terminal128_80ep_seed42.zip"
BC_REPLAY="$REPO_ROOT/artifacts/datasets/bc/phase2_phase1_teacher_terminal128_train_seed20261019.npz"
RECOVERY_BANK="$REPO_ROOT/artifacts/datasets/recovery/phase2_bc_recovery_states64_seed20261021.npz"
SUFFIX="teacherterminal_recoverybank50_replay003_kl001_ppo500k"
TRAIN_LOG="$REPO_ROOT/train_phase2_${SUFFIX}.log"
LOG_DIR="$REPO_ROOT/artifacts/logs/random_sb3_log_sim_phase_2_${SUFFIX}"
JOB_DIR="$REPO_ROOT/artifacts/jobs/phase2_recovery_curriculum500k"
STATUS="$JOB_DIR/status.txt"
RECOVERY_RANK="$REPO_ROOT/artifacts/evaluations/phase2_recovery_curriculum500k_bank64_rank"
SELECTED="$RECOVERY_RANK/phase2_recovery_curriculum500k_best.zip"
PARKED_RANK="$REPO_ROOT/artifacts/evaluations/phase2_recovery_curriculum500k_parked50_rank"

mkdir -p "$JOB_DIR" "$(dirname "$RECOVERY_BANK")" "$RECOVERY_RANK" "$PARKED_RANK"
cd "$REPO_ROOT"
source "$CONDA_SH"
conda activate env_isaaclab
export PATH="/home/aiprah/miniconda3/envs/env_isaaclab/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export IsaacLab_PATH="/home/aiprah/Documents/IsaacLab"

exec 9>"$JOB_DIR/job.lock"
if ! flock -n 9; then
    echo "The Phase-2 recovery curriculum is already running." >&2
    exit 3
fi

write_status() {
    {
        printf 'updated_at=%s\n' "$(date -Is)"
        printf 'state=%s\n' "$1"
        printf 'detail=%s\n' "$2"
        printf 'bc_checkpoint=%s\n' "$BC_CHECKPOINT"
        printf 'recovery_bank=%s\n' "$RECOVERY_BANK"
        printf 'ppo_final=%s\n' "$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_2_${SUFFIX}.zip"
        printf 'selected_checkpoint=%s\n' "$SELECTED"
        printf 'total_timesteps=500000\n'
        printf 'actor_freeze_timesteps=50000\n'
        printf 'recovery_reset_probability=0.5\n'
        printf 'bc_replay_coef=0.03\n'
        printf 'kl_anchor_coef=0.001\n'
        printf 'action_mask_mode=minimal\n'
    } > "$STATUS"
}

fail() {
    write_status "$1" "$2"
    printf '[phase2-recovery] %s: %s\n' "$1" "$2" >&2
    exit 1
}

for required in "$BC_CHECKPOINT" "$BC_REPLAY" "$TCC_CHECKPOINT"; do
    [[ -f "$required" ]] || fail "MISSING_INPUT" "$required"
done

if [[ ! -f "$RECOVERY_BANK" ]]; then
    write_status "COLLECTING_RECOVERY_BANK" "Collecting 64 unlabeled physical states from genuine BC reach timeouts"
    "$ISAACLAB_SH" -p scripts/collect_phase2_recovery_state_bank.py \
        --checkpoint "$BC_CHECKPOINT" \
        --tcc_checkpoint "$TCC_CHECKPOINT" \
        --dataset_path "$TCC_DATASET" \
        --output_path "$RECOVERY_BANK" \
        --num_envs 64 \
        --target_states 64 \
        --max_episodes 500 \
        --snapshot_stride 5 \
        --max_states_per_failure 48 \
        --late_state_start 350 \
        --stochastic_actions \
        --seed 20261021 \
        --action_mask_mode minimal \
        --headless \
        --device cuda:0 \
        > "$JOB_DIR/collect_recovery_bank.log" 2>&1 || \
        fail "FAILED_RECOVERY_COLLECTION" "Inspect $JOB_DIR/collect_recovery_bank.log"
fi

write_status "TRAINING_PPO" "500k PPO on a 50/50 nominal/recovery reset mixture"
"$ISAACLAB_SH" -p scripts/parallel_run.py \
    --num_envs 64 \
    --task_phase phase_2 \
    --phase2_initial_state red2_blue2 \
    --phase_entry_empirical_bank "$RECOVERY_BANK" \
    --phase_entry_empirical_probability 0.5 \
    --goal_dataset_root /mnt/data/aiprah/data/random_sim_dataset_tcc \
    --pretrained_checkpoint "$BC_CHECKPOINT" \
    --disable_obs_normalization \
    --action_masking \
    --action_mask_mode minimal \
    --bc_replay_dataset "$BC_REPLAY" \
    --bc_replay_coef 0.03 \
    --bc_replay_batch_size 128 \
    --bc_replay_decision_fraction 0.5 \
    --kl_anchor_checkpoint "$BC_CHECKPOINT" \
    --kl_anchor_coef 0.001 \
    --gamma 0.999 \
    --gae_lambda 0.999 \
    --freeze_policy_timesteps 50000 \
    --learning_rate 2e-6 \
    --lr_schedule low_high_cosine \
    --freeze_learning_rate 5e-7 \
    --lr_ramp_timesteps 50000 \
    --lr_decay_start_fraction 0.70 \
    --final_learning_rate 2e-7 \
    --ent_coef 0.0003 \
    --ent_decay_start_fraction 0.70 \
    --final_ent_coef 0 \
    --total_timesteps 500000 \
    --log_suffix "$SUFFIX" \
    --headless \
    > "$TRAIN_LOG" 2>&1 || fail "FAILED_PPO" "Inspect $TRAIN_LOG"

write_status "REPORTING" "Generating the PPO training report"
"$ENV_PYTHON" evaluation/training_analysis.py --log_dir "$LOG_DIR" \
    > "$JOB_DIR/training_report.log" 2>&1 || \
    fail "FAILED_REPORT" "Inspect $JOB_DIR/training_report.log"

FINAL="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_2_${SUFFIX}.zip"
BEST="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_2_${SUFFIX}.zip"
[[ -f "$FINAL" ]] || fail "MISSING_CHECKPOINT" "$FINAL"
[[ -f "$BEST" ]] || fail "MISSING_CHECKPOINT" "$BEST"

write_status "RANKING_RECOVERY" "Ranking BC and PPO checkpoints on all 64 recovery states"
"$ENV_PYTHON" scripts/rank_checkpoints.py \
    --checkpoints "$BC_CHECKPOINT" "$BEST" "$FINAL" \
        "$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_phase_2_${SUFFIX}_*_steps.zip" \
    --episodes 64 \
    --seed 20261022 \
    --task_phase phase_2 \
    --phase2_initial_state red2_blue2 \
    --phase_entry_empirical_bank "$RECOVERY_BANK" \
    --phase_entry_empirical_probability 1.0 \
    --phase_entry_empirical_sample_mode sequential \
    --tcc_checkpoint "$TCC_CHECKPOINT" \
    --dataset_path "$TCC_DATASET" \
    --fixed_rings \
    --action_masking \
    --action_mask_mode minimal \
    --output_dir "$RECOVERY_RANK" \
    --best_name phase2_recovery_curriculum500k_best \
    --device cuda:0 \
    > "$JOB_DIR/rank_recovery64.log" 2>&1 || \
    fail "FAILED_RECOVERY_RANK" "Inspect $JOB_DIR/rank_recovery64.log"
[[ -f "$SELECTED" ]] || fail "MISSING_SELECTED" "$SELECTED"

write_status "RANKING_NOMINAL" "Checking BC and recovery winner on 50 nominal parked Phase-2 episodes"
"$ENV_PYTHON" scripts/rank_checkpoints.py \
    --checkpoints "$BC_CHECKPOINT" "$SELECTED" \
    --episodes 50 \
    --seed 20261021 \
    --task_phase phase_2 \
    --phase2_initial_state red2_blue2 \
    --phase_entry_park_reset \
    --tcc_checkpoint "$TCC_CHECKPOINT" \
    --dataset_path "$TCC_DATASET" \
    --fixed_rings \
    --action_masking \
    --action_mask_mode minimal \
    --output_dir "$PARKED_RANK" \
    --best_name phase2_recovery_curriculum500k_nominal_best \
    --device cuda:0 \
    > "$JOB_DIR/rank_parked50.log" 2>&1 || \
    fail "FAILED_NOMINAL_RANK" "Inspect $JOB_DIR/rank_parked50.log"

write_status "COMPLETED" "Recovery-state PPO, recovery ranking and nominal regression test completed"
echo "[phase2-recovery] Completed: $SELECTED"
