#!/usr/bin/env bash
set -euo pipefail

# Resume only the evaluations for the completed Phase-2 recovery PPO run.
# A complete 64-episode Isaac evaluation takes longer than one hour, so each
# checkpoint receives a three-hour timeout.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ENV_PYTHON="/home/aiprah/miniconda3/envs/env_isaaclab/bin/python"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"
TCC_CHECKPOINT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase2_tcc/checkpoints/4000.ckpt"
TCC_DATASET="/mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_2"
BC_CHECKPOINT="$REPO_ROOT/artifacts/checkpoints/bc/phase2_bc_phase1_teacher_terminal128_80ep_seed42.zip"
RECOVERY_BANK="$REPO_ROOT/artifacts/datasets/recovery/phase2_bc_recovery_states64_seed20261021.npz"
SUFFIX="teacherterminal_recoverybank50_replay003_kl001_ppo500k"
BEST="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_2_${SUFFIX}.zip"
FINAL="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_2_${SUFFIX}.zip"
RECOVERY_RANK="$REPO_ROOT/artifacts/evaluations/phase2_recovery_curriculum500k_bank64_rank_retry"
SELECTED="$RECOVERY_RANK/phase2_recovery_curriculum500k_best.zip"
PARKED_RANK="$REPO_ROOT/artifacts/evaluations/phase2_recovery_curriculum500k_parked50_rank_retry"
JOB_DIR="$REPO_ROOT/artifacts/jobs/phase2_recovery_curriculum500k"
STATUS="$JOB_DIR/evaluation_retry_status.txt"

mkdir -p "$RECOVERY_RANK" "$PARKED_RANK" "$JOB_DIR"
cd "$REPO_ROOT"
source "$CONDA_SH"
conda activate env_isaaclab
export PATH="/home/aiprah/miniconda3/envs/env_isaaclab/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export IsaacLab_PATH="/home/aiprah/Documents/IsaacLab"

exec 9>"$JOB_DIR/evaluation_retry.lock"
if ! flock -n 9; then
    echo "The Phase-2 recovery evaluation retry is already running." >&2
    exit 3
fi

write_status() {
    {
        printf 'updated_at=%s\n' "$(date -Is)"
        printf 'state=%s\n' "$1"
        printf 'detail=%s\n' "$2"
        printf 'recovery_ranking=%s\n' "$RECOVERY_RANK/checkpoint_ranking.csv"
        printf 'selected_checkpoint=%s\n' "$SELECTED"
        printf 'nominal_ranking=%s\n' "$PARKED_RANK/checkpoint_ranking.csv"
        printf 'timeout_per_checkpoint_seconds=10800\n'
    } > "$STATUS"
}

fail() {
    write_status "$1" "$2"
    echo "[phase2-recovery-eval] $1: $2" >&2
    exit 1
}

for required in "$BC_CHECKPOINT" "$RECOVERY_BANK" "$BEST" "$FINAL" "$TCC_CHECKPOINT"; do
    [[ -f "$required" ]] || fail "MISSING_INPUT" "$required"
done

write_status "RANKING_RECOVERY" "Evaluating four checkpoints on all 64 recovery states"
"$ENV_PYTHON" scripts/rank_checkpoints.py \
    --checkpoints "$BC_CHECKPOINT" "$BEST" "$FINAL" \
        "$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_phase_2_${SUFFIX}_*_steps.zip" \
    --episodes 64 \
    --timeout_seconds 10800 \
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
    > "$JOB_DIR/rank_recovery64_retry.log" 2>&1 || \
    fail "FAILED_RECOVERY_RANK" "Inspect $JOB_DIR/rank_recovery64_retry.log"

[[ -f "$SELECTED" ]] || fail "MISSING_SELECTED" "$SELECTED"

write_status "RANKING_NOMINAL" "Evaluating BC and recovery winner on 50 nominal parked episodes"
"$ENV_PYTHON" scripts/rank_checkpoints.py \
    --checkpoints "$BC_CHECKPOINT" "$SELECTED" \
    --episodes 50 \
    --timeout_seconds 10800 \
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
    > "$JOB_DIR/rank_parked50_retry.log" 2>&1 || \
    fail "FAILED_NOMINAL_RANK" "Inspect $JOB_DIR/rank_parked50_retry.log"

write_status "COMPLETED" "Recovery and nominal deterministic evaluations completed"
echo "[phase2-recovery-eval] Completed: $SELECTED"
