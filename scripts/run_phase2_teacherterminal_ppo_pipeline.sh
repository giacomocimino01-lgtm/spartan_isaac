#!/usr/bin/env bash
set -euo pipefail

# Conservative PPO adaptation of the 95% Phase-2 teacher-terminal BC.
# The BC itself participates in checkpoint ranking, so PPO can never replace
# it merely because it is newer.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_PYTHON="/home/aiprah/miniconda3/envs/env_isaaclab/bin/python"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"
TCC_ROOT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs"
TCC_DATA_ROOT="/mnt/data/aiprah/data/random_sim_dataset_tcc/train"

BC_CHECKPOINT="$REPO_ROOT/artifacts/checkpoints/bc/phase2_bc_phase1_teacher_terminal128_80ep_seed42.zip"
BC_REPLAY="$REPO_ROOT/artifacts/datasets/bc/phase2_phase1_teacher_terminal128_train_seed20261019.npz"
PHASE1_POLICY="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_1_native_minimal_recovery_mixed_replay_probe_300k.zip"
START_EMBEDDINGS="$REPO_ROOT/artifacts/evaluations/phase_router_phase01234_first_frame_20261007/start_embeddings.pt"

SUFFIX="phase1_teacherterminal_replay_kl_ppo_probe400k"
LOG_DIR="$REPO_ROOT/artifacts/logs/random_sb3_log_sim_phase_2_${SUFFIX}"
TRAIN_LOG="$REPO_ROOT/train_phase2_${SUFFIX}.log"
FINAL="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_2_${SUFFIX}.zip"
BEST_DISTANCE="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_2_${SUFFIX}.zip"
STEP_384K="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_phase_2_${SUFFIX}_384000_steps.zip"

JOB_DIR="$REPO_ROOT/artifacts/jobs/phase2_teacherterminal_ppo400k"
STATUS="$JOB_DIR/status.txt"
RANK_DIR="$REPO_ROOT/artifacts/evaluations/phase2_teacherterminal_ppo400k_parked_rank50"
SELECTED="$RANK_DIR/phase2_teacherterminal_ppo400k_best50.zip"
BC_HANDOFF_DIR="$REPO_ROOT/artifacts/evaluations/phase12_teacherterminal_bc80_50ep_seed20261021"
SELECTED_HANDOFF_DIR="$REPO_ROOT/artifacts/evaluations/phase12_teacherterminal_selected_ppo400k_50ep_seed20261021"

mkdir -p "$JOB_DIR" "$RANK_DIR" "$BC_HANDOFF_DIR" "$SELECTED_HANDOFF_DIR"
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
    echo "The Phase-2 teacher-terminal PPO pipeline is already running." >&2
    exit 3
fi

write_status() {
    {
        printf 'updated_at=%s\n' "$(date -Is)"
        printf 'state=%s\n' "$1"
        printf 'detail=%s\n' "$2"
        printf 'bc_checkpoint=%s\n' "$BC_CHECKPOINT"
        printf 'bc_replay=%s\n' "$BC_REPLAY"
        printf 'ppo_final=%s\n' "$FINAL"
        printf 'selected_checkpoint=%s\n' "$SELECTED"
        printf 'rank_dir=%s\n' "$RANK_DIR"
        printf 'bc_handoff_eval=%s\n' "$BC_HANDOFF_DIR"
        printf 'selected_handoff_eval=%s\n' "$SELECTED_HANDOFF_DIR"
        printf 'total_timesteps=400000\n'
        printf 'actor_freeze_timesteps=150000\n'
        printf 'learning_rate=2e-6\n'
        printf 'bc_replay_coef=0.1\n'
        printf 'kl_anchor_coef=0.01\n'
        printf 'action_mask_mode=minimal\n'
        printf 'phase_entry_pose=parked\n'
    } > "$STATUS"
}

fail() {
    write_status "$1" "$2"
    printf '[phase2-teacherterminal-ppo] %s: %s\n' "$1" "$2" >&2
    exit 1
}

for required in "$BC_CHECKPOINT" "$BC_REPLAY" "$PHASE1_POLICY" "$START_EMBEDDINGS"; do
    [[ -f "$required" ]] || fail "MISSING_INPUT" "$required"
done

if [[ ! -f "$FINAL" ]]; then
    write_status "TRAINING_PPO" "400k parked-entry PPO with dense replay and KL anchor to the 95% BC"
    "$ISAACLAB_SH" -p scripts/parallel_run.py \
        --num_envs 64 \
        --task_phase phase_2 \
        --phase2_initial_state red2_blue2 \
        --phase_entry_park_reset \
        --goal_dataset_root /mnt/data/aiprah/data/random_sim_dataset_tcc \
        --pretrained_checkpoint "$BC_CHECKPOINT" \
        --disable_obs_normalization \
        --action_masking \
        --action_mask_mode minimal \
        --bc_replay_dataset "$BC_REPLAY" \
        --bc_replay_coef 0.1 \
        --bc_replay_batch_size 256 \
        --bc_replay_decision_fraction 0.5 \
        --kl_anchor_checkpoint "$BC_CHECKPOINT" \
        --kl_anchor_coef 0.01 \
        --gamma 0.999 \
        --gae_lambda 0.999 \
        --freeze_policy_timesteps 150000 \
        --learning_rate 2e-6 \
        --lr_schedule low_high_cosine \
        --freeze_learning_rate 5e-7 \
        --lr_ramp_timesteps 50000 \
        --lr_decay_start_fraction 0.70 \
        --final_learning_rate 2e-7 \
        --ent_coef 0.0001 \
        --ent_decay_start_fraction 0.70 \
        --final_ent_coef 0 \
        --total_timesteps 400000 \
        --log_suffix "$SUFFIX" \
        --headless \
        > "$TRAIN_LOG" 2>&1 || \
        fail "FAILED_PPO" "Inspect $TRAIN_LOG"
fi

write_status "REPORTING" "Generating the PPO training report"
"$ENV_PYTHON" evaluation/training_analysis.py --log_dir "$LOG_DIR" \
    > "$JOB_DIR/training_report.log" 2>&1 || \
    fail "FAILED_REPORT" "Inspect $JOB_DIR/training_report.log"

for required in "$FINAL" "$BEST_DISTANCE" "$STEP_384K"; do
    [[ -f "$required" ]] || fail "MISSING_CHECKPOINT" "$required"
done

write_status "RANKING" "Ranking BC and PPO candidates over 50 deterministic parked-entry Phase-2 episodes"
"$ENV_PYTHON" scripts/rank_checkpoints.py \
    --checkpoints "$BC_CHECKPOINT" "$BEST_DISTANCE" "$STEP_384K" "$FINAL" \
    --episodes 50 \
    --seed 20261021 \
    --task_phase phase_2 \
    --phase2_initial_state red2_blue2 \
    --phase_entry_park_reset \
    --tcc_checkpoint "$TCC_ROOT/random_sim_phase2_tcc/checkpoints/4000.ckpt" \
    --dataset_path "$TCC_DATA_ROOT/phase_2" \
    --fixed_rings \
    --action_masking \
    --action_mask_mode minimal \
    --output_dir "$RANK_DIR" \
    --best_name phase2_teacherterminal_ppo400k_best50 \
    --device cuda:0 \
    > "$JOB_DIR/rank50.log" 2>&1 || \
    fail "FAILED_RANKING" "Inspect $JOB_DIR/rank50.log"
[[ -f "$SELECTED" ]] || fail "MISSING_SELECTED" "$SELECTED"

evaluate_handoff() {
    local phase2_checkpoint="$1"
    local output_dir="$2"
    local log_path="$3"
    "$ISAACLAB_SH" -p scripts/eval_full_task.py \
        --phase phase_1 \
            "$PHASE1_POLICY" \
            "$TCC_ROOT/random_sim_phase1_tcc/checkpoints/31000.ckpt" \
            "$TCC_DATA_ROOT/phase_1" \
        --phase phase_2 \
            "$phase2_checkpoint" \
            "$TCC_ROOT/random_sim_phase2_tcc/checkpoints/4000.ckpt" \
            "$TCC_DATA_ROOT/phase_2" \
        --start_embeddings "$START_EMBEDDINGS" \
        --sequence phase_1 phase_2 \
        --phase_success_mode symbolic \
        --episodes 50 \
        --seed 20261021 \
        --max_steps_per_phase 900 \
        --episode_length_s 240 \
        --action_mask_mode minimal \
        --output_dir "$output_dir" \
        --disable_video \
        --headless \
        --device cuda:0 \
        > "$log_path" 2>&1
}

if [[ ! -f "$BC_HANDOFF_DIR/summary.json" ]]; then
    write_status "EVALUATING_BC_HANDOFF" "Measuring the untouched BC on 50 learned-Phase-1 handoffs"
    evaluate_handoff "$BC_CHECKPOINT" "$BC_HANDOFF_DIR" "$JOB_DIR/eval_bc_handoff50.log" || \
        fail "FAILED_BC_HANDOFF_EVAL" "Inspect $JOB_DIR/eval_bc_handoff50.log"
fi

write_status "EVALUATING_SELECTED_HANDOFF" "Measuring the selected BC/PPO candidate on the same 50 handoffs"
evaluate_handoff "$SELECTED" "$SELECTED_HANDOFF_DIR" "$JOB_DIR/eval_selected_handoff50.log" || \
    fail "FAILED_SELECTED_HANDOFF_EVAL" "Inspect $JOB_DIR/eval_selected_handoff50.log"

write_status "COMPLETED" "PPO training, ranking and matched 50-episode handoff evaluations completed"
echo "[phase2-teacherterminal-ppo] Completed: $SELECTED"

