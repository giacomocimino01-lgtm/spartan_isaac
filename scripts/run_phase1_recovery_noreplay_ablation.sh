#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
JOB_DIR="artifacts/jobs/phase1_recovery_noreplay_ablation"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/job.lock"
SUFFIX="native_minimal_recovery_mixed_noreplay_ablation_300k"
TRAIN_LOG="$REPO_ROOT/train_phase1_${SUFFIX}.log"
EVAL_LOG="$REPO_ROOT/eval_phase1_${SUFFIX}.log"
TCC_CHECKPOINT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase1_tcc/checkpoints/31000.ckpt"
DATASET_PATH="/mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_1"
BASE_CHECKPOINT="artifacts/evaluations/phase1_native_minimal_bc_stratified50_gamma0999_gae0999_probe_400k_rank_50ep/phase1_bc_stratified50_best_50ep.zip"
BASE_VECNORMALIZE="artifacts/evaluations/phase1_native_minimal_bc_stratified50_gamma0999_gae0999_probe_400k_rank_50ep/phase1_bc_stratified50_best_50ep_vecnormalize.pkl"
REPLAY_BEST="artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_1_native_minimal_recovery_mixed_replay_probe_300k.zip"
ABLATION_BEST="artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_1_${SUFFIX}.zip"
ABLATION_FINAL="artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_1_${SUFFIX}.zip"

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
    echo "The Phase-1 no-replay ablation is already running." >&2
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
        printf 'evaluation_log=%s\n' "$EVAL_LOG"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

for required in "$TCC_CHECKPOINT" "$BASE_CHECKPOINT" "$BASE_VECNORMALIZE" "$REPLAY_BEST"; do
    if [[ ! -f "$required" ]]; then
        write_status "FAILED_SETUP" "Missing required file: $required"
        exit 2
    fi
done
if [[ ! -d "$DATASET_PATH" ]]; then
    write_status "FAILED_SETUP" "Missing dataset directory: $DATASET_PATH"
    exit 2
fi

write_status "TRAINING" "300k recovery-mixed PPO ablation without BC replay"
if ! "$ISAACLAB_SH" -p scripts/parallel_run.py \
    --num_envs 64 \
    --task_phase phase_1 \
    --phase1_initial_state recovery_mixed \
    --goal_dataset_root /mnt/data/aiprah/data/random_sim_dataset_tcc \
    --pretrained_checkpoint "$BASE_CHECKPOINT" \
    --pretrained_vecnormalize "$BASE_VECNORMALIZE" \
    --disable_obs_normalization \
    --action_masking \
    --action_mask_mode minimal \
    --gamma 0.999 \
    --gae_lambda 0.999 \
    --freeze_policy_timesteps 0 \
    --learning_rate 2e-6 \
    --lr_schedule constant \
    --ent_coef 0.0001 \
    --ent_decay_start_fraction 0.70 \
    --final_ent_coef 0 \
    --total_timesteps 300000 \
    --log_suffix "$SUFFIX" \
    --headless > "$TRAIN_LOG" 2>&1; then
    write_status "FAILED_TRAINING" "Inspect $TRAIN_LOG"
    exit 1
fi

for required in "$ABLATION_BEST" "$ABLATION_FINAL"; do
    if [[ ! -f "$required" ]]; then
        write_status "FAILED_CHECKPOINT" "Training finished without expected checkpoint: $required"
        exit 2
    fi
done

CHECKPOINTS=("$BASE_CHECKPOINT" "$REPLAY_BEST" "$ABLATION_BEST" "$ABLATION_FINAL")
: > "$EVAL_LOG"

run_ranking() {
    local state="$1"
    local initial_state="$2"
    local episodes="$3"
    local output_dir="$4"
    local best_name="$5"

    write_status "$state" "initial_state=$initial_state episodes=$episodes"
    "$ENV_BIN/python" scripts/rank_checkpoints.py \
        --checkpoints "${CHECKPOINTS[@]}" \
        --episodes "$episodes" \
        --seed 42 \
        --task_phase phase_1 \
        --phase1_initial_state "$initial_state" \
        --tcc_checkpoint "$TCC_CHECKPOINT" \
        --dataset_path "$DATASET_PATH" \
        --action_masking \
        --action_mask_mode minimal \
        --output_dir "$output_dir" \
        --best_name "$best_name" \
        --device cuda:0 >> "$EVAL_LOG" 2>&1
}

if ! run_ranking \
    "EVALUATING_RED3_BLUE1" \
    "red3_blue1" \
    20 \
    "artifacts/evaluations/phase1_noreplay_ablation_red3_blue1_20ep" \
    "phase1_noreplay_ablation_red3_blue1_best"; then
    write_status "FAILED_EVAL_RED3_BLUE1" "Inspect $EVAL_LOG"
    exit 1
fi

if ! run_ranking \
    "EVALUATING_RED1_BLUE3" \
    "red1_blue3" \
    20 \
    "artifacts/evaluations/phase1_noreplay_ablation_red1_blue3_20ep" \
    "phase1_noreplay_ablation_red1_blue3_best"; then
    write_status "FAILED_EVAL_RED1_BLUE3" "Inspect $EVAL_LOG"
    exit 1
fi

if ! run_ranking \
    "EVALUATING_STACK_GREEN" \
    "stack_green" \
    50 \
    "artifacts/evaluations/phase1_noreplay_ablation_stack_green_50ep" \
    "phase1_noreplay_ablation_stack_green_best"; then
    write_status "FAILED_EVAL_STACK_GREEN" "Inspect $EVAL_LOG"
    exit 1
fi

write_status "COMPLETED" "No-replay ablation and deterministic evaluations completed"
