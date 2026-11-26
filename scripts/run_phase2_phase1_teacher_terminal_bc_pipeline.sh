#!/usr/bin/env bash
set -euo pipefail

# Rebuild Phase 2 from real, exhaustively varied Phase-1 terminal states.
# The job is resumable: completed artifacts are reused after an interruption.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_PYTHON="/home/aiprah/miniconda3/envs/env_isaaclab/bin/python"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"
TCC_ROOT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs"
TCC_DATA_ROOT="/mnt/data/aiprah/data/random_sim_dataset_tcc/train"

SEED="${SEED:-20261019}"
SOURCE_DATASET="$REPO_ROOT/artifacts/datasets/bc/phase2_phase1_teacher_terminal160_seed${SEED}.npz"
TRAIN_DATASET="$REPO_ROOT/artifacts/datasets/bc/phase2_phase1_teacher_terminal128_train_seed${SEED}.npz"
VALID_DATASET="$REPO_ROOT/artifacts/datasets/bc/phase2_phase1_teacher_terminal32_valid_seed${SEED}.npz"
BC_STEM="$REPO_ROOT/artifacts/checkpoints/bc/phase2_bc_phase1_teacher_terminal128_80ep_seed42"
BC_CHECKPOINT="${BC_STEM}.zip"

PHASE1_POLICY="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_1_native_minimal_recovery_mixed_replay_probe_300k.zip"
START_EMBEDDINGS="$REPO_ROOT/artifacts/evaluations/phase_router_phase01234_first_frame_20261007/start_embeddings.pt"
EVAL_DIR="$REPO_ROOT/artifacts/evaluations/phase12_teacherterminal_bc80_20ep_seed20261020"
JOB_DIR="$REPO_ROOT/artifacts/jobs/phase2_phase1_teacher_terminal_bc80"
STATUS="$JOB_DIR/status.txt"

mkdir -p "$JOB_DIR" "$EVAL_DIR" "$(dirname "$SOURCE_DATASET")" "$(dirname "$BC_STEM")"
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
    echo "The Phase-2 teacher-terminal pipeline is already running." >&2
    exit 3
fi

write_status() {
    {
        printf 'updated_at=%s\n' "$(date -Is)"
        printf 'state=%s\n' "$1"
        printf 'detail=%s\n' "$2"
        printf 'source_dataset=%s\n' "$SOURCE_DATASET"
        printf 'train_dataset=%s\n' "$TRAIN_DATASET"
        printf 'valid_dataset=%s\n' "$VALID_DATASET"
        printf 'bc_checkpoint=%s\n' "$BC_CHECKPOINT"
        printf 'evaluation_dir=%s\n' "$EVAL_DIR"
        printf 'layout_cases=144\n'
        printf 'epochs=80\n'
        printf 'action_mask_mode=minimal\n'
    } > "$STATUS"
}

fail() {
    write_status "$1" "$2"
    printf '[phase2-teacher-terminal] %s: %s\n' "$1" "$2" >&2
    exit 1
}

if [[ ! -f "$SOURCE_DATASET" ]]; then
    write_status "COLLECTING" "Collecting 160 dense Phase-2 demonstrations after enumerated Phase-1 handoffs"
    "$ISAACLAB_SH" -p scripts/collect_phase2_phase1_teacher_terminal_demos.py \
        --num_envs 8 \
        --num_episodes 160 \
        --output_path "$SOURCE_DATASET" \
        --phase1_tcc_checkpoint "$TCC_ROOT/random_sim_phase1_tcc/checkpoints/31000.ckpt" \
        --phase2_tcc_checkpoint "$TCC_ROOT/random_sim_phase2_tcc/checkpoints/4000.ckpt" \
        --dataset_root "$TCC_DATA_ROOT" \
        --seed "$SEED" \
        --action_mask_mode minimal \
        --headless \
        --device cuda:0 \
        > "$JOB_DIR/collect.log" 2>&1 || \
        fail "FAILED_COLLECTION" "Inspect $JOB_DIR/collect.log"
fi

write_status "CHECKING_DATASET" "Checking all dense expert labels against the minimal mask"
PYTHONPATH=source "$ENV_PYTHON" scripts/check_bc_dataset_mask.py \
    --dataset "$SOURCE_DATASET" \
    --task_phase phase_2 \
    --action_mask_mode minimal \
    > "$JOB_DIR/check_mask.log" 2>&1 || \
    fail "FAILED_MASK_CHECK" "Expert labels violate the minimal mask"

if [[ ! -f "$TRAIN_DATASET" || ! -f "$VALID_DATASET" ]]; then
    write_status "SPLITTING" "Creating an episode-disjoint 128/32 train-validation split"
    "$ENV_PYTHON" scripts/split_episode_bc_dataset.py \
        --source "$SOURCE_DATASET" \
        --train_output "$TRAIN_DATASET" \
        --valid_output "$VALID_DATASET" \
        --validation_episodes 32 \
        --split_seed 20261020 \
        > "$JOB_DIR/split.log" 2>&1 || \
        fail "FAILED_SPLIT" "Inspect $JOB_DIR/split.log"
fi

if [[ ! -f "$BC_CHECKPOINT" ]]; then
    write_status "TRAINING_BC" "Training Phase-2 BC from scratch on 128 true-handoff episodes"
    "$ISAACLAB_SH" -p scripts/pretrain_ppo.py \
        --dataset_path "$TRAIN_DATASET" \
        --output_path "$BC_STEM" \
        --task_phase phase_2 \
        --action_mask_mode minimal \
        --epochs 80 \
        --batch_size 256 \
        --lr 1e-3 \
        --seed 42 \
        --headless \
        > "$JOB_DIR/train_bc.log" 2>&1 || \
        fail "FAILED_BC" "Inspect $JOB_DIR/train_bc.log"
fi

write_status "CHECKING_OFFLINE" "Measuring exact-action accuracy on train and held-out episodes"
for spec in \
    "train:$TRAIN_DATASET" \
    "valid:$VALID_DATASET"; do
    name="${spec%%:*}"
    dataset="${spec#*:}"
    "$ENV_PYTHON" scripts/check_bc_policy.py \
        --checkpoint "$BC_CHECKPOINT" \
        --dataset_path "$dataset" \
        --normalization raw \
        --task_phase phase_2 \
        --device cuda \
        > "$JOB_DIR/check_${name}.log" 2>&1 || \
        fail "FAILED_OFFLINE_CHECK" "$name"
done

write_status "EVALUATING_TRUE_HANDOFF" "Running 20 deterministic learned-Phase-1 to new-Phase-2 episodes with videos"
"$ISAACLAB_SH" -p scripts/eval_full_task.py \
    --phase phase_1 \
        "$PHASE1_POLICY" \
        "$TCC_ROOT/random_sim_phase1_tcc/checkpoints/31000.ckpt" \
        "$TCC_DATA_ROOT/phase_1" \
    --phase phase_2 \
        "$BC_CHECKPOINT" \
        "$TCC_ROOT/random_sim_phase2_tcc/checkpoints/4000.ckpt" \
        "$TCC_DATA_ROOT/phase_2" \
    --start_embeddings "$START_EMBEDDINGS" \
    --sequence phase_1 phase_2 \
    --phase_success_mode symbolic \
    --episodes 20 \
    --seed 20261020 \
    --max_steps_per_phase 900 \
    --episode_length_s 240 \
    --action_mask_mode minimal \
    --output_dir "$EVAL_DIR" \
    --video_frame_stride 5 \
    --video_fps 15 \
    --headless \
    --device cuda:0 \
    > "$JOB_DIR/eval_true_handoff.log" 2>&1 || \
    fail "FAILED_TRUE_HANDOFF_EVAL" "Inspect $JOB_DIR/eval_true_handoff.log"

write_status "COMPLETED" "Collection, BC training and 20-episode true-handoff evaluation completed"
echo "[phase2-teacher-terminal] Completed: $BC_CHECKPOINT"

