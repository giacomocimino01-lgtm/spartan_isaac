#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
export IsaacLab_PATH="/home/aiprah/Documents/IsaacLab"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"
CONDA_BIN="/home/aiprah/miniconda3/bin/conda"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
XIRL_ROOT="/home/aiprah/Documents/google-research/xirl"
TCC_DATA_ROOT="/mnt/data/aiprah/data/random_sim_dataset_tcc"
TCC_RUN_DIR="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase4_tcc"

JOB_DIR="$REPO_ROOT/artifacts/jobs/phase4_bimanual_minimal_80ep_pipeline"
DATA_DIR="$REPO_ROOT/artifacts/datasets/bc"
BC_DIR="$REPO_ROOT/artifacts/checkpoints/bc"
EVAL_DIR="$REPO_ROOT/artifacts/evaluations/phase4_bimanual_minimal_80ep_pipeline"
REPORT_DIR="$REPO_ROOT/artifacts/reports/random_sb3_log_sim_phase_4_bimanual_minimal_bc80_replay_probe_400k"
STATUS="$JOB_DIR/status.txt"

SOURCE_DATASET="$DATA_DIR/phase4_bimanual_minimal160.npz"
TRAIN_DATASET="$DATA_DIR/phase4_bimanual_minimal128_train.npz"
VALID_DATASET="$DATA_DIR/phase4_bimanual_minimal32_valid.npz"
BC_STEM="$BC_DIR/phase4_bc_bimanual_minimal128_80ep_seed42"
BC_CHECKPOINT="${BC_STEM}.zip"
PPO_SUFFIX="bimanual_minimal_bc80_replay_probe_400k"
PPO_FINAL="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_4_${PPO_SUFFIX}.zip"
PPO_BEST="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_4_${PPO_SUFFIX}.zip"
PPO_STEP="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_phase_4_${PPO_SUFFIX}_384000_steps.zip"
RANK_DIR="$EVAL_DIR/ppo_rank_50ep"
TCC_CHECKPOINT=""

mkdir -p "$JOB_DIR" "$DATA_DIR" "$BC_DIR" "$EVAL_DIR" "$REPORT_DIR"
mkdir -p "$TCC_DATA_ROOT/train/phase_4" "$TCC_DATA_ROOT/valid/phase_4"
cd "$REPO_ROOT"

write_status() {
    {
        printf 'updated_at=%s\n' "$(date -Is)"
        printf 'state=%s\n' "$1"
        printf 'detail=%s\n' "$2"
        printf 'phase=phase_4\n'
        printf 'goal=one_ring_per_coloured_peg_to_two_per_outside_side\n'
        printf 'target_token_right=peg_gray:id8\n'
        printf 'target_token_left=peg_gray1:id9\n'
        printf 'tcc_data_root=%s\n' "$TCC_DATA_ROOT"
        printf 'tcc_run_dir=%s\n' "$TCC_RUN_DIR"
        printf 'selected_tcc_checkpoint=%s\n' "$TCC_CHECKPOINT"
        printf 'bc_checkpoint=%s\n' "$BC_CHECKPOINT"
        printf 'ppo_final=%s\n' "$PPO_FINAL"
        printf 'evaluation_dir=%s\n' "$EVAL_DIR"
    } > "$STATUS"
}

fail() {
    write_status "$1" "$2"
    printf '[phase4-pipeline] %s: %s\n' "$1" "$2" >&2
    exit 1
}

best_tcc_checkpoint() {
    "$ENV_BIN/python" -c \
        "import sys; sys.path.insert(0, '$REPO_ROOT/scripts'); from app_launcher_utils import resolve_tcc_checkpoint; print(resolve_tcc_checkpoint('$TCC_RUN_DIR'))" \
        | tail -n 1
}

video_count() {
    local split="$1"
    find "$TCC_DATA_ROOT/$split/phase_4" -mindepth 1 -maxdepth 1 \
        -type d -name '[0-9]*' 2>/dev/null | wc -l
}

source "$CONDA_SH"
conda activate env_isaaclab
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

TRAIN_COUNT="$(video_count train)"
if (( TRAIN_COUNT < 160 )); then
    TRAIN_REMAINING=$((160 - TRAIN_COUNT))
    write_status "COLLECTING_TCC_TRAIN" "Collecting $TRAIN_REMAINING remaining Phase-4 train videos"
    "$ISAACLAB_SH" -p scripts/collect_tcc_dataset.py \
        --num_envs 8 --num_videos "$TRAIN_REMAINING" --output_root "$TCC_DATA_ROOT" \
        --split train --class_name phase_4 --frame_stride 5 \
        --max_steps_per_video 1600 --max_attempts 320 \
        --ring_order random --arm_mode random --seed 5404 \
        --headless --device cuda:0 > "$JOB_DIR/collect_tcc_train.log" 2>&1 || \
        fail "FAILED_TCC_TRAIN_COLLECTION" "Inspect $JOB_DIR/collect_tcc_train.log"
fi
[[ "$(video_count train)" -eq 160 ]] || \
    fail "FAILED_TCC_TRAIN_COUNT" "Phase-4 train dataset must contain exactly 160 videos"

VALID_COUNT="$(video_count valid)"
if (( VALID_COUNT < 20 )); then
    VALID_REMAINING=$((20 - VALID_COUNT))
    write_status "COLLECTING_TCC_VALID" "Collecting $VALID_REMAINING remaining Phase-4 validation videos"
    "$ISAACLAB_SH" -p scripts/collect_tcc_dataset.py \
        --num_envs 8 --num_videos "$VALID_REMAINING" --output_root "$TCC_DATA_ROOT" \
        --split valid --class_name phase_4 --frame_stride 5 \
        --max_steps_per_video 1600 --max_attempts 60 \
        --ring_order random --arm_mode random --seed 6404 \
        --headless --device cuda:0 > "$JOB_DIR/collect_tcc_valid.log" 2>&1 || \
        fail "FAILED_TCC_VALID_COLLECTION" "Inspect $JOB_DIR/collect_tcc_valid.log"
fi
[[ "$(video_count valid)" -eq 20 ]] || \
    fail "FAILED_TCC_VALID_COUNT" "Phase-4 validation dataset must contain exactly 20 videos"

if [[ ! -f "$TCC_RUN_DIR/checkpoints/4001.ckpt" ]]; then
    write_status "TRAINING_TCC" "Training Phase-4 TCC for 4000 iterations on the RTX 4090"
    cd "$XIRL_ROOT"
    CUDA_VISIBLE_DEVICES=1 "$CONDA_BIN" run --no-capture-output -n xirl python3 -u pretrain.py \
        --config=configs/xmagical/pretraining/tcc_phase4.py \
        --config.optim.train_max_iters=4000 \
        --experiment_name=random_sim_phase4_tcc --device=cuda:0 \
        > "$JOB_DIR/pretrain_tcc.log" 2>&1 || \
        fail "FAILED_TCC_TRAINING" "Inspect $JOB_DIR/pretrain_tcc.log"
    cd "$REPO_ROOT"
fi

TCC_CHECKPOINT="$(best_tcc_checkpoint)"
[[ -n "$TCC_CHECKPOINT" && -f "$TCC_CHECKPOINT" ]] || \
    fail "FAILED_TCC_CHECKPOINT" "No Phase-4 TCC checkpoint was produced"

if [[ ! -f "$SOURCE_DATASET" ]]; then
    write_status "COLLECTING_BC" "Collecting 160 dense bimanual native-minimal demonstrations"
    "$ISAACLAB_SH" -p scripts/collect_ppo_demos.py \
        --num_envs 8 --num_episodes 160 --output_path "$SOURCE_DATASET" \
        --task_phase phase_4 --phase4_strategy bimanual --action_mask_mode minimal \
        --manipulability_scale 1 --ring_order fixed --arm_mode alternate \
        --seed 7404 --max_steps_per_episode 900 --headless --device cuda:0 \
        > "$JOB_DIR/collect_bc.log" 2>&1 || \
        fail "FAILED_BC_COLLECTION" "Inspect $JOB_DIR/collect_bc.log"
fi

PYTHONPATH=source "$ENV_BIN/python" scripts/check_bc_dataset_mask.py \
    --dataset "$SOURCE_DATASET" --task_phase phase_4 --action_mask_mode minimal \
    > "$JOB_DIR/check_bc_mask.log" 2>&1 || \
    fail "FAILED_BC_MASK_CHECK" "Phase-4 expert labels are not minimal-mask compatible"

if [[ ! -f "$TRAIN_DATASET" || ! -f "$VALID_DATASET" ]]; then
    write_status "SPLITTING_BC" "Creating episode-disjoint 128/32 train-validation split"
    "$ENV_BIN/python" scripts/split_episode_bc_dataset.py \
        --source "$SOURCE_DATASET" --train_output "$TRAIN_DATASET" \
        --valid_output "$VALID_DATASET" --validation_episodes 32 \
        --split_seed 20261006 > "$JOB_DIR/split_bc.log" 2>&1 || \
        fail "FAILED_BC_SPLIT" "Inspect $JOB_DIR/split_bc.log"
fi

if [[ ! -f "$BC_CHECKPOINT" ]]; then
    write_status "TRAINING_BC" "Training Phase-4 bimanual BC for 80 epochs"
    "$ISAACLAB_SH" -p scripts/pretrain_ppo.py \
        --dataset_path "$TRAIN_DATASET" --output_path "$BC_STEM" \
        --epochs 80 --batch_size 256 --lr 1e-3 --seed 42 --headless \
        > "$JOB_DIR/pretrain_bc.log" 2>&1 || \
        fail "FAILED_BC_TRAINING" "Inspect $JOB_DIR/pretrain_bc.log"
fi

write_status "CHECKING_BC" "Checking train/validation exact-action accuracy"
"$ENV_BIN/python" scripts/check_bc_policy.py \
    --checkpoint "$BC_CHECKPOINT" --dataset_path "$TRAIN_DATASET" \
    --normalization raw --task_phase phase_4 --device cuda \
    > "$JOB_DIR/check_bc_train.log" 2>&1 || \
    fail "FAILED_BC_TRAIN_CHECK" "Inspect $JOB_DIR/check_bc_train.log"
"$ENV_BIN/python" scripts/check_bc_policy.py \
    --checkpoint "$BC_CHECKPOINT" --dataset_path "$VALID_DATASET" \
    --normalization raw --task_phase phase_4 --device cuda \
    > "$JOB_DIR/check_bc_valid.log" 2>&1 || \
    fail "FAILED_BC_VALID_CHECK" "Inspect $JOB_DIR/check_bc_valid.log"

write_status "EVALUATING_BC" "Evaluating BC on 20 deterministic Phase-4 episodes"
"$ISAACLAB_SH" -p scripts/eval_run.py \
    --checkpoint "$BC_CHECKPOINT" --tcc_checkpoint "$TCC_CHECKPOINT" \
    --dataset_path "$TCC_DATA_ROOT/train/phase_4" --task_phase phase_4 \
    --episodes 20 --seed 42 --metrics_output "$EVAL_DIR/bc_20ep.csv" \
    --disable_video --quiet_steps --action_masking --action_mask_mode minimal \
    --headless --device cuda:0 > "$JOB_DIR/eval_bc.log" 2>&1 || \
    fail "FAILED_BC_EVALUATION" "Inspect $JOB_DIR/eval_bc.log"

"$ENV_BIN/python" - "$EVAL_DIR/bc_20ep.csv" >> "$JOB_DIR/eval_bc.log" 2>&1 <<'PY' || \
    fail "HALTED_BC_GATE" "BC success is below 50%; PPO was intentionally not started"
import csv
import sys
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
successes = sum(row["success"].lower() == "true" for row in rows)
rate = successes / len(rows) if rows else 0.0
print(f"[phase4 BC gate] success={successes}/{len(rows)} ({rate:.1%})")
if len(rows) != 20 or rate < 0.50:
    raise SystemExit(1)
PY

write_status "TRAINING_PPO" "Training 400k minimal PPO transitions with BC replay anchoring"
"$ISAACLAB_SH" -p scripts/parallel_run.py \
    --num_envs 64 --seed 42 --task_phase phase_4 \
    --goal_dataset_root "$TCC_DATA_ROOT" \
    --pretrained_checkpoint "$BC_CHECKPOINT" --disable_obs_normalization \
    --action_masking --action_mask_mode minimal \
    --bc_replay_dataset "$TRAIN_DATASET" --bc_replay_coef 0.1 \
    --bc_replay_batch_size 256 --bc_replay_decision_fraction 0.5 \
    --gamma 0.999 --gae_lambda 0.999 \
    --freeze_policy_timesteps 150000 --learning_rate 2e-6 \
    --lr_schedule low_high_cosine --freeze_learning_rate 5e-6 \
    --lr_ramp_timesteps 0 --lr_decay_start_fraction 0.70 \
    --final_learning_rate 2e-7 --ent_coef 0.0001 \
    --ent_decay_start_fraction 0.70 --final_ent_coef 0 \
    --total_timesteps 400000 --log_suffix "$PPO_SUFFIX" --headless \
    > "$JOB_DIR/train_ppo.log" 2>&1 || \
    fail "FAILED_PPO" "Inspect $JOB_DIR/train_ppo.log"

write_status "REPORTING_PPO" "Generating Phase-4 PPO training analysis"
"$ENV_BIN/python" evaluation/training_analysis.py \
    --log_dir "$REPO_ROOT/artifacts/logs/random_sb3_log_sim_phase_4_${PPO_SUFFIX}" \
    > "$JOB_DIR/report_ppo.log" 2>&1 || \
    fail "FAILED_PPO_REPORT" "Inspect $JOB_DIR/report_ppo.log"

for required in "$PPO_FINAL" "$PPO_BEST" "$PPO_STEP"; do
    [[ -f "$required" ]] || fail "FAILED_PPO_CHECKPOINT" "Missing checkpoint: $required"
done

write_status "RANKING_PPO" "Ranking BC and PPO checkpoints over 50 deterministic episodes each"
"$ENV_BIN/python" scripts/rank_checkpoints.py \
    --checkpoints "$BC_CHECKPOINT" "$PPO_BEST" "$PPO_STEP" "$PPO_FINAL" \
    --episodes 50 --seed 42 --task_phase phase_4 \
    --tcc_checkpoint "$TCC_CHECKPOINT" \
    --dataset_path "$TCC_DATA_ROOT/train/phase_4" --fixed_rings \
    --action_masking --action_mask_mode minimal --output_dir "$RANK_DIR" \
    --best_name phase4_bimanual_minimal_best_50ep --device cuda:0 \
    > "$JOB_DIR/rank_ppo.log" 2>&1 || \
    fail "FAILED_PPO_RANK" "Inspect $JOB_DIR/rank_ppo.log"

write_status "COMPLETED" "Phase-4 TCC, BC, anchored PPO, report and deterministic ranking completed"
