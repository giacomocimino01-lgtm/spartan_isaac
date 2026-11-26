#!/usr/bin/env bash
set -uo pipefail

# Reproduce the successful Phase-1 lineage for Phase 2:
# native-minimal dense demonstrations -> BC -> full-task minimal PPO ->
# recovery-mixed minimal PPO with weighted dense replay -> deterministic ranks.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
DATA_DIR="artifacts/datasets/bc"

SMOKE_FULL_BI="$DATA_DIR/phase2_native_minimal_smoke_full_bimanual4.npz"
SMOKE_FULL_SERIAL="$DATA_DIR/phase2_native_minimal_smoke_full_serial4.npz"
SMOKE_CENTER2="$DATA_DIR/phase2_native_minimal_smoke_center2_4.npz"
SMOKE_CENTER3="$DATA_DIR/phase2_native_minimal_smoke_center3_4.npz"

FULL_BI_SOURCE="$DATA_DIR/phase2_native_minimal_full_bimanual112.npz"
FULL_BI_TRAIN="$DATA_DIR/phase2_native_minimal_full_bimanual88_train.npz"
FULL_BI_VALID="$DATA_DIR/phase2_native_minimal_full_bimanual24_valid.npz"
FULL_SERIAL_SOURCE="$DATA_DIR/phase2_native_minimal_full_serial48.npz"
FULL_SERIAL_TRAIN="$DATA_DIR/phase2_native_minimal_full_serial40_train.npz"
FULL_SERIAL_VALID="$DATA_DIR/phase2_native_minimal_full_serial8_valid.npz"
FULL_TRAIN="$DATA_DIR/phase2_native_minimal_full128_train.npz"
FULL_VALID="$DATA_DIR/phase2_native_minimal_full32_valid.npz"

CENTER2_BI_SOURCE="$DATA_DIR/phase2_native_minimal_center2_bimanual64.npz"
CENTER2_BI_TRAIN="$DATA_DIR/phase2_native_minimal_center2_bimanual48_train.npz"
CENTER2_BI_VALID="$DATA_DIR/phase2_native_minimal_center2_bimanual16_valid.npz"
CENTER2_SERIAL_SOURCE="$DATA_DIR/phase2_native_minimal_center2_serial32.npz"
CENTER2_SERIAL_TRAIN="$DATA_DIR/phase2_native_minimal_center2_serial24_train.npz"
CENTER2_SERIAL_VALID="$DATA_DIR/phase2_native_minimal_center2_serial8_valid.npz"
CENTER2_TRAIN="$DATA_DIR/phase2_native_minimal_center2_72_train.npz"
CENTER2_VALID="$DATA_DIR/phase2_native_minimal_center2_24_valid.npz"

CENTER3_SOURCE="$DATA_DIR/phase2_native_minimal_center3_serial96.npz"
CENTER3_TRAIN="$DATA_DIR/phase2_native_minimal_center3_72_train.npz"
CENTER3_VALID="$DATA_DIR/phase2_native_minimal_center3_24_valid.npz"
RECOVERY_REPLAY="$DATA_DIR/phase2_native_minimal_recovery_weighted_train.npz"

BC_STEM="artifacts/checkpoints/bc/phase2_bc_native_minimal_full160_dense"
BC_CHECKPOINT="${BC_STEM}.zip"
BC_EVAL_DIR="artifacts/evaluations/phase2_bc_native_minimal_full20ep"
BC_EVAL_CSV="$BC_EVAL_DIR/episodes.csv"

FULL_SUFFIX="native_minimal_dense_replay_stratified50_probe_400k"
FULL_LOG_DIR="artifacts/logs/random_sb3_log_sim_phase_2_${FULL_SUFFIX}"
FULL_TRAIN_LOG="$REPO_ROOT/train_phase2_${FULL_SUFFIX}.log"
FULL_FINAL="artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_2_${FULL_SUFFIX}.zip"
FULL_BEST_DISTANCE="artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_2_${FULL_SUFFIX}.zip"
FULL_STEP="artifacts/checkpoints/ppo/dvrk_ppo_phase_2_${FULL_SUFFIX}_384000_steps.zip"
FULL_RANK_DIR="artifacts/evaluations/phase2_native_minimal_full_probe_50ep"
FULL_RANK_LOG="$REPO_ROOT/rank_phase2_native_minimal_full_probe_50ep.log"
FULL_SELECTED="$FULL_RANK_DIR/phase2_native_minimal_full_best_50ep.zip"
FULL_SELECTED_VEC="$FULL_RANK_DIR/phase2_native_minimal_full_best_50ep_vecnormalize.pkl"

RECOVERY_SUFFIX="native_minimal_recovery_mixed_replay_probe_300k"
RECOVERY_LOG_DIR="artifacts/logs/random_sb3_log_sim_phase_2_${RECOVERY_SUFFIX}"
RECOVERY_TRAIN_LOG="$REPO_ROOT/train_phase2_${RECOVERY_SUFFIX}.log"
RECOVERY_FINAL="artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_2_${RECOVERY_SUFFIX}.zip"
RECOVERY_BEST_DISTANCE="artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_2_${RECOVERY_SUFFIX}.zip"
RECOVERY_EVAL_ROOT="artifacts/evaluations/phase2_native_minimal_recovery_probe"

JOB_DIR="artifacts/jobs/phase2_native_minimal_phase1_recipe_pipeline"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/job.lock"
PIPELINE_LOG="$REPO_ROOT/phase2_native_minimal_phase1_recipe_pipeline.log"

mkdir -p "$JOB_DIR" "$BC_EVAL_DIR" "$RECOVERY_EVAL_ROOT"
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
    echo "The native-minimal Phase-2 Phase-1-recipe pipeline is already running." >&2
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
        printf 'full_train_dataset=%s\n' "$FULL_TRAIN"
        printf 'recovery_replay_dataset=%s\n' "$RECOVERY_REPLAY"
        printf 'bc_checkpoint=%s\n' "$BC_CHECKPOINT"
        printf 'full_selected_checkpoint=%s\n' "$FULL_SELECTED"
        printf 'recovery_final_checkpoint=%s\n' "$RECOVERY_FINAL"
        printf 'evaluation_root=%s\n' "$RECOVERY_EVAL_ROOT"
        printf 'pipeline_log=%s\n' "$PIPELINE_LOG"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

fail() {
    write_status "$1" "$2"
    exit 1
}

collect() {
    local initial_state="$1"
    local strategy="$2"
    local episodes="$3"
    local ring_order="$4"
    local arm_mode="$5"
    local output="$6"
    local seed="$7"
    local log="$8"
    "$ISAACLAB_SH" -p scripts/collect_ppo_demos.py \
        --num_envs 8 --num_episodes "$episodes" \
        --output_path "$output" \
        --task_phase phase_2 --phase2_initial_state "$initial_state" \
        --phase2_strategy "$strategy" \
        --action_mask_mode minimal --manipulability_scale 1 \
        --ring_order "$ring_order" --arm_mode "$arm_mode" \
        --seed "$seed" --headless > "$log" 2>&1
}

split() {
    local source="$1"
    local train="$2"
    local valid="$3"
    local valid_episodes="$4"
    local seed="$5"
    local log="$6"
    "$ENV_BIN/python" scripts/split_episode_bc_dataset.py \
        --source "$source" --train_output "$train" --valid_output "$valid" \
        --validation_episodes "$valid_episodes" --split_seed "$seed" \
        > "$log" 2>&1
}

mask_check() {
    local dataset="$1"
    local log="$2"
    PYTHONPATH=source "$ENV_BIN/python" scripts/check_bc_dataset_mask.py \
        --dataset "$dataset" --task_phase phase_2 \
        --action_mask_mode minimal > "$log" 2>&1
}

evaluate_bc() {
    "$ISAACLAB_SH" -p scripts/eval_run.py \
        --checkpoint "$BC_CHECKPOINT" \
        --tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase2_tcc/checkpoints/4000.ckpt \
        --dataset_path /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_2 \
        --task_phase phase_2 --phase2_initial_state red2_blue2 \
        --episodes 20 --seed 42 --metrics_output "$BC_EVAL_CSV" \
        --disable_video --quiet_steps --action_masking \
        --action_mask_mode minimal --headless > eval_phase2_bc_native_minimal_full20ep.log 2>&1
}

rank_state() {
    local initial_state="$1"
    local episodes="$2"
    local output_dir="$3"
    local best_name="$4"
    local log="$5"
    shift 5
    "$ENV_BIN/python" scripts/rank_checkpoints.py \
        --checkpoints "$@" \
        --episodes "$episodes" --seed 42 --task_phase phase_2 \
        --phase2_initial_state "$initial_state" \
        --tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase2_tcc/checkpoints/4000.ckpt \
        --dataset_path /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_2 \
        --fixed_rings --action_masking --action_mask_mode minimal \
        --output_dir "$output_dir" --best_name "$best_name" \
        --device cuda:0 > "$log" 2>&1
}

: > "$PIPELINE_LOG"

write_status "SMOKE_FULL_BIMANUAL" "Collecting four dense native-minimal bimanual full episodes"
collect red2_blue2 bimanual 4 fixed alternate "$SMOKE_FULL_BI" 9101 \
    collect_phase2_native_minimal_smoke_full_bimanual4.log || \
    fail "FAILED_SMOKE_FULL_BIMANUAL" "Inspect collect_phase2_native_minimal_smoke_full_bimanual4.log"
mask_check "$SMOKE_FULL_BI" check_phase2_native_minimal_smoke_full_bimanual4.log || \
    fail "FAILED_SMOKE_FULL_BIMANUAL_MASK" "Bimanual full labels are not minimal-mask compatible"

write_status "SMOKE_FULL_SERIAL" "Collecting four dense native-minimal serial full episodes"
collect red2_blue2 serial 4 random random "$SMOKE_FULL_SERIAL" 9102 \
    collect_phase2_native_minimal_smoke_full_serial4.log || \
    fail "FAILED_SMOKE_FULL_SERIAL" "Inspect collect_phase2_native_minimal_smoke_full_serial4.log"
mask_check "$SMOKE_FULL_SERIAL" check_phase2_native_minimal_smoke_full_serial4.log || \
    fail "FAILED_SMOKE_FULL_SERIAL_MASK" "Serial full labels are not minimal-mask compatible"

write_status "SMOKE_RECOVERY" "Collecting and validating native-minimal center2 and center3 episodes"
collect center2 bimanual 4 fixed alternate "$SMOKE_CENTER2" 9103 \
    collect_phase2_native_minimal_smoke_center2_4.log || \
    fail "FAILED_SMOKE_CENTER2" "Inspect collect_phase2_native_minimal_smoke_center2_4.log"
collect center3 serial 4 random random "$SMOKE_CENTER3" 9104 \
    collect_phase2_native_minimal_smoke_center3_4.log || \
    fail "FAILED_SMOKE_CENTER3" "Inspect collect_phase2_native_minimal_smoke_center3_4.log"
mask_check "$SMOKE_CENTER2" check_phase2_native_minimal_smoke_center2_4.log || \
    fail "FAILED_SMOKE_CENTER2_MASK" "Center2 labels are not minimal-mask compatible"
mask_check "$SMOKE_CENTER3" check_phase2_native_minimal_smoke_center3_4.log || \
    fail "FAILED_SMOKE_CENTER3_MASK" "Center3 labels are not minimal-mask compatible"

write_status "COLLECTING_FULL_BIMANUAL" "Collecting 112 dense native-minimal asynchronous bimanual demonstrations"
collect red2_blue2 bimanual 112 fixed alternate "$FULL_BI_SOURCE" 9212 \
    collect_phase2_native_minimal_full_bimanual112.log || \
    fail "FAILED_FULL_BIMANUAL_COLLECTION" "Inspect collect_phase2_native_minimal_full_bimanual112.log"
split "$FULL_BI_SOURCE" "$FULL_BI_TRAIN" "$FULL_BI_VALID" 24 20261101 \
    split_phase2_native_minimal_full_bimanual112.log || \
    fail "FAILED_FULL_BIMANUAL_SPLIT" "Inspect split_phase2_native_minimal_full_bimanual112.log"

write_status "COLLECTING_FULL_SERIAL" "Collecting 48 dense native-minimal serial demonstrations with random order and arms"
collect red2_blue2 serial 48 random random "$FULL_SERIAL_SOURCE" 9248 \
    collect_phase2_native_minimal_full_serial48.log || \
    fail "FAILED_FULL_SERIAL_COLLECTION" "Inspect collect_phase2_native_minimal_full_serial48.log"
split "$FULL_SERIAL_SOURCE" "$FULL_SERIAL_TRAIN" "$FULL_SERIAL_VALID" 8 20261102 \
    split_phase2_native_minimal_full_serial48.log || \
    fail "FAILED_FULL_SERIAL_SPLIT" "Inspect split_phase2_native_minimal_full_serial48.log"

write_status "MERGING_FULL" "Building disjoint dense full-task train and validation sets"
"$ENV_BIN/python" scripts/merge_bc_datasets.py \
    --dataset "$FULL_BI_TRAIN" 1 --dataset "$FULL_SERIAL_TRAIN" 1 \
    --output "$FULL_TRAIN" > merge_phase2_native_minimal_full_train.log 2>&1 || \
    fail "FAILED_FULL_TRAIN_MERGE" "Inspect merge_phase2_native_minimal_full_train.log"
"$ENV_BIN/python" scripts/merge_bc_datasets.py \
    --dataset "$FULL_BI_VALID" 1 --dataset "$FULL_SERIAL_VALID" 1 \
    --output "$FULL_VALID" > merge_phase2_native_minimal_full_valid.log 2>&1 || \
    fail "FAILED_FULL_VALID_MERGE" "Inspect merge_phase2_native_minimal_full_valid.log"
mask_check "$FULL_TRAIN" check_phase2_native_minimal_full_train_mask.log || \
    fail "FAILED_FULL_TRAIN_MASK" "Full training labels are not 100% minimal-mask compatible"
mask_check "$FULL_VALID" check_phase2_native_minimal_full_valid_mask.log || \
    fail "FAILED_FULL_VALID_MASK" "Full validation labels are not 100% minimal-mask compatible"

write_status "TRAINING_BC" "Training the dense native-minimal full-task BC for 20 epochs"
"$ISAACLAB_SH" -p scripts/pretrain_ppo.py \
    --dataset_path "$FULL_TRAIN" --output_path "$BC_STEM" \
    --epochs 20 --batch_size 256 --lr 1e-3 --headless \
    > pretrain_phase2_native_minimal_full160_dense.log 2>&1 || \
    fail "FAILED_BC" "Inspect pretrain_phase2_native_minimal_full160_dense.log"

write_status "CHECKING_BC" "Checking full-task BC train and held-out validation accuracy"
"$ENV_BIN/python" scripts/check_bc_policy.py \
    --checkpoint "$BC_CHECKPOINT" --dataset_path "$FULL_TRAIN" \
    --normalization raw --task_phase phase_2 --device cuda \
    > check_phase2_native_minimal_full_train.log 2>&1 || \
    fail "FAILED_BC_TRAIN_CHECK" "Inspect check_phase2_native_minimal_full_train.log"
"$ENV_BIN/python" scripts/check_bc_policy.py \
    --checkpoint "$BC_CHECKPOINT" --dataset_path "$FULL_VALID" \
    --normalization raw --task_phase phase_2 --device cuda \
    > check_phase2_native_minimal_full_valid.log 2>&1 || \
    fail "FAILED_BC_VALID_CHECK" "Inspect check_phase2_native_minimal_full_valid.log"

write_status "EVALUATING_BC" "Requiring the BC to solve at least 50% directly under the minimal mask"
evaluate_bc || fail "FAILED_BC_EVALUATION" "Inspect eval_phase2_bc_native_minimal_full20ep.log"
"$ENV_BIN/python" - "$BC_EVAL_CSV" >> "$PIPELINE_LOG" 2>&1 <<'PY' || \
    fail "HALTED_BC_GATE" "Native-minimal BC stayed below 50%; PPO was not started"
import csv
import sys
with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream))
successes = sum(row["success"].lower() == "true" for row in rows)
rate = successes / len(rows) if rows else 0.0
print(f"[native-minimal BC gate] success={successes}/{len(rows)} ({rate:.1%})")
if len(rows) != 20 or rate < 0.50:
    raise SystemExit(1)
PY

write_status "TRAINING_FULL_PPO" "400k fully-minimal full-task PPO with Phase-1 freeze and dense replay recipe"
"$ISAACLAB_SH" -p scripts/parallel_run.py \
    --num_envs 64 --task_phase phase_2 --phase2_initial_state red2_blue2 \
    --goal_dataset_root /mnt/data/aiprah/data/random_sim_dataset_tcc \
    --pretrained_checkpoint "$BC_CHECKPOINT" --disable_obs_normalization \
    --action_masking --action_mask_mode minimal \
    --bc_replay_dataset "$FULL_TRAIN" --bc_replay_coef 0.1 \
    --bc_replay_batch_size 256 --bc_replay_decision_fraction 0.5 \
    --gamma 0.999 --gae_lambda 0.999 \
    --freeze_policy_timesteps 150000 --learning_rate 2e-6 \
    --lr_schedule low_high_cosine --freeze_learning_rate 5e-6 \
    --lr_ramp_timesteps 0 --lr_decay_start_fraction 0.70 \
    --final_learning_rate 2e-7 --ent_coef 0.0001 \
    --ent_decay_start_fraction 0.70 --final_ent_coef 0 \
    --total_timesteps 400000 --log_suffix "$FULL_SUFFIX" --headless \
    > "$FULL_TRAIN_LOG" 2>&1 || \
    fail "FAILED_FULL_PPO" "Inspect $FULL_TRAIN_LOG"

write_status "REPORTING_FULL_PPO" "Generating the full-task PPO report"
"$ENV_BIN/python" evaluation/training_analysis.py --log_dir "$FULL_LOG_DIR" \
    >> "$FULL_TRAIN_LOG" 2>&1 || \
    fail "FAILED_FULL_REPORT" "Full PPO completed but report generation failed"

for required in "$FULL_FINAL" "$FULL_BEST_DISTANCE" "$FULL_STEP"; do
    [[ -f "$required" ]] || fail "FAILED_FULL_CHECKPOINT" "Missing checkpoint: $required"
done

write_status "RANKING_FULL_PPO" "Selecting the native-minimal base over 50 complete episodes"
rank_state red2_blue2 50 "$FULL_RANK_DIR" phase2_native_minimal_full_best_50ep \
    "$FULL_RANK_LOG" "$BC_CHECKPOINT" "$FULL_BEST_DISTANCE" "$FULL_STEP" "$FULL_FINAL" || \
    fail "FAILED_FULL_RANKING" "Inspect $FULL_RANK_LOG"
[[ -f "$FULL_SELECTED" ]] || fail "FAILED_FULL_SELECTION" "Full-task selected checkpoint is missing"

"$ENV_BIN/python" - "$FULL_RANK_DIR/best_deterministic.json" >> "$PIPELINE_LOG" 2>&1 <<'PY' || \
    fail "HALTED_FULL_GATE" "Best native-minimal full checkpoint stayed below 35%; recovery PPO was not started"
import json
import sys
data = json.load(open(sys.argv[1]))
print(f"[full PPO gate] success={data['success_rate']:.1%}")
if float(data["success_rate"]) < 0.35:
    raise SystemExit(1)
PY

write_status "COLLECTING_CENTER2" "Collecting 96 dense native-minimal center2 demonstrations"
collect center2 bimanual 64 fixed alternate "$CENTER2_BI_SOURCE" 9364 \
    collect_phase2_native_minimal_center2_bimanual64.log || \
    fail "FAILED_CENTER2_BIMANUAL_COLLECTION" "Inspect collect_phase2_native_minimal_center2_bimanual64.log"
split "$CENTER2_BI_SOURCE" "$CENTER2_BI_TRAIN" "$CENTER2_BI_VALID" 16 20261103 \
    split_phase2_native_minimal_center2_bimanual64.log || \
    fail "FAILED_CENTER2_BIMANUAL_SPLIT" "Inspect split_phase2_native_minimal_center2_bimanual64.log"
collect center2 serial 32 random random "$CENTER2_SERIAL_SOURCE" 9332 \
    collect_phase2_native_minimal_center2_serial32.log || \
    fail "FAILED_CENTER2_SERIAL_COLLECTION" "Inspect collect_phase2_native_minimal_center2_serial32.log"
split "$CENTER2_SERIAL_SOURCE" "$CENTER2_SERIAL_TRAIN" "$CENTER2_SERIAL_VALID" 8 20261104 \
    split_phase2_native_minimal_center2_serial32.log || \
    fail "FAILED_CENTER2_SERIAL_SPLIT" "Inspect split_phase2_native_minimal_center2_serial32.log"
"$ENV_BIN/python" scripts/merge_bc_datasets.py \
    --dataset "$CENTER2_BI_TRAIN" 1 --dataset "$CENTER2_SERIAL_TRAIN" 1 \
    --output "$CENTER2_TRAIN" > merge_phase2_native_minimal_center2_train.log 2>&1 || \
    fail "FAILED_CENTER2_TRAIN_MERGE" "Inspect merge_phase2_native_minimal_center2_train.log"
"$ENV_BIN/python" scripts/merge_bc_datasets.py \
    --dataset "$CENTER2_BI_VALID" 1 --dataset "$CENTER2_SERIAL_VALID" 1 \
    --output "$CENTER2_VALID" > merge_phase2_native_minimal_center2_valid.log 2>&1 || \
    fail "FAILED_CENTER2_VALID_MERGE" "Inspect merge_phase2_native_minimal_center2_valid.log"

write_status "COLLECTING_CENTER3" "Collecting 96 dense native-minimal last-ring recovery demonstrations"
collect center3 serial 96 random random "$CENTER3_SOURCE" 9396 \
    collect_phase2_native_minimal_center3_serial96.log || \
    fail "FAILED_CENTER3_COLLECTION" "Inspect collect_phase2_native_minimal_center3_serial96.log"
split "$CENTER3_SOURCE" "$CENTER3_TRAIN" "$CENTER3_VALID" 24 20261105 \
    split_phase2_native_minimal_center3_serial96.log || \
    fail "FAILED_CENTER3_SPLIT" "Inspect split_phase2_native_minimal_center3_serial96.log"

for dataset in "$CENTER2_TRAIN" "$CENTER2_VALID" "$CENTER3_TRAIN" "$CENTER3_VALID"; do
    mask_check "$dataset" "check_$(basename "${dataset%.npz}")_mask.log" || \
        fail "FAILED_RECOVERY_MASK" "Recovery labels are not 100% minimal-mask compatible: $dataset"
done

write_status "MERGING_RECOVERY_REPLAY" "Weighting full x1, center2 x3 and center3 x8 like the Phase-1 recovery recipe"
"$ENV_BIN/python" scripts/merge_bc_datasets.py \
    --dataset "$FULL_TRAIN" 1 --dataset "$CENTER2_TRAIN" 3 \
    --dataset "$CENTER3_TRAIN" 8 --output "$RECOVERY_REPLAY" \
    > merge_phase2_native_minimal_recovery_weighted.log 2>&1 || \
    fail "FAILED_RECOVERY_MERGE" "Inspect merge_phase2_native_minimal_recovery_weighted.log"
mask_check "$RECOVERY_REPLAY" check_phase2_native_minimal_recovery_weighted_mask.log || \
    fail "FAILED_RECOVERY_REPLAY_MASK" "Weighted recovery replay is not minimal-mask compatible"

VEC_ARGS=()
if [[ -f "$FULL_SELECTED_VEC" ]]; then
    VEC_ARGS=(--pretrained_vecnormalize "$FULL_SELECTED_VEC")
fi

write_status "TRAINING_RECOVERY_PPO" "300k fully-minimal PPO on physical center2/center3 reset states with weighted replay"
"$ISAACLAB_SH" -p scripts/parallel_run.py \
    --num_envs 64 --task_phase phase_2 --phase2_initial_state recovery_mixed \
    --goal_dataset_root /mnt/data/aiprah/data/random_sim_dataset_tcc \
    --pretrained_checkpoint "$FULL_SELECTED" "${VEC_ARGS[@]}" \
    --disable_obs_normalization --action_masking --action_mask_mode minimal \
    --bc_replay_dataset "$RECOVERY_REPLAY" --bc_replay_coef 0.1 \
    --bc_replay_batch_size 256 --bc_replay_decision_fraction 0.5 \
    --gamma 0.999 --gae_lambda 0.999 --freeze_policy_timesteps 0 \
    --learning_rate 2e-6 --lr_schedule constant --ent_coef 0.0001 \
    --ent_decay_start_fraction 0.70 --final_ent_coef 0 \
    --total_timesteps 300000 --log_suffix "$RECOVERY_SUFFIX" --headless \
    > "$RECOVERY_TRAIN_LOG" 2>&1 || \
    fail "FAILED_RECOVERY_PPO" "Inspect $RECOVERY_TRAIN_LOG"

write_status "REPORTING_RECOVERY_PPO" "Generating the recovery PPO report"
"$ENV_BIN/python" evaluation/training_analysis.py --log_dir "$RECOVERY_LOG_DIR" \
    >> "$RECOVERY_TRAIN_LOG" 2>&1 || \
    fail "FAILED_RECOVERY_REPORT" "Recovery PPO completed but report generation failed"

for required in "$RECOVERY_FINAL" "$RECOVERY_BEST_DISTANCE"; do
    [[ -f "$required" ]] || fail "FAILED_RECOVERY_CHECKPOINT" "Missing checkpoint: $required"
done
RECOVERY_CHECKPOINTS=("$FULL_SELECTED" "$RECOVERY_BEST_DISTANCE" "$RECOVERY_FINAL")

write_status "EVALUATING_RECOVERY_FULL" "Ranking recovery checkpoints over 50 complete fully-minimal episodes"
rank_state red2_blue2 50 "$RECOVERY_EVAL_ROOT/full_50ep" phase2_native_minimal_recovery_full_best \
    "$REPO_ROOT/rank_phase2_native_minimal_recovery_full_50ep.log" "${RECOVERY_CHECKPOINTS[@]}" || \
    fail "FAILED_RECOVERY_FULL_RANK" "Inspect rank_phase2_native_minimal_recovery_full_50ep.log"

write_status "EVALUATING_RECOVERY_CENTER2" "Ranking recovery checkpoints over 30 center2 episodes"
rank_state center2 30 "$RECOVERY_EVAL_ROOT/center2_30ep" phase2_native_minimal_recovery_center2_best \
    "$REPO_ROOT/rank_phase2_native_minimal_recovery_center2_30ep.log" "${RECOVERY_CHECKPOINTS[@]}" || \
    fail "FAILED_RECOVERY_CENTER2_RANK" "Inspect rank_phase2_native_minimal_recovery_center2_30ep.log"

write_status "EVALUATING_RECOVERY_CENTER3" "Ranking recovery checkpoints over 30 center3 episodes"
rank_state center3 30 "$RECOVERY_EVAL_ROOT/center3_30ep" phase2_native_minimal_recovery_center3_best \
    "$REPO_ROOT/rank_phase2_native_minimal_recovery_center3_30ep.log" "${RECOVERY_CHECKPOINTS[@]}" || \
    fail "FAILED_RECOVERY_CENTER3_RANK" "Inspect rank_phase2_native_minimal_recovery_center3_30ep.log"

write_status "COMPLETED" "Native-minimal BC, full PPO, recovery PPO, reports and deterministic ranks completed"
