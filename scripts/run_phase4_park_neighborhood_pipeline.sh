#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_PYTHON="/home/aiprah/miniconda3/envs/env_isaaclab/bin/python"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"
TCC_DATA_ROOT="/mnt/data/aiprah/data/random_sim_dataset_tcc"
TCC_CHECKPOINT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase4_tcc/checkpoints/2600.ckpt"

DATA_DIR="$REPO_ROOT/artifacts/datasets/bc"
BC_DIR="$REPO_ROOT/artifacts/checkpoints/bc"
JOB_DIR="$REPO_ROOT/artifacts/jobs/phase4_park_neighborhood_pipeline"
EVAL_DIR="$REPO_ROOT/artifacts/evaluations/phase4_park_neighborhood_pipeline"
STATUS="$JOB_DIR/status.txt"

EXACT_SOURCE="$DATA_DIR/phase4_park_exact96_fresh.npz"
NEAR_SOURCE="$DATA_DIR/phase4_park_neighborhood64_fresh.npz"
EXACT_TRAIN="$DATA_DIR/phase4_park_exact80_train.npz"
EXACT_VALID="$DATA_DIR/phase4_park_exact16_valid.npz"
NEAR_TRAIN="$DATA_DIR/phase4_park_neighborhood48_train.npz"
NEAR_VALID="$DATA_DIR/phase4_park_neighborhood16_valid.npz"
EXACT_DECISION_TRAIN="$DATA_DIR/phase4_park_exact80_decision_train.npz"
EXACT_DECISION_VALID="$DATA_DIR/phase4_park_exact16_decision_valid.npz"
NEAR_DECISION_TRAIN="$DATA_DIR/phase4_park_neighborhood48_decision_train.npz"
NEAR_DECISION_VALID="$DATA_DIR/phase4_park_neighborhood16_decision_valid.npz"
BC_ADAPT_DATASET="$DATA_DIR/phase4_park_neighborhood_bc_adapt.npz"
PPO_REPLAY_DATASET="$DATA_DIR/phase4_park_neighborhood_replay.npz"

BASE_CHECKPOINT="$REPO_ROOT/artifacts/evaluations/phase4_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/phase4_bimanual_minimal_best_50ep.zip"
BC_STEM="$BC_DIR/phase4_bc_park_neighborhood_from_canonical98"
BC_CHECKPOINT="${BC_STEM}.zip"
PPO_SUFFIX="park_neighborhood_bc_replay_1m"
PPO_BEST="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_4_${PPO_SUFFIX}.zip"
PPO_FINAL="$REPO_ROOT/artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_4_${PPO_SUFFIX}.zip"
PPO_LOG_DIR="$REPO_ROOT/artifacts/logs/random_sb3_log_sim_phase_4_${PPO_SUFFIX}"
REPORT_DIR="$REPO_ROOT/artifacts/reports/random_sb3_log_sim_phase_4_${PPO_SUFFIX}"

mkdir -p "$DATA_DIR" "$BC_DIR" "$JOB_DIR" "$EVAL_DIR" "$REPORT_DIR"
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
        printf 'phase=phase_4\n'
        printf 'parked_exact_episodes=96\n'
        printf 'parked_neighborhood_episodes=64\n'
        printf 'parked_neighborhood_scale=1.0\n'
        printf 'ppo_neighborhood_probability=0.40\n'
        printf 'base_checkpoint=%s\n' "$BASE_CHECKPOINT"
        printf 'bc_checkpoint=%s\n' "$BC_CHECKPOINT"
        printf 'ppo_best=%s\n' "$PPO_BEST"
        printf 'ppo_final=%s\n' "$PPO_FINAL"
        printf 'evaluation_dir=%s\n' "$EVAL_DIR"
    } > "$STATUS"
}

fail() {
    write_status "$1" "$2"
    printf '[phase4-park-neighborhood] %s: %s\n' "$1" "$2" >&2
    exit 1
}

[[ -f "$BASE_CHECKPOINT" ]] || fail "MISSING_BASE_CHECKPOINT" "$BASE_CHECKPOINT"
[[ -f "$TCC_CHECKPOINT" ]] || fail "MISSING_TCC_CHECKPOINT" "$TCC_CHECKPOINT"

if [[ ! -f "$EXACT_SOURCE" ]]; then
    write_status "COLLECTING_EXACT_PARK" "Collecting 96 fresh exact-parked demonstrations"
    "$ISAACLAB_SH" -p scripts/collect_ppo_demos.py \
        --num_envs 8 --num_episodes 96 --output_path "$EXACT_SOURCE" \
        --task_phase phase_4 --phase4_strategy bimanual \
        --action_mask_mode minimal --manipulability_scale 1 \
        --ring_order fixed --arm_mode alternate --phase_entry_park_reset \
        --seed 9414 --max_steps_per_episode 900 --headless --device cuda:0 \
        > "$JOB_DIR/collect_exact.log" 2>&1 || \
        fail "FAILED_EXACT_COLLECTION" "Inspect $JOB_DIR/collect_exact.log"
fi

if [[ ! -f "$NEAR_SOURCE" ]]; then
    write_status "COLLECTING_PARK_NEIGHBORHOOD" "Collecting 64 demonstrations inside the safe parked ellipsoid"
    "$ISAACLAB_SH" -p scripts/collect_ppo_demos.py \
        --num_envs 8 --num_episodes 64 --output_path "$NEAR_SOURCE" \
        --task_phase phase_4 --phase4_strategy bimanual \
        --action_mask_mode minimal --manipulability_scale 1 \
        --ring_order fixed --arm_mode alternate --phase_entry_park_reset \
        --phase_entry_park_neighborhood_probability 1.0 \
        --phase_entry_park_neighborhood_scale 1.0 \
        --seed 9424 --max_steps_per_episode 900 --headless --device cuda:0 \
        > "$JOB_DIR/collect_neighborhood.log" 2>&1 || \
        fail "FAILED_NEIGHBORHOOD_COLLECTION" "Inspect $JOB_DIR/collect_neighborhood.log"
fi

for dataset in "$EXACT_SOURCE" "$NEAR_SOURCE"; do
    PYTHONPATH=source "$ENV_PYTHON" scripts/check_bc_dataset_mask.py \
        --dataset "$dataset" --task_phase phase_4 --action_mask_mode minimal \
        >> "$JOB_DIR/check_source_masks.log" 2>&1 || \
        fail "FAILED_SOURCE_MASK_CHECK" "$dataset"
done

write_status "SPLITTING_DATA" "Creating distribution-preserving train/validation splits"
"$ENV_PYTHON" scripts/split_episode_bc_dataset.py \
    --source "$EXACT_SOURCE" --train_output "$EXACT_TRAIN" \
    --valid_output "$EXACT_VALID" --validation_episodes 16 \
    --split_seed 20261013 > "$JOB_DIR/split_exact.log" 2>&1 || \
    fail "FAILED_EXACT_SPLIT" "Inspect split_exact.log"
"$ENV_PYTHON" scripts/split_episode_bc_dataset.py \
    --source "$NEAR_SOURCE" --train_output "$NEAR_TRAIN" \
    --valid_output "$NEAR_VALID" --validation_episodes 16 \
    --split_seed 20261014 > "$JOB_DIR/split_neighborhood.log" 2>&1 || \
    fail "FAILED_NEIGHBORHOOD_SPLIT" "Inspect split_neighborhood.log"

"$ENV_PYTHON" scripts/build_decision_bc_dataset.py \
    --source "$EXACT_SOURCE" --train_output "$EXACT_DECISION_TRAIN" \
    --valid_output "$EXACT_DECISION_VALID" --decisions_per_episode 0 \
    --validation_episodes 16 --split_seed 20261013 \
    > "$JOB_DIR/decisions_exact.log" 2>&1 || \
    fail "FAILED_EXACT_DECISIONS" "Inspect decisions_exact.log"
"$ENV_PYTHON" scripts/build_decision_bc_dataset.py \
    --source "$NEAR_SOURCE" --train_output "$NEAR_DECISION_TRAIN" \
    --valid_output "$NEAR_DECISION_VALID" --decisions_per_episode 0 \
    --validation_episodes 16 --split_seed 20261014 \
    > "$JOB_DIR/decisions_neighborhood.log" 2>&1 || \
    fail "FAILED_NEIGHBORHOOD_DECISIONS" "Inspect decisions_neighborhood.log"

# Dense rows teach command persistence; repeated decision rows prevent those
# low-level repetitions from drowning the actual semantic choices.
"$ENV_PYTHON" scripts/merge_bc_datasets.py \
    --dataset "$EXACT_TRAIN" 1 \
    --dataset "$NEAR_TRAIN" 1 \
    --dataset "$EXACT_DECISION_TRAIN" 12 \
    --dataset "$NEAR_DECISION_TRAIN" 12 \
    --output "$BC_ADAPT_DATASET" > "$JOB_DIR/merge_bc_adapt.log" 2>&1 || \
    fail "FAILED_BC_ADAPT_MERGE" "Inspect merge_bc_adapt.log"

"$ENV_PYTHON" scripts/merge_bc_datasets.py \
    --dataset "$EXACT_TRAIN" 1 \
    --dataset "$NEAR_TRAIN" 1 \
    --dataset "$EXACT_DECISION_TRAIN" 4 \
    --dataset "$NEAR_DECISION_TRAIN" 4 \
    --output "$PPO_REPLAY_DATASET" > "$JOB_DIR/merge_ppo_replay.log" 2>&1 || \
    fail "FAILED_PPO_REPLAY_MERGE" "Inspect merge_ppo_replay.log"

PYTHONPATH=source "$ENV_PYTHON" scripts/check_bc_dataset_mask.py \
    --dataset "$BC_ADAPT_DATASET" --task_phase phase_4 \
    --action_mask_mode minimal > "$JOB_DIR/check_adapt_mask.log" 2>&1 || \
    fail "FAILED_ADAPT_MASK_CHECK" "BC adaptation labels violate minimal mask"

write_status "TRAINING_BC" "Adapting the canonical 98% policy to exact and nearby parked starts"
"$ISAACLAB_SH" -p scripts/pretrain_ppo.py \
    --dataset_path "$BC_ADAPT_DATASET" --output_path "$BC_STEM" \
    --initial_checkpoint "$BASE_CHECKPOINT" --task_phase phase_4 \
    --action_mask_mode minimal --epochs 15 --batch_size 256 --lr 1e-5 \
    --seed 42 --headless > "$JOB_DIR/train_bc.log" 2>&1 || \
    fail "FAILED_BC" "Inspect $JOB_DIR/train_bc.log"

write_status "CHECKING_BC_OFFLINE" "Checking held-out exact and neighborhood decision accuracy"
for spec in \
    "exact_dense:$EXACT_VALID" \
    "near_dense:$NEAR_VALID" \
    "exact_decision:$EXACT_DECISION_VALID" \
    "near_decision:$NEAR_DECISION_VALID"; do
    name="${spec%%:*}"
    dataset="${spec#*:}"
    "$ENV_PYTHON" scripts/check_bc_policy.py \
        --checkpoint "$BC_CHECKPOINT" --dataset_path "$dataset" \
        --normalization raw --task_phase phase_4 --device cuda \
        > "$JOB_DIR/check_${name}.log" 2>&1 || \
        fail "FAILED_BC_OFFLINE_CHECK" "$name"
done

run_bc_eval() {
    local name="$1"
    shift
    "$ISAACLAB_SH" -p scripts/eval_run.py \
        --checkpoint "$BC_CHECKPOINT" --tcc_checkpoint "$TCC_CHECKPOINT" \
        --dataset_path "$TCC_DATA_ROOT/train/phase_4" --task_phase phase_4 \
        --episodes 30 --seed 42 --metrics_output "$EVAL_DIR/${name}_30ep.csv" \
        --disable_video --quiet_steps --action_masking --action_mask_mode minimal \
        --headless --device cuda:0 "$@" > "$JOB_DIR/eval_${name}.log" 2>&1 || \
        fail "FAILED_BC_EVAL" "$name"
}

write_status "EVALUATING_BC_GATES" "Evaluating exact parked, neighborhood and canonical starts"
run_bc_eval parked_exact --phase_entry_park_reset
run_bc_eval parked_neighborhood --phase_entry_park_reset \
    --phase_entry_park_neighborhood_probability 1.0 \
    --phase_entry_park_neighborhood_scale 1.0
run_bc_eval canonical

"$ENV_PYTHON" - \
    "$EVAL_DIR/parked_exact_30ep.csv" \
    "$EVAL_DIR/parked_neighborhood_30ep.csv" \
    "$EVAL_DIR/canonical_30ep.csv" <<'PY' > "$JOB_DIR/bc_gate_summary.log" 2>&1 || \
    fail "HALTED_BC_GATE" "Required: exact>=80%, neighborhood>=70%, canonical>=90%"
import csv
import sys

def rate(path):
    rows = list(csv.DictReader(open(path, newline="")))
    return sum(row["success"].lower() == "true" for row in rows) / len(rows), len(rows)

exact, exact_n = rate(sys.argv[1])
near, near_n = rate(sys.argv[2])
canonical, canonical_n = rate(sys.argv[3])
print(f"parked_exact={exact:.1%} ({exact_n} episodes)")
print(f"parked_neighborhood={near:.1%} ({near_n} episodes)")
print(f"canonical={canonical:.1%} ({canonical_n} episodes)")
if exact_n != 30 or near_n != 30 or canonical_n != 30:
    raise SystemExit(1)
if exact < 0.80 or near < 0.70 or canonical < 0.90:
    raise SystemExit(1)
PY

write_status "TRAINING_PPO" "BC gates passed; training 1M PPO on a 60/40 exact/neighborhood reset mixture"
"$ISAACLAB_SH" -p scripts/parallel_run.py \
    --num_envs 64 --seed 42 --task_phase phase_4 \
    --goal_dataset_root "$TCC_DATA_ROOT" \
    --pretrained_checkpoint "$BC_CHECKPOINT" --disable_obs_normalization \
    --phase_entry_park_reset \
    --phase_entry_park_neighborhood_probability 0.40 \
    --phase_entry_park_neighborhood_scale 1.0 \
    --action_masking --action_mask_mode minimal \
    --bc_replay_dataset "$PPO_REPLAY_DATASET" --bc_replay_coef 0.10 \
    --bc_replay_batch_size 256 --bc_replay_decision_fraction 0.50 \
    --gamma 0.999 --gae_lambda 0.999 \
    --freeze_policy_timesteps 100000 --learning_rate 3e-6 \
    --lr_schedule low_high_cosine --freeze_learning_rate 2e-6 \
    --lr_ramp_timesteps 100000 --lr_decay_start_fraction 0.75 \
    --final_learning_rate 3e-7 --ent_coef 0.0001 \
    --ent_decay_start_fraction 0.75 --final_ent_coef 0 \
    --total_timesteps 1000000 --log_suffix "$PPO_SUFFIX" --headless \
    --device cuda:0 > "$JOB_DIR/train_ppo.log" 2>&1 || \
    fail "FAILED_PPO" "Inspect $JOB_DIR/train_ppo.log"

write_status "REPORTING_PPO" "Generating the Phase-4 parked-neighborhood training report"
"$ENV_PYTHON" evaluation/training_analysis.py --log_dir "$PPO_LOG_DIR" \
    > "$JOB_DIR/report_ppo.log" 2>&1 || \
    fail "FAILED_PPO_REPORT" "Inspect report_ppo.log"

for required in "$PPO_BEST" "$PPO_FINAL"; do
    [[ -f "$required" ]] || fail "MISSING_PPO_CHECKPOINT" "$required"
done

rank_mode() {
    local name="$1"
    shift
    "$ENV_PYTHON" scripts/rank_checkpoints.py \
        --checkpoints "$BC_CHECKPOINT" "$PPO_BEST" "$PPO_FINAL" \
        --episodes 50 --seed 42 --task_phase phase_4 \
        --tcc_checkpoint "$TCC_CHECKPOINT" \
        --dataset_path "$TCC_DATA_ROOT/train/phase_4" --fixed_rings \
        --action_masking --action_mask_mode minimal \
        --output_dir "$EVAL_DIR/${name}_rank_50ep" \
        --best_name "phase4_${name}_best_50ep" --device cuda:0 "$@" \
        > "$JOB_DIR/rank_${name}.log" 2>&1 || \
        fail "FAILED_RANK" "$name"
}

write_status "RANKING_PPO" "Ranking BC/PPO on exact parked, neighborhood and canonical resets"
rank_mode parked_exact --phase_entry_park_reset
rank_mode parked_neighborhood --phase_entry_park_reset \
    --phase_entry_park_neighborhood_probability 1.0 \
    --phase_entry_park_neighborhood_scale 1.0
rank_mode canonical

write_status "COMPLETED" "Fresh BC, gated PPO, report and three deterministic rankings completed"
