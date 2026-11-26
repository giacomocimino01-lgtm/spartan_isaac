#!/usr/bin/env bash
set -uo pipefail

# Controlled comparison across Phase 0/1/2:
#   160 fresh serial minimal demonstrations per phase (128 train / 32 valid)
#   one reproducible BC per phase with the same optimizer recipe
#   1M PPO transitions from BC, no replay/KL/curriculum, three PPO seeds
#   100 deterministic final-checkpoint episodes per phase/seed

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
PROTOCOL="controlled_serial_minimal_1m"
SEEDS=(101 202 303)
PHASES=(phase_0 phase_1 phase_2)
DATA_SEED=4242
BC_SEED=42
EVAL_SEED=9001
TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=100

DATA_DIR="artifacts/datasets/bc/controlled_serial_minimal"
BC_DIR="artifacts/checkpoints/bc"
EVAL_ROOT="artifacts/evaluations/${PROTOCOL}_3seed"
REPORT_DIR="artifacts/reports/${PROTOCOL}_3seed"
JOB_DIR="artifacts/jobs/${PROTOCOL}_3seed"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/job.lock"
PIPELINE_LOG="$JOB_DIR/pipeline.log"

mkdir -p "$DATA_DIR" "$BC_DIR" "$EVAL_ROOT" "$REPORT_DIR" "$JOB_DIR"
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
    echo "The controlled serial-minimal protocol is already running." >&2
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
        printf 'protocol=%s\n' "$PROTOCOL"
        printf 'phases=%s\n' "${PHASES[*]}"
        printf 'training_seeds=%s\n' "${SEEDS[*]}"
        printf 'total_timesteps_per_run=%s\n' "$TOTAL_TIMESTEPS"
        printf 'evaluation_episodes_per_run=%s\n' "$EVAL_EPISODES"
        printf 'pipeline_log=%s\n' "$PIPELINE_LOG"
        printf 'evaluation_root=%s\n' "$EVAL_ROOT"
        printf 'report_dir=%s\n' "$REPORT_DIR"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

fail() {
    write_status "$1" "$2"
    echo "$2" >&2
    exit 1
}

dataset_source() { printf '%s/%s_serial_minimal_160.npz' "$DATA_DIR" "$1"; }
dataset_train() { printf '%s/%s_serial_minimal_train128.npz' "$DATA_DIR" "$1"; }
dataset_valid() { printf '%s/%s_serial_minimal_valid32.npz' "$DATA_DIR" "$1"; }
bc_stem() { printf '%s/%s_bc_serial_minimal_train128_seed%s' "$BC_DIR" "$1" "$BC_SEED"; }

tcc_checkpoint() {
    case "$1" in
        phase_0) printf '%s' '/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase0_tcc/checkpoints/3800.ckpt' ;;
        phase_1) printf '%s' '/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase1_tcc/checkpoints/31000.ckpt' ;;
        phase_2) printf '%s' '/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase2_tcc/checkpoints/4000.ckpt' ;;
        *) return 1 ;;
    esac
}

collect_dataset() {
    local phase="$1"
    local output="$2"
    local log="$3"
    local extra=()
    case "$phase" in
        phase_0) extra=(--target_peg peg_green) ;;
        phase_1) extra=(--phase1_initial_state stack_green) ;;
        phase_2) extra=(--phase2_initial_state red2_blue2 --phase2_strategy serial) ;;
        *) return 2 ;;
    esac
    "$ISAACLAB_SH" -p scripts/collect_ppo_demos.py \
        --num_envs 8 --num_episodes 160 \
        --output_path "$output" --task_phase "$phase" \
        --action_mask_mode minimal --manipulability_scale 1 \
        --ring_order random --arm_mode random \
        --seed "$DATA_SEED" --headless "${extra[@]}" > "$log" 2>&1
}

validate_dataset() {
    local phase="$1"
    local dataset="$2"
    local log="$3"
    PYTHONPATH=source "$ENV_BIN/python" scripts/check_bc_dataset_mask.py \
        --dataset "$dataset" --task_phase "$phase" \
        --action_mask_mode minimal >> "$log" 2>&1 || return 1
    "$ENV_BIN/python" - "$dataset" "$phase" >> "$log" 2>&1 <<'PY'
import sys
import numpy as np

path, phase = sys.argv[1:]
data = np.load(path, allow_pickle=True)
required = {"obs", "actions", "episode_ids", "decision_indices", "metadata"}
missing = required - set(data.files)
if missing:
    raise SystemExit(f"missing dataset fields: {sorted(missing)}")
obs = data["obs"]
actions = data["actions"]
episode_ids = data["episode_ids"]
metadata = data["metadata"].item()
episodes = len(np.unique(episode_ids))
bimanual = (actions[:, 0] != 3) & (actions[:, 2] != 3)
print(
    f"[dataset-gate] phase={phase} episodes={episodes} transitions={len(actions)} "
    f"bimanual_rows={int(bimanual.sum())} metadata={metadata}"
)
if obs.ndim != 2 or obs.shape[1] != 244 or actions.shape != (len(obs), 4):
    raise SystemExit("invalid observation/action shapes")
if episodes != 160 or int(metadata.get("successful_episodes", -1)) != 160:
    raise SystemExit("dataset does not contain exactly 160 successful episodes")
if metadata.get("task_phase") != phase:
    raise SystemExit("phase metadata mismatch")
if metadata.get("action_mask_mode") != "minimal":
    raise SystemExit("dataset was not collected natively under minimal semantics")
if metadata.get("ring_order") != "random" or metadata.get("arm_mode") != "random":
    raise SystemExit("expert-order metadata mismatch")
if metadata.get("record_mode") != "all_steps":
    raise SystemExit("dense all-step demonstrations are required")
if bimanual.any():
    raise SystemExit("controlled expert dataset contains bimanual actions")
PY
}

split_dataset() {
    local source="$1"
    local train="$2"
    local valid="$3"
    local log="$4"
    "$ENV_BIN/python" scripts/split_episode_bc_dataset.py \
        --source "$source" --train_output "$train" --valid_output "$valid" \
        --validation_episodes 32 --split_seed 20260927 >> "$log" 2>&1
}

evaluate_checkpoint() {
    local phase="$1"
    local checkpoint="$2"
    local episodes="$3"
    local seed="$4"
    local output="$5"
    local log="$6"
    local extra=()
    case "$phase" in
        phase_1) extra=(--phase1_initial_state stack_green) ;;
        phase_2) extra=(--phase2_initial_state red2_blue2) ;;
    esac
    mkdir -p "$(dirname "$output")"
    "$ISAACLAB_SH" -p scripts/eval_run.py \
        --checkpoint "$checkpoint" \
        --tcc_checkpoint "$(tcc_checkpoint "$phase")" \
        --dataset_path "/mnt/data/aiprah/data/random_sim_dataset_tcc/train/$phase" \
        --task_phase "$phase" --episodes "$episodes" --seed "$seed" \
        --metrics_output "$output" --disable_video --quiet_steps \
        --action_masking --action_mask_mode minimal --headless \
        "${extra[@]}" > "$log" 2>&1
}

run_ppo() {
    local phase="$1"
    local seed="$2"
    local checkpoint="$3"
    local suffix="$4"
    local log="$5"
    local extra=()
    case "$phase" in
        phase_1) extra=(--phase1_initial_state stack_green) ;;
        phase_2) extra=(--phase2_initial_state red2_blue2) ;;
    esac
    "$ISAACLAB_SH" -p scripts/parallel_run.py \
        --num_envs 64 --seed "$seed" --task_phase "$phase" \
        --goal_dataset_root /mnt/data/aiprah/data/random_sim_dataset_tcc \
        --pretrained_checkpoint "$checkpoint" --disable_obs_normalization \
        --action_masking --action_mask_mode minimal \
        --gamma 0.999 --gae_lambda 0.999 \
        --freeze_policy_timesteps 150000 \
        --learning_rate 3e-5 --lr_schedule low_high_cosine \
        --freeze_learning_rate 3e-6 --lr_ramp_timesteps 150000 \
        --lr_decay_start_fraction 0.70 --final_learning_rate 3e-6 \
        --ent_coef 0.001 --ent_decay_start_fraction 0.70 --final_ent_coef 0 \
        --total_timesteps "$TOTAL_TIMESTEPS" --log_suffix "$suffix" \
        --headless "${extra[@]}" > "$log" 2>&1
}

: > "$PIPELINE_LOG"
{
    printf 'protocol=%s\n' "$PROTOCOL"
    printf 'created_at=%s\n' "$(date -Is)"
    printf 'phases=%s\n' "${PHASES[*]}"
    printf 'training_seeds=%s\n' "${SEEDS[*]}"
    printf 'data_seed=%s\n' "$DATA_SEED"
    printf 'bc_seed=%s\n' "$BC_SEED"
    printf 'evaluation_seed=%s\n' "$EVAL_SEED"
    printf 'expert=serial,minimal,random_order,random_feasible_arm,160_successes\n'
    printf 'split=128_train,32_validation\n'
    printf 'bc=20_epochs,batch256,lr1e-3\n'
    printf 'ppo=1M,no_replay,no_KL,no_curriculum,fixed_canonical_resets\n'
    printf 'ppo_hyperparameters=gamma0.999,gae0.999,freeze150k,lr3e-6_to_3e-5_to_3e-6,entropy0.001_to_0\n'
    printf 'evaluation=final_checkpoint,deterministic,100_episodes\n'
} > "$JOB_DIR/protocol_manifest.txt"

for phase in "${PHASES[@]}"; do
    source_dataset="$(dataset_source "$phase")"
    train_dataset="$(dataset_train "$phase")"
    valid_dataset="$(dataset_valid "$phase")"
    data_log="$JOB_DIR/${phase}_dataset.log"
    if [[ ! -f "$source_dataset" ]]; then
        write_status "COLLECTING_BC" "$phase: collecting 160 fresh serial-minimal demonstrations"
        collect_dataset "$phase" "$source_dataset" "$data_log" || \
            fail "FAILED_COLLECTION" "$phase collection failed; inspect $data_log"
    fi
    validate_dataset "$phase" "$source_dataset" "$data_log" || \
        fail "FAILED_DATASET_GATE" "$phase dataset failed serial/minimal validation"
    if [[ ! -f "$train_dataset" || ! -f "$valid_dataset" ]]; then
        write_status "SPLITTING_BC" "$phase: splitting 128 train / 32 validation episodes"
        split_dataset "$source_dataset" "$train_dataset" "$valid_dataset" "$data_log" || \
            fail "FAILED_SPLIT" "$phase dataset split failed"
    fi
done

for phase in "${PHASES[@]}"; do
    train_dataset="$(dataset_train "$phase")"
    valid_dataset="$(dataset_valid "$phase")"
    stem="$(bc_stem "$phase")"
    checkpoint="${stem}.zip"
    bc_log="$JOB_DIR/${phase}_bc_train.log"
    if [[ ! -f "$checkpoint" ]]; then
        write_status "TRAINING_BC" "$phase: common 20-epoch BC recipe"
        "$ISAACLAB_SH" -p scripts/pretrain_ppo.py \
            --dataset_path "$train_dataset" --output_path "$stem" \
            --epochs 20 --batch_size 256 --lr 1e-3 --seed "$BC_SEED" \
            --headless > "$bc_log" 2>&1 || \
            fail "FAILED_BC" "$phase BC training failed; inspect $bc_log"
    fi
    write_status "CHECKING_BC" "$phase: offline train/validation checks"
    "$ENV_BIN/python" scripts/check_bc_policy.py \
        --checkpoint "$checkpoint" --dataset_path "$train_dataset" \
        --normalization raw --task_phase "$phase" --device cuda \
        > "$JOB_DIR/${phase}_bc_train_check.log" 2>&1 || \
        fail "FAILED_BC_CHECK" "$phase BC train check failed"
    "$ENV_BIN/python" scripts/check_bc_policy.py \
        --checkpoint "$checkpoint" --dataset_path "$valid_dataset" \
        --normalization raw --task_phase "$phase" --device cuda \
        > "$JOB_DIR/${phase}_bc_valid_check.log" 2>&1 || \
        fail "FAILED_BC_CHECK" "$phase BC validation check failed"
    bc_eval="$EVAL_ROOT/$phase/bc_seed${BC_SEED}/episodes.csv"
    if [[ ! -f "$bc_eval" ]]; then
        write_status "EVALUATING_BC" "$phase: 100 deterministic BC baseline episodes"
        evaluate_checkpoint "$phase" "$checkpoint" 100 "$EVAL_SEED" \
            "$bc_eval" "$JOB_DIR/${phase}_bc_eval.log" || \
            fail "FAILED_BC_EVAL" "$phase deterministic BC evaluation failed"
    fi
done

for phase in "${PHASES[@]}"; do
    checkpoint="$(bc_stem "$phase").zip"
    for seed in "${SEEDS[@]}"; do
        suffix="${PROTOCOL}_seed${seed}"
        train_log="$JOB_DIR/${phase}_seed${seed}_train.log"
        final_checkpoint="artifacts/checkpoints/ppo/dvrk_ppo_finale_${phase}_${suffix}.zip"
        if [[ ! -f "$final_checkpoint" ]]; then
            write_status "TRAINING_PPO" "$phase seed=$seed: 1M controlled transitions"
            run_ppo "$phase" "$seed" "$checkpoint" "$suffix" "$train_log" || \
                fail "FAILED_PPO" "$phase seed=$seed PPO failed; inspect $train_log"
        fi
        log_dir="artifacts/logs/random_sb3_log_sim_${phase}_${suffix}"
        "$ENV_BIN/python" evaluation/training_analysis.py --log_dir "$log_dir" \
            >> "$train_log" 2>&1 || \
            fail "FAILED_TRAINING_REPORT" "$phase seed=$seed report generation failed"
        eval_csv="$EVAL_ROOT/$phase/seed_${seed}/episodes.csv"
        if [[ ! -f "$eval_csv" ]]; then
            write_status "EVALUATING_PPO" "$phase seed=$seed: 100 deterministic final-checkpoint episodes"
            evaluate_checkpoint "$phase" "$final_checkpoint" "$EVAL_EPISODES" \
                "$EVAL_SEED" "$eval_csv" "$JOB_DIR/${phase}_seed${seed}_eval.log" || \
                fail "FAILED_PPO_EVAL" "$phase seed=$seed evaluation failed"
        fi
    done
done

write_status "AGGREGATING" "Generating cross-phase, cross-seed metrics and learning curves"
"$ENV_BIN/python" evaluation/controlled_protocol_analysis.py \
    --evaluation_root "$EVAL_ROOT" \
    --log_root artifacts/logs \
    --seeds "${SEEDS[@]}" \
    --suffix_prefix "$PROTOCOL" \
    --output_dir "$REPORT_DIR" \
    --total_timesteps "$TOTAL_TIMESTEPS" > "$JOB_DIR/aggregate.log" 2>&1 || \
    fail "FAILED_AGGREGATION" "Controlled protocol completed but aggregation failed"

write_status "COMPLETED" "All 9 PPO runs and 900 deterministic evaluation episodes completed"
printf '[controlled-protocol] completed at %s\n' "$(date -Is)" >> "$PIPELINE_LOG"
