#!/usr/bin/env bash
set -uo pipefail

# Goal-agnostic -> minimal masking curriculum for Phase 1 and Phase 2.
# Each completed stage is reported and must pass a held-in training health gate
# before its checkpoint is propagated to the next stage.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
GOAL_DATASET_ROOT="/mnt/data/aiprah/data/random_sim_dataset_tcc"
STAGE_TIMESTEPS=1500000

PHASE1_GUIDED_CHECKPOINT="artifacts/checkpoints/ppo/modelli_salvati_sim/dvrk_ppo_best_terminal_distance_phase_1_bc_raw_no_norm_lr3e-5_ent0.001_freeze500k_5m.zip"
PHASE2_RELAXED_CHECKPOINT="artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_2_relaxed_physicalmask_finetune_1p5m.zip"

P1_STRUCTURED_SUFFIX="goalagnostic_finetune_1p5m"
P1_MINIMAL_SUFFIX="minimalmask_curriculum_finetune_1p5m"
P2_STRUCTURED_SUFFIX="goalagnostic_finetune_1p5m"
P2_MINIMAL_SUFFIX="minimalmask_curriculum_finetune_1p5m"

P1_STRUCTURED_FINAL="artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_1_${P1_STRUCTURED_SUFFIX}.zip"
P2_STRUCTURED_FINAL="artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_2_${P2_STRUCTURED_SUFFIX}.zip"

JOB_DIR="$REPO_ROOT/artifacts/jobs/phase12_mask_curriculum_queue"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/queue.lock"
QUEUE_LOG="$REPO_ROOT/phase12_mask_curriculum_queue.log"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

mkdir -p "$JOB_DIR"
cd "$REPO_ROOT"
export PATH="$ENV_BIN:$PATH"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "A Phase 1/2 mask-curriculum queue is already running." >&2
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
        printf 'queue_log=%s\n' "$QUEUE_LOG"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
    printf '[%s] %s: %s\n' "$(date -Is)" "$state" "$detail"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        write_status "FAILED" "Missing required file: $1"
        exit 2
    fi
}

require_file "$ISAACLAB_SH"
require_file "$PHASE1_GUIDED_CHECKPOINT"
require_file "$PHASE2_RELAXED_CHECKPOINT"
require_file "evaluation/training_analysis.py"

run_training_stage() {
    local phase="$1"
    local mask_mode="$2"
    local suffix="$3"
    local checkpoint="$4"
    local stage_name="${phase}_${mask_mode}"
    local train_log="$REPO_ROOT/train_${phase}_${suffix}.log"
    local log_dir="artifacts/logs/random_sb3_log_sim_${phase}_${suffix}"
    local -a command=(
        "$ISAACLAB_SH" -p scripts/parallel_run.py
        --num_envs 64
        --task_phase "$phase"
        --goal_dataset_root "$GOAL_DATASET_ROOT"
        --pretrained_checkpoint "$checkpoint"
        --disable_obs_normalization
        --action_masking
        --action_mask_mode "$mask_mode"
        --freeze_policy_timesteps 250000
        --learning_rate 1e-5
        --lr_schedule low_high_cosine
        --freeze_learning_rate 3e-6
        --lr_ramp_timesteps 250000
        --lr_decay_start_fraction 0.70
        --final_learning_rate 1e-6
        --ent_coef 0.005
        --ent_decay_start_fraction 0.70
        --final_ent_coef 0.0005
        --total_timesteps "$STAGE_TIMESTEPS"
        --log_suffix "$suffix"
        --headless
    )

    write_status "RUNNING_${stage_name}" "Training log: $train_log"
    if (( DRY_RUN )); then
        printf 'DRY RUN:'
        printf ' %q' "${command[@]}"
        printf '\n'
        return 0
    fi

    if ! "${command[@]}" > "$train_log" 2>&1; then
        write_status "FAILED_${stage_name}" "Training failed; inspect $train_log"
        return 1
    fi

    write_status "REPORTING_${stage_name}" "Generating analysis from $log_dir"
    if ! "$ENV_BIN/python" evaluation/training_analysis.py --log_dir "$log_dir" >> "$QUEUE_LOG" 2>&1; then
        write_status "FAILED_${stage_name}_REPORT" "Report generation failed for $log_dir"
        return 1
    fi

    write_status "CHECKING_${stage_name}" "Evaluating final episode windows"
    if ! "$ENV_BIN/python" - "$log_dir/episode_progress.csv" >> "$QUEUE_LOG" 2>&1 <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="") as stream:
    rows = list(csv.DictReader(stream))

def as_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}

if len(rows) < 200:
    print(f"[health] FAIL: only {len(rows)} completed episodes")
    raise SystemExit(1)

last_500 = rows[-min(500, len(rows)):]
last_100 = rows[-100:]
success_500 = sum(as_bool(row["is_success"]) for row in last_500) / len(last_500)
success_100 = sum(as_bool(row["is_success"]) for row in last_100) / len(last_100)
timeout_100 = sum(as_bool(row["time_limit_truncated"]) for row in last_100) / len(last_100)
print(
    f"[health] last500_success={success_500:.3f} "
    f"last100_success={success_100:.3f} last100_timeout={timeout_100:.3f}"
)
if success_500 < 0.55 or success_100 < 0.50:
    print("[health] FAIL: checkpoint will not be propagated")
    raise SystemExit(1)
print("[health] PASS")
PY
    then
        write_status "HALTED_${stage_name}_HEALTH" "Success gate failed; next stage was not started"
        return 1
    fi

    write_status "COMPLETED_${stage_name}" "Training, report, and health gate completed"
}

write_status "STARTED" "Phase 1 then Phase 2: goal_agnostic -> minimal"

if ! run_training_stage "phase_1" "goal_agnostic" "$P1_STRUCTURED_SUFFIX" "$PHASE1_GUIDED_CHECKPOINT"; then
    exit 1
fi
if (( ! DRY_RUN )); then
    require_file "$P1_STRUCTURED_FINAL"
fi
if ! run_training_stage "phase_1" "minimal" "$P1_MINIMAL_SUFFIX" "$P1_STRUCTURED_FINAL"; then
    exit 1
fi

if ! run_training_stage "phase_2" "goal_agnostic" "$P2_STRUCTURED_SUFFIX" "$PHASE2_RELAXED_CHECKPOINT"; then
    exit 1
fi
if (( ! DRY_RUN )); then
    require_file "$P2_STRUCTURED_FINAL"
fi
if ! run_training_stage "phase_2" "minimal" "$P2_MINIMAL_SUFFIX" "$P2_STRUCTURED_FINAL"; then
    exit 1
fi

write_status "COMPLETED" "All Phase 1/2 curriculum stages and reports are ready"
