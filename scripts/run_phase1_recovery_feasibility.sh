#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_BIN="/home/aiprah/miniconda3/envs/env_isaaclab/bin"
JOB_DIR="$REPO_ROOT/artifacts/jobs/phase1_recovery_feasibility"
STATUS_FILE="$JOB_DIR/status.txt"
LOCK_FILE="$JOB_DIR/job.lock"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$JOB_DIR/$RUN_ID"

mkdir -p "$RUN_DIR"
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
    echo "The Phase-1 recovery feasibility test is already running." >&2
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
        printf 'run_dir=%s\n' "$RUN_DIR"
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

run_case() {
    local initial_state="$1"
    local output="$RUN_DIR/${initial_state}.npz"
    local log="$RUN_DIR/${initial_state}.log"

    write_status "RUNNING_${initial_state}" "Collecting 20 successful scripted recoveries"
    timeout --signal=TERM --kill-after=30s 45m \
        "$ISAACLAB_SH" -p scripts/collect_ppo_demos.py \
        --num_envs 8 \
        --num_episodes 20 \
        --output_path "$output" \
        --task_phase phase_1 \
        --phase1_initial_state "$initial_state" \
        --action_mask_mode minimal \
        --manipulability_scale 1 \
        --arm_mode random \
        --seed 4242 \
        --max_steps_per_episode 800 \
        --headless \
        --device cuda:0 > "$log" 2>&1
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        write_status "FAILED_${initial_state}" "exit_code=$exit_code; inspect $log"
        return "$exit_code"
    fi
    if [[ ! -s "$output" ]]; then
        write_status "FAILED_${initial_state}" "Collector exited without dataset: $output"
        return 4
    fi
}

if ! run_case red3_blue1; then
    exit 1
fi
if ! run_case red1_blue3; then
    exit 1
fi

write_status "COMPLETED" "Both 20-episode recovery directions succeeded"
