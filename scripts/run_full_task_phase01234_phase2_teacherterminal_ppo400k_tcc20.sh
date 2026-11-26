#!/usr/bin/env bash
set -euo pipefail

# Paired comparison against the 20-episode seed-20261033 baseline.  Every
# setting is held fixed except for the Phase-2 policy, which is replaced by
# the teacher-terminal PPO checkpoint selected after the conservative 400k
# fine-tuning run.

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"
TCC_ROOT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs"
DATA_ROOT="/mnt/data/aiprah/data/random_sim_dataset_tcc/train"
OUTPUT_DIR="$REPO_ROOT/artifacts/evaluations/full_task_phase01234_phase2_teacherterminal_ppo400k_tcc_random075_20ep_seed20261033"
JOB_DIR="$REPO_ROOT/artifacts/jobs/full_task_phase01234_phase2_teacherterminal_ppo400k_tcc20"
STATUS="$JOB_DIR/status.txt"
LOG="$JOB_DIR/evaluation.log"
GPU_INDEX="${GPU_INDEX:-0}"

PHASE2_CHECKPOINT="$REPO_ROOT/artifacts/evaluations/phase2_teacherterminal_ppo400k_parked_rank50/phase2_teacherterminal_ppo400k_best50.zip"

mkdir -p "$OUTPUT_DIR" "$JOB_DIR"
cd "$REPO_ROOT"
source "$CONDA_SH"
conda activate env_isaaclab
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONUNBUFFERED=1

write_status() {
    {
        printf 'updated_at=%s\n' "$(date -Is)"
        printf 'state=%s\n' "$1"
        printf 'detail=%s\n' "$2"
        printf 'phase2_checkpoint=%s\n' "$PHASE2_CHECKPOINT"
        printf 'output_dir=%s\n' "$OUTPUT_DIR"
        printf 'episodes=20\n'
        printf 'seed=20261033\n'
        printf 'phase_success_mode=tcc\n'
        printf 'ring_randomization_scale=0.75\n'
        printf 'action_mask_mode=minimal\n'
        printf 'gpu_index=%s\n' "$GPU_INDEX"
    } > "$STATUS"
}

on_exit() {
    rc=$?
    if (( rc != 0 )); then
        write_status "FAILED" "Evaluation exited with status $rc; inspect $LOG"
    fi
}
trap on_exit EXIT

[[ -f "$PHASE2_CHECKPOINT" ]] || {
    write_status "MISSING_INPUT" "$PHASE2_CHECKPOINT"
    exit 2
}

write_status "RUNNING" "Paired 20-episode deterministic full-task TCC evaluation"

"$ISAACLAB_SH" -p scripts/eval_full_task.py \
    --phase phase_0 \
        artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_0_adaptive075_minimalmask_finetune_3m.zip \
        "$TCC_ROOT/random_sim_phase0_tcc/checkpoints/4001.ckpt" \
        "$DATA_ROOT/phase_0" \
    --phase phase_1 \
        artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_1_native_minimal_recovery_mixed_replay_probe_300k.zip \
        "$TCC_ROOT/random_sim_phase1_tcc/checkpoints/31000.ckpt" \
        "$DATA_ROOT/phase_1" \
    --phase phase_2 \
        "$PHASE2_CHECKPOINT" \
        "$TCC_ROOT/random_sim_phase2_tcc/checkpoints/4000.ckpt" \
        "$DATA_ROOT/phase_2" \
    --phase phase_3 \
        artifacts/evaluations/phase3_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/phase3_bimanual_minimal_best_50ep.zip \
        "$TCC_ROOT/random_sim_phase3_tcc/checkpoints/2600.ckpt" \
        "$DATA_ROOT/phase_3" \
    --phase phase_4 \
        artifacts/checkpoints/bc/phase4_bc_phase3_teacher_terminal128_80ep_seed42.zip \
        "$TCC_ROOT/random_sim_phase4_tcc/checkpoints/2600.ckpt" \
        "$DATA_ROOT/phase_4" \
    --start_embeddings \
        artifacts/evaluations/phase_router_phase01234_first_frame_20261007/start_embeddings.pt \
    --sequence phase_0 phase_1 phase_2 phase_3 phase_4 \
    --phase_success_mode tcc \
    --tcc_success_threshold phase_0 150 \
    --tcc_success_threshold phase_1 50 \
    --tcc_success_threshold phase_2 600 \
    --tcc_success_threshold phase_3 50 \
    --tcc_success_threshold phase_4 100 \
    --tcc_success_consecutive_steps 5 \
    --tcc_success_min_steps 20 \
    --episodes 20 \
    --seed 20261033 \
    --max_steps_per_phase 900 \
    --episode_length_s 240 \
    --randomize_rings \
    --ring_randomization_scale 0.75 \
    --action_mask_mode minimal \
    --output_dir "$OUTPUT_DIR" \
    --video_frame_stride 5 \
    --video_fps 15 \
    --headless \
    --device cuda:0 \
    > "$LOG" 2>&1

write_status "COMPLETED" "Paired full-task TCC evaluation completed"
trap - EXIT
