#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/aiprah/Documents/m_dVrk"
ISAACLAB_SH="/home/aiprah/Documents/IsaacLab/isaaclab.sh"
ENV_PYTHON="/home/aiprah/miniconda3/envs/env_isaaclab/bin/python"
CONDA_SH="/home/aiprah/miniconda3/etc/profile.d/conda.sh"
TCC_ROOT="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs"
DATA_ROOT="/mnt/data/aiprah/data/random_sim_dataset_tcc/train"

PHASE4_CHECKPOINT="$REPO_ROOT/artifacts/checkpoints/bc/phase4_bc_phase3_teacher_terminal128_80ep_seed42.zip"
JOB_DIR="$REPO_ROOT/artifacts/jobs/phase4_teacherterminal_robustness_queue"
PHASE34_ROOT="$REPO_ROOT/artifacts/evaluations/phase34_teacherterminal_bc80_multiseed100"
FULL_DIR="$REPO_ROOT/artifacts/evaluations/full_task_phase01234_teacherterminal_bc80_tcc_random075_20ep_seed20261033"
STATUS="$JOB_DIR/status.txt"
SEEDS=(20261028 20261029 20261030 20261031 20261032)

mkdir -p "$JOB_DIR" "$PHASE34_ROOT" "$FULL_DIR"
cd "$REPO_ROOT"
source "$CONDA_SH"
conda activate env_isaaclab
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

write_status() {
    {
        printf 'updated_at=%s\n' "$(date -Is)"
        printf 'state=%s\n' "$1"
        printf 'detail=%s\n' "$2"
        printf 'phase4_checkpoint=%s\n' "$PHASE4_CHECKPOINT"
        printf 'phase34_root=%s\n' "$PHASE34_ROOT"
        printf 'full_task_dir=%s\n' "$FULL_DIR"
        printf 'phase34_protocol=5_seeds_x_20_episodes_deterministic_symbolic\n'
        printf 'full_task_protocol=20_episodes_deterministic_tcc_random_scale_0.75\n'
    } > "$STATUS"
}

fail() {
    write_status "$1" "$2"
    printf '[phase4-robustness] %s: %s\n' "$1" "$2" >&2
    exit 1
}

[[ -f "$PHASE4_CHECKPOINT" ]] || fail "MISSING_INPUT" "$PHASE4_CHECKPOINT"

for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    run=$((index + 1))
    output="$PHASE34_ROOT/seed${seed}_20ep"
    write_status "PHASE34_MULTISEED_${run}_OF_5" "Running seed $seed: 20 deterministic episodes"
    EPISODES=20 \
    SEED="$seed" \
    PHASE4_CHECKPOINT="$PHASE4_CHECKPOINT" \
    RUN_LABEL="teacherterminal_bc80_seed${seed}" \
        bash scripts/run_phase34_true_handoff_bc_eval.sh "$output" \
        > "$JOB_DIR/phase34_seed${seed}.log" 2>&1 || \
        fail "FAILED_PHASE34_SEED_${seed}" "Inspect $JOB_DIR/phase34_seed${seed}.log"
done

write_status "SUMMARIZING_PHASE34" "Aggregating 100 deterministic Phase-3 to Phase-4 episodes"
"$ENV_PYTHON" - "$PHASE34_ROOT" "${SEEDS[@]}" <<'PY' > "$JOB_DIR/phase34_multiseed_summary.json"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
seeds = [int(value) for value in sys.argv[2:]]
runs = []
for seed in seeds:
    path = root / f"seed{seed}_20ep" / "summary.json"
    with path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    runs.append(
        {
            "seed": seed,
            "episodes": summary["episodes"],
            "successes": summary["successes"],
            "success_rate": summary["success_rate"],
            "router_queries": summary["router"]["queries"],
            "router_correct": summary["router"]["correct"],
        }
    )

episodes = sum(run["episodes"] for run in runs)
successes = sum(run["successes"] for run in runs)
router_queries = sum(run["router_queries"] for run in runs)
router_correct = sum(run["router_correct"] for run in runs)
result = {
    "protocol": "phase3_to_phase4_deterministic_symbolic_multiseed",
    "runs": runs,
    "episodes": episodes,
    "successes": successes,
    "success_rate": successes / episodes,
    "router_queries": router_queries,
    "router_correct": router_correct,
    "router_accuracy": router_correct / router_queries,
}
print(json.dumps(result, indent=2))
PY

write_status "FULL_TASK_TCC" "Running 20 deterministic Phase-0 to Phase-4 episodes with initial randomization 0.75"
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
        artifacts/evaluations/phase2_native_minimal_full_probe_50ep/phase2_native_minimal_full_best_50ep.zip \
        "$TCC_ROOT/random_sim_phase2_tcc/checkpoints/4000.ckpt" \
        "$DATA_ROOT/phase_2" \
    --phase phase_3 \
        artifacts/evaluations/phase3_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/phase3_bimanual_minimal_best_50ep.zip \
        "$TCC_ROOT/random_sim_phase3_tcc/checkpoints/2600.ckpt" \
        "$DATA_ROOT/phase_3" \
    --phase phase_4 \
        "$PHASE4_CHECKPOINT" \
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
    --output_dir "$FULL_DIR" \
    --video_frame_stride 5 \
    --video_fps 15 \
    --headless \
    --device cuda:0 \
    > "$JOB_DIR/full_task_tcc_random075.log" 2>&1 || \
    fail "FAILED_FULL_TASK" "Inspect $JOB_DIR/full_task_tcc_random075.log"

write_status "COMPLETED" "Multiseed Phase-3 to Phase-4 and randomized full-task evaluations completed"

