#!/usr/bin/env bash
set -euo pipefail

episodes="${1:?episodes required}"
seed="${2:?seed required}"
output_dir="${3:?output directory required}"

repo=/home/aiprah/Documents/m_dVrk
isaaclab=/home/aiprah/Documents/IsaacLab/isaaclab.sh

source /home/aiprah/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd "$repo"

exec "$isaaclab" -p scripts/eval_full_task.py \
  --headless \
  --device cuda:0 \
  --phase phase_0 \
    /home/aiprah/Documents/m_dVrk/artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_0_adaptive075_minimalmask_finetune_3m.zip \
    /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase0_tcc/checkpoints/4001.ckpt \
    /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_0 \
  --phase phase_1 \
    /home/aiprah/Documents/m_dVrk/artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_1_native_minimal_recovery_mixed_replay_probe_300k.zip \
    /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase1_tcc/checkpoints/31000.ckpt \
    /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_1 \
  --phase phase_2 \
    /home/aiprah/Documents/m_dVrk/artifacts/evaluations/phase2_teacherterminal_ppo400k_parked_rank50/phase2_teacherterminal_ppo400k_best50.zip \
    /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase2_tcc/checkpoints/4000.ckpt \
    /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_2 \
  --phase phase_3 \
    /home/aiprah/Documents/m_dVrk/artifacts/evaluations/phase3_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/phase3_bimanual_minimal_best_50ep.zip \
    /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase3_tcc/checkpoints/2600.ckpt \
    /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_3 \
  --phase phase_4 \
    /home/aiprah/Documents/m_dVrk/artifacts/checkpoints/bc/phase4_bc_phase3_teacher_terminal128_80ep_seed42.zip \
    /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase4_tcc/checkpoints/2600.ckpt \
    /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_4 \
  --sequence phase_0 phase_1 phase_2 phase_3 phase_4 \
  --router_mode calibrated \
  --phase_success_mode symbolic \
  --calibrated_router_artifact /home/aiprah/Documents/m_dVrk/artifacts/evaluations/phase_router_handoff12_recalibrated_grid_20261117/calibrated_router_handoff_L1.pt \
  --soft_end_reference_bank /home/aiprah/Documents/m_dVrk/artifacts/evaluations/soft_alignment_reference_bank_phase0123_train_20261119.pt \
  --soft_end_temperature 5.0 \
  --soft_end_region_mode last_frames \
  --soft_end_last_frames 2 \
  --soft_end_reference_capture_hz 2.0 \
  --transition_threshold phase_0 -2.0 \
  --transition_threshold phase_1 -2.0 \
  --transition_threshold phase_2 -2.0 \
  --transition_threshold phase_3 -2.0 \
  --transition_margin 0.0 \
  --start_probability_threshold 0.6 \
  --transition_consecutive_steps 3 \
  --transition_consecutive_steps_phase phase_0 4 \
  --transition_consecutive_steps_phase phase_1 5 \
  --transition_consecutive_steps_phase phase_2 4 \
  --disable_transition_prior \
  --action_mask_mode minimal \
  --episodes "$episodes" \
  --seed "$seed" \
  --output_dir "$output_dir"
