#!/usr/bin/env bash
set -euo pipefail

experiment_name="real_only_phase0_tcc_uniform80_v2"
experiment_root="/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs"
checkpoint="${experiment_root}/${experiment_name}/checkpoints/4001.ckpt"
repo_root="/home/aiprah/Documents/m_dVrk"
output_dir="${repo_root}/artifacts/reports/tcc_phase0_sim_real_realonly_eval_20261102"

echo "[queue] Waiting for ${experiment_name}"
while pgrep -f "[p]retrain.py.*--experiment_name=${experiment_name}" >/dev/null; do
  sleep 30
done

if [[ ! -f "${checkpoint}" ]]; then
  echo "[queue] Training ended without the expected checkpoint: ${checkpoint}" >&2
  exit 1
fi

mkdir -p "${output_dir}"
cd "${repo_root}"

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
PYTHONUNBUFFERED=1 \
/home/aiprah/miniconda3/envs/env_isaaclab/bin/python \
  evaluation/tcc_sim_real_analysis.py \
  --checkpoint "sim_only_4001=${experiment_root}/random_sim_phase0_tcc/checkpoints/4001.ckpt" \
  --checkpoint "sim_real_4001=${experiment_root}/sim_real_phase0_tcc_uniform80_v1/checkpoints/4001.ckpt" \
  --checkpoint "real_only_4001=${checkpoint}" \
  --sim-train /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_0 \
  --sim-valid /mnt/data/aiprah/data/random_sim_dataset_tcc/valid/phase_0 \
  --mixed-train /mnt/data/aiprah/data/sim_real_dataset_tcc_uniform80_v1/train/phase_0 \
  --mixed-valid /mnt/data/aiprah/data/sim_real_dataset_tcc_uniform80_v1/valid/phase_0 \
  --frames-per-video 40 \
  --batch-size 64 \
  --device cuda:0 \
  --output-json "${output_dir}/results.json" \
  --output-markdown "${output_dir}/report.md"

echo "[queue] Evaluation completed: ${output_dir}/report.md"
