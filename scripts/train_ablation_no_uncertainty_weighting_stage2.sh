#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE1_BEST="train_res/curvature_radius/k_16_cr_0.075_th_0.5_cn_16/k_8/pa_rangelm_stage1/models/val_best.pth"
RUN_ROOT="train_res/curvature_radius/k_16_cr_0.075_th_0.5_cn_16/k_8/ablation_rangelm_no_uncertainty_weighting_s2026"
STAGE_DIR="${RUN_ROOT}/stage2_no_uncertainty_weighting"

if [[ ! -f "${STAGE1_BEST}" ]]; then
  echo "Shared full-Prototype Stage-I val_best.pth not found: ${STAGE1_BEST}" >&2
  exit 1
fi

mkdir -p "${STAGE_DIR}" "${RUN_ROOT}/configs_used"
cp configs/pa_rangelm_no_uncertainty_weighting_stage2.yaml \
  "${RUN_ROOT}/configs_used/stage2_no_uncertainty_weighting.yaml"

# Table 6: use the shared full-Prototype Stage-I checkpoint and train only
# Stage II for 50 epochs. Retain bounded correction + differentiable LM/XYZ
# supervision, but remove the uncertainty head and use uniform LM weights.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" train.py \
  --config_path configs/pa_rangelm_no_uncertainty_weighting_stage2.yaml \
  --train_config_path configs/pcn.yaml \
  --num_keypoint 8 \
  --keypoint curvature_radius \
  --curve_k 16 \
  --curve_radius 0.075 \
  --curve_thres 0.5 \
  --curvature_neighbor 16 \
  --sample_ratio 1.0 \
  --subset_seed 2026 \
  --seed 2026 \
  --experiment_name ablation_rangelm_no_uncertainty_weighting_s2026/stage2_no_uncertainty_weighting \
  --pretrained_path "${STAGE1_BEST}" \
  --use_wandb \
  --wandb_project PA_RangeLM_PCN \
  --wandb_run_name ablation_no_uncertainty_weighting_stage2_s2026 \
  --wandb_mode online
