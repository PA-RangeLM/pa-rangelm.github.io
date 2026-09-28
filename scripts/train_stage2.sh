#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

STAGE1_BEST="train_res/curvature_radius/k_16_cr_0.075_th_0.5_cn_16/k_8/pa_rangelm_stage1/models/val_best.pth"
if [[ ! -f "${STAGE1_BEST}" ]]; then
  echo "Existing prototype-only Stage 1 checkpoint not found: ${STAGE1_BEST}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train.py \
  --config_path configs/pa_rangelm_stage2.yaml \
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
  --experiment_name pa_rangelm_stage2_s2026 \
  --pretrained_path "${STAGE1_BEST}"
