#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="train_res/curvature_radius/k_16_cr_0.075_th_0.5_cn_16/k_8/ablation_proto_global_only_s2026"
STAGE_DIR="${RUN_ROOT}/stage1_global_only"
mkdir -p "${STAGE_DIR}" "${RUN_ROOT}/configs_used"
cp configs/pa_rangelm_global_only_stage1.yaml "${RUN_ROOT}/configs_used/stage1_global_only.yaml"
cp configs/pcn.yaml "${RUN_ROOT}/configs_used/pcn.yaml"

# Table 5: w/o Cross-Attention (global-only).
# Stage I is trained from scratch on the same canonical PCN split as the full model.
# The only prototype-branch change is removal of encoder-memory cross-attention.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" train.py \
  --config_path configs/pa_rangelm_global_only_stage1.yaml \
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
  --experiment_name ablation_proto_global_only_s2026/stage1_global_only \
  --use_wandb \
  --wandb_project PA_RangeLM_PCN \
  --wandb_run_name ablation_proto_global_only_stage1_s2026 \
  --wandb_mode online
