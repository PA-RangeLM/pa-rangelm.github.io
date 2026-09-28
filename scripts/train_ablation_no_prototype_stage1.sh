#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ROOT="train_res/curvature_radius/k_16_cr_0.075_th_0.5_cn_16/k_8/ablation_no_prototype_s2026"
STAGE_DIR="${RUN_ROOT}/stage1_no_prototype"
mkdir -p "${STAGE_DIR}" "${RUN_ROOT}/configs_used"
cp configs/pa_rangelm_no_prototype_stage1.yaml "${RUN_ROOT}/configs_used/stage1_no_prototype.yaml"
cp configs/pcn.yaml "${RUN_ROOT}/configs_used/pcn.yaml"

# Stage I uses the standard, unrotated PCN training split. Rotation is applied
# only at final test time, matching the paper's rotated-PCN protocol.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train.py \
  --config_path configs/pa_rangelm_no_prototype_stage1.yaml \
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
  --experiment_name ablation_no_prototype_s2026/stage1_no_prototype \
  --use_wandb \
  --wandb_project PA_RangeLM_PCN \
  --wandb_run_name ablation_no_prototype_stage1_s2026 \
  --wandb_mode online \
  2>&1 | tee "${STAGE_DIR}/console_stage1_train.log"
