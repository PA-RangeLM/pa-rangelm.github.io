#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="train_res/curvature_radius/k_16_cr_0.075_th_0.5_cn_16/k_8/ablation_proto_memory_only_s2026"
STAGE1_BEST="${RUN_ROOT}/stage1_memory_only/models/val_best.pth"
STAGE_DIR="${RUN_ROOT}/stage2_rangelm"

if [[ ! -f "${STAGE1_BEST}" ]]; then
  echo "Stage-I memory-only val_best.pth not found: ${STAGE1_BEST}" >&2
  echo "Run scripts/train_ablation_proto_memory_only_stage1.sh first." >&2
  exit 1
fi

mkdir -p "${STAGE_DIR}" "${RUN_ROOT}/configs_used"
cp configs/pa_rangelm_memory_only_stage2.yaml "${RUN_ROOT}/configs_used/stage2_memory_only_rangelm.yaml"

# Use the same Stage-II RangeLM protocol as the other prototype ablations.
# The matching memory-only val_best checkpoint initializes Stage II.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" train.py \
  --config_path configs/pa_rangelm_memory_only_stage2.yaml \
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
  --experiment_name ablation_proto_memory_only_s2026/stage2_rangelm \
  --pretrained_path "${STAGE1_BEST}" \
  --use_wandb \
  --wandb_project PA_RangeLM_PCN \
  --wandb_run_name ablation_proto_memory_only_stage2_s2026 \
  --wandb_mode online
