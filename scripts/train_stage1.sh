#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train.py \
  --config_path configs/pa_rangelm_stage1.yaml \
  --train_config_path configs/pcn.yaml --num_keypoint 8 \
  --keypoint curvature_radius --curve_k 16 --curve_radius 0.075 \
  --curve_thres 0.5 --curvature_neighbor 16 --sample_ratio 1.0 \
  --subset_seed 2026 --seed 2026 --experiment_name pa_rangelm_stage1
