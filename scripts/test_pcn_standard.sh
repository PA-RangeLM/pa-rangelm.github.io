#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-}"

if [[ -z "${MODEL_PATH}" ]] || [[ ! -f "${MODEL_PATH}" ]]; then
  echo "PA-RangeLM Stage-II checkpoint not found: ${MODEL_PATH}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python evaluate.py \
  --config_path configs/pa_rangelm_stage2.yaml \
  --model_path "${MODEL_PATH}" \
  --num_keypoint 8 \
  --keypoint curvature_radius \
  --curve_k 16 \
  --curve_radius 0.075 \
  --curve_thres 0.5 \
  --solver torch_lm \
  --torch_lm_iterations 80 \
  --torch_lm_damping 0.001 \
  --torch_lm_dtype float32 \
  --torch_lm_target_converged 0.999
