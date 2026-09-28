#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="train_res/pa_rangelm_stage2_s2026"
MODEL_PATH="${MODEL_PATH:-}"
EVAL_DIR="${RUN_ROOT}/evaluation"
CSV_PATH="${EVAL_DIR}/rotated_pcn_independent_s2026.csv"
LOG_PATH="${EVAL_DIR}/eval_rotated_pcn_independent_s2026.log"

if [[ -z "${MODEL_PATH}" ]] || [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Full PA-RangeLM Stage-II val_best.pth not found: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${EVAL_DIR}"

# Full PA-RangeLM: bounded range correction + uncertainty weighting.
# Evaluate only the complete 1,200-sample PCN test split after deterministic
# independent full-range XYZ rotations (seed 2026).
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" evaluate_pcn.py \
  --config_path configs/pa_rangelm_stage2.yaml \
  --model_path "${MODEL_PATH}" \
  --output_csv "${CSV_PATH}" \
  --data_config configs/pcn.yaml \
  --batch_size 4 \
  --num_keypoint 8 \
  --keypoint curvature_radius \
  --curve_k 16 \
  --curve_radius 0.075 \
  --curve_thres 0.5 \
  --random_rotate \
  --rotation_seed 2026 \
  --rotation_mode independent \
  --torch_lm_iterations 80 \
  --torch_lm_damping 0.001 \
  --torch_lm_dtype float32 \
  --torch_lm_target_converged 0.999 \
  | tee "${LOG_PATH}"
