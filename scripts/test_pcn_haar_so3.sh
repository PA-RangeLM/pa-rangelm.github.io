#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-}"
RESULT_DIR="pcn_test_results"
LOG_DIR="eval_logs/pcn_uniform_so3_s2026"
CSV_PATH="${RESULT_DIR}/pa_rangelm_uniform_so3_s2026.csv"
LOG_PATH="${LOG_DIR}/pa_rangelm_uniform_so3_s2026.log"

if [[ -z "${MODEL_PATH}" ]] || [[ ! -f "${MODEL_PATH}" ]]; then
  echo "PA-RangeLM Stage-II checkpoint not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ -e "${CSV_PATH}" ]]; then
  if [[ "$(wc -l < "${CSV_PATH}")" -eq 1201 ]]; then
    echo "Complete evaluation CSV already exists; refusing to overwrite: ${CSV_PATH}" >&2
    exit 1
  fi
  suffix="incomplete_$(date +%Y%m%d_%H%M%S)"
  mv "${CSV_PATH}" "${CSV_PATH}.${suffix}"
  if [[ -e "${LOG_PATH}" ]]; then
    mv "${LOG_PATH}" "${LOG_PATH}.${suffix}"
  fi
elif [[ -e "${LOG_PATH}" ]]; then
  mv "${LOG_PATH}" "${LOG_PATH}.incomplete_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

# Full PA-RangeLM checkpoint evaluated on the same deterministic Haar-uniform
# deterministic Haar-uniform SO(3) PCN protocol. No standard-pose test is run.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" \
  evaluate_pcn.py \
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
  --rotation_mode uniform_so3 \
  --torch_lm_iterations 80 \
  --torch_lm_damping 0.001 \
  --torch_lm_dtype float32 \
  --torch_lm_target_converged 0.999 \
  2>&1 | tee "${LOG_PATH}"
