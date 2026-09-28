#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_DIR="eval_logs/uncertainty_quality_pcn_independent_s2026"
LOG_PATH="${OUTPUT_DIR}/eval.log"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ -z "${MODEL_PATH}" ]] || [[ ! -f "${MODEL_PATH}" ]]; then
  echo "PA-RangeLM checkpoint not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
  echo "Complete result already exists; refusing to overwrite: ${OUTPUT_DIR}/summary.json" >&2
  exit 1
fi
if [[ -d "${OUTPUT_DIR}" ]]; then
  INCOMPLETE_DIR="${OUTPUT_DIR}.incomplete_$(date +%Y%m%d_%H%M%S)"
  mv "${OUTPUT_DIR}" "${INCOMPLETE_DIR}"
  echo "Moved the previous incomplete run to: ${INCOMPLETE_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pa_rangelm_matplotlib_cache}"
mkdir -p "${MPLCONFIGDIR}"

echo "$(date '+%Y-%m-%d %H:%M:%S') | START | PA-RangeLM uncertainty quality on rotated PCN" | tee "${LOG_PATH}"

set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" \
  evaluate_uncertainty.py \
  --config_path configs/pa_rangelm_stage2.yaml \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
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
  --high_error_fraction 0.10 \
  --coverage_min 0.10 \
  --coverage_step 0.05 \
  --range_pairs_per_shape 16384 \
  --bootstrap_resamples 2000 \
  --bootstrap_seed 2026 \
  2>&1 | tee -a "${LOG_PATH}"
STATUS=${PIPESTATUS[0]}
set -e

echo "$(date '+%Y-%m-%d %H:%M:%S') | END | exit=${STATUS}" | tee -a "${LOG_PATH}"
exit "${STATUS}"
