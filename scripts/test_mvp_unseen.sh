#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-}"
RESULT_PREFIX="mvp_results/ours_rotated_unseen8_independent_s2026"
CSV_PATH="${RESULT_PREFIX}.csv"
LOG_PATH="logs/MVP/ours_rotated_unseen8_independent_s2026.log"

if [[ -z "${MODEL_PATH}" ]] || [[ ! -f "${MODEL_PATH}" ]]; then
  echo "PA-RangeLM Stage-II checkpoint not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ -e "${CSV_PATH}" ]] && [[ "$(wc -l < "${CSV_PATH}")" -eq 10401 ]]; then
  echo "Complete unseen-8 CSV already exists; refusing to overwrite: ${CSV_PATH}" >&2
  exit 1
fi

mkdir -p mvp_results logs/MVP
SUFFIX="incomplete_$(date +%Y%m%d_%H%M%S)"
for path in "${CSV_PATH}" "${RESULT_PREFIX}.json" \
  "${RESULT_PREFIX}_per_category.csv" "${LOG_PATH}"; do
  if [[ -e "${path}" ]]; then
    mv "${path}" "${path}.${SUFFIX}"
  fi
done

# Use the unseen samples and deterministic Table-1 rotations.
PYTHONUNBUFFERED=1 PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}" \
MPLCONFIGDIR="${MPLCONFIGDIR:-logs/MVP/matplotlib_cache}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" evaluate_mvp.py \
  --data_path data/MVP_Benchmark/Completion/MVP_Test_CP.h5 \
  --model_path "${MODEL_PATH}" \
  --config_path configs/pa_rangelm_stage2.yaml \
  --output_csv "${CSV_PATH}" \
  --category_scope unseen8 \
  --batch_size 4 \
  --num_workers 0 \
  --num_keypoint 8 \
  --keypoint curvature_radius \
  --curve_k 16 \
  --curve_radius 0.075 \
  --views all \
  --random_rotate \
  --rotation_seed 2026 \
  --rotation_mode independent \
  --solver torch_lm \
  --torch_lm_iterations 80 \
  --torch_lm_damping 0.001 \
  --torch_lm_dtype float32 \
  --torch_lm_target_converged 0.999 \
  2>&1 | tee "${LOG_PATH}"
