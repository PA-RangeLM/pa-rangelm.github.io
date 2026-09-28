#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ROOT="train_res/pa_rangelm_kitti"
MODEL_PATH="${MODEL_PATH:-}"
CONFIG_PATH="configs/pa_rangelm_stage2.yaml"
OUTPUT_DIR="${RUN_ROOT}/evaluation/kitti_rot_y_s2026_weighted_torch_lm"
LOG_PATH="${RUN_ROOT}/evaluation/eval_kitti_rot_y_s2026_fd_mmd_canonical.log"

if [[ -z "${MODEL_PATH}" ]] || [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Stage-II val_best.pth not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Saved Stage-II config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

CAR_COUNT="$(find data/KITTI/cars -maxdepth 1 -type f -name '*.pcd' | wc -l)"
BBOX_COUNT="$(find data/KITTI/bboxes -maxdepth 1 -type f -name '*.txt' | wc -l)"
REFERENCE_COUNT="$(find data/PCN -path '*/complete/02958343/*.pcd' -type f | wc -l)"
if [[ "${CAR_COUNT}" -ne 2401 || "${BBOX_COUNT}" -ne 2401 ]]; then
  echo "Incomplete KITTI data: cars=${CAR_COUNT}/2401, bboxes=${BBOX_COUNT}/2401" >&2
  echo "Download and extract the Google Drive bboxes folder into data/KITTI/bboxes." >&2
  exit 2
fi
if [[ "${REFERENCE_COUNT}" -ne 5927 ]]; then
  echo "Unexpected PCN car reference count: ${REFERENCE_COUNT}/5927" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
# KITTI comparison requires both Fidelity/FD and the exact PCN-car MMD.
# Keep MMD enabled by default so one command produces the complete table input.
MMD_ARGS=(--compute_mmd --mmd_batch_size "${MMD_BATCH_SIZE:-16}")
INFERENCE_ARGS=()
PREDICTION_COUNT=0
if [[ -d "${OUTPUT_DIR}/predictions" ]]; then
  PREDICTION_COUNT="$(find "${OUTPUT_DIR}/predictions" -mindepth 1 -maxdepth 1 -type d | wc -l)"
fi
FIDELITY_COUNT=0
if [[ -f "${OUTPUT_DIR}/fidelity_per_sample.jsonl" ]]; then
  FIDELITY_COUNT="$(wc -l < "${OUTPUT_DIR}/fidelity_per_sample.jsonl")"
fi
if [[ "${PREDICTION_COUNT}" -eq 2401 && "${FIDELITY_COUNT}" -eq 2401 && "${FORCE_KITTI_INFERENCE:-0}" != "1" ]]; then
  # Keep already-written predictions fixed. MMD inverse-rotates them into the
  # canonical PCN reference frame and uses its own resumable progress file.
  INFERENCE_ARGS+=(--skip_inference)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python evaluate_kitti.py \
  --model_path "${MODEL_PATH}" \
  --config_path "${CONFIG_PATH}" \
  --kitti_root data/KITTI \
  --category_file data/KITTI/KITTI.json \
  --pcn_root data/PCN \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${KITTI_BATCH_SIZE:-4}" \
  --num_workers "${KITTI_NUM_WORKERS:-0}" \
  --seed 2026 \
  --rotation_mode single_axis \
  --rotation_axis y \
  --rotation_min_deg -180 \
  --rotation_max_deg 180 \
  --num_keypoints 8 \
  --curve_k 16 \
  --curve_radius 0.075 \
  --torch_lm_iterations 80 \
  --torch_lm_damping 0.001 \
  --torch_lm_dtype float32 \
  --torch_lm_target_converged 0.999 \
  --solver torch_lm \
  "${INFERENCE_ARGS[@]}" \
  "${MMD_ARGS[@]}" \
  2>&1 | tee -a "${LOG_PATH}"
