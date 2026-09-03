#!/bin/bash
# VisDrone2019-DET full evaluation script
set -euo pipefail

CONDA_ENV="sam3_env"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_CKPT="/home/alien/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
FT_CKPT="/home/alien/sam3/finetune_sam3/final_best/sam3_frozen_best.pt"
VISDRONE_ROOT="/home/alien/Downloads/VisDrone2019-DET-test-dev"
OUTPUT_DIR="${PROJECT_ROOT}/evaluation_results/visdrone"
CONFIDENCE=0.30
IOU_THRESH=0.50
NMS_THRESH=0.50

echo "=========================================="
echo "  VisDrone Evaluation"
echo "  $(date)"
echo "=========================================="

cd "${PROJECT_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home/alien/miniconda3/envs/${CONDA_ENV}/bin/python evaluation/evaluate.py \
  --dataset visdrone \
  --visdrone-root "${VISDRONE_ROOT}" \
  --base-checkpoint "${BASE_CKPT}" \
  --finetuned-checkpoint "${FT_CKPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --confidence-threshold "${CONFIDENCE}" \
  --iou-threshold "${IOU_THRESH}" \
  --nms-threshold "${NMS_THRESH}" \
  --model both \
  --resume \
  "$@"

echo "Done. Results in: ${OUTPUT_DIR}"
