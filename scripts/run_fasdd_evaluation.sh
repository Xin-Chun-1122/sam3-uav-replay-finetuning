#!/bin/bash
# FASDD-UAV full evaluation script
set -euo pipefail

CONDA_ENV="sam3_env"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_CKPT="/home/alien/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
FT_CKPT="/home/alien/sam3/finetune_sam3/final_best/sam3_frozen_best.pt"
FASDD_JSON="/home/alien/Downloads/FASDD_UAV (1)/annotations/COCO_UAV/Annotations/test.json"
FASDD_IMAGES="/home/alien/Downloads/FASDD_UAV (1)/images"
OUTPUT_DIR="${PROJECT_ROOT}/evaluation_results/fasdd"
CONFIDENCE=0.30
IOU_THRESH=0.50
NMS_THRESH=0.50

echo "=========================================="
echo "  FASDD-UAV Evaluation"
echo "  $(date)"
echo "=========================================="
echo "Base checkpoint:     ${BASE_CKPT}"
echo "Fine-tuned ckpt:     ${FT_CKPT}"
echo "Output dir:          ${OUTPUT_DIR}"
echo "Confidence thresh:   ${CONFIDENCE}"
echo ""

cd "${PROJECT_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home/alien/miniconda3/envs/${CONDA_ENV}/bin/python evaluation/evaluate.py \
  --dataset fasdd \
  --fasdd-json "${FASDD_JSON}" \
  --fasdd-images "${FASDD_IMAGES}" \
  --base-checkpoint "${BASE_CKPT}" \
  --finetuned-checkpoint "${FT_CKPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --confidence-threshold "${CONFIDENCE}" \
  --iou-threshold "${IOU_THRESH}" \
  --nms-threshold "${NMS_THRESH}" \
  --model both \
  --resume \
  "$@"

echo ""
echo "Done. Results in: ${OUTPUT_DIR}"
