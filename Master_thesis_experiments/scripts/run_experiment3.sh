#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/alien/sam3/Master_thesis_experiments
DATASET="$ROOT/datasets/raw/RefDrone"
UNITS="$ROOT/annotations/experiment3/category_prompt_units.json"
SPECIFICITY="$ROOT/annotations/experiment3/specificity_50_manifest.json"
OUTPUT="$ROOT/predictions/experiment3"

mkdir -p "$OUTPUT"

python "$ROOT/scripts/infer_sam3_refdrone.py" \
  --dataset-root "$DATASET" \
  --category-units "$UNITS" \
  --specificity-manifest "$SPECIFICITY" \
  --checkpoint /home/alien/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt \
  --model-name sam3_original \
  --output "$OUTPUT/sam3_original.jsonl" \
  --device cuda

python "$ROOT/scripts/infer_sam3_refdrone.py" \
  --dataset-root "$DATASET" \
  --category-units "$UNITS" \
  --specificity-manifest "$SPECIFICITY" \
  --checkpoint /home/alien/sam3/finetune_sam3/checkpoint_25_merged.pt \
  --model-name sam3_adapted_with_replay \
  --output "$OUTPUT/sam3_adapted_with_replay.jsonl" \
  --device cuda

