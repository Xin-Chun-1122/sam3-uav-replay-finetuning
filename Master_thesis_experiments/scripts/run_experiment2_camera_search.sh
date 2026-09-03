#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/alien/sam3
EXP="$ROOT/Master_thesis_experiments"
DATA="$EXP/datasets/raw/VisDrone2019-MOT-test-dev"
EVENTS="$EXP/annotations/experiment2_camera_search/events_priority24.json"
ORIGINAL=/home/alien/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt
ADAPTED="$ROOT/finetune_sam3/checkpoint_25_merged.pt"
PRED="$EXP/predictions/experiment2_camera_search"
RESULT="$EXP/results/experiment2_camera_search"

cd "$ROOT"
export LOG_LEVEL=ERROR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python "$EXP/scripts/infer_sam3_mot_events.py" \
  --dataset-root "$DATA" --events "$EVENTS" --checkpoint "$ORIGINAL" \
  --model-name sam3_original --output-dir "$PRED/sam3_original" \
  --init-mode text --detection-threshold 0.48943960666656494 --max-objects 10

python "$EXP/scripts/infer_sam3_mot_events.py" \
  --dataset-root "$DATA" --events "$EVENTS" --checkpoint "$ORIGINAL" \
  --model-name sam3_native_id_tracker --output-dir "$PRED/sam3_native_id_tracker" \
  --init-mode native_id_refine --detection-threshold 0.48943960666656494 --max-objects 10

python "$EXP/scripts/infer_sam3_mot_events.py" \
  --dataset-root "$DATA" --events "$EVENTS" --checkpoint "$ADAPTED" \
  --model-name sam3_adapted_with_replay --output-dir "$PRED/sam3_adapted_with_replay" \
  --init-mode text --detection-threshold 0.4773983955383301 --max-objects 10

for model in sam3_original sam3_native_id_tracker sam3_adapted_with_replay; do
  python "$EXP/scripts/evaluate_mot_events.py" \
    --dataset-root "$DATA" --events "$EVENTS" \
    --predictions-dir "$PRED/$model" \
    --output "$RESULT/${model}_priority24.json"
done
