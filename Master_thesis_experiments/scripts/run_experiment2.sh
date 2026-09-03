#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/alien/sam3
EXP="$ROOT/Master_thesis_experiments"
DATA="$EXP/datasets/raw/VisDrone2019-MOT-test-dev"
EVENTS="$EXP/annotations/experiment2/events.json"
ORIGINAL=/home/alien/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt
ADAPTED="$ROOT/finetune_sam3/checkpoint_25_merged.pt"

cd "$ROOT"
export LOG_LEVEL=ERROR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python "$EXP/scripts/infer_sam3_mot_events.py" \
  --dataset-root "$DATA" --events "$EVENTS" --checkpoint "$ORIGINAL" \
  --model-name sam3_original --output-dir "$EXP/predictions/experiment2/sam3_original" \
  --init-mode text --detection-threshold 0.48943960666656494 --max-objects 10

python "$EXP/scripts/infer_sam3_mot_events.py" \
  --dataset-root "$DATA" --events "$EVENTS" --checkpoint "$ORIGINAL" \
  --model-name sam3_native_id_tracker --output-dir "$EXP/predictions/experiment2/sam3_native_id_tracker" \
  --init-mode native_id_refine --detection-threshold 0.48943960666656494 --max-objects 10

python "$EXP/scripts/infer_sam3_mot_events.py" \
  --dataset-root "$DATA" --events "$EVENTS" --checkpoint "$ADAPTED" \
  --model-name sam3_adapted_with_replay --output-dir "$EXP/predictions/experiment2/sam3_adapted_with_replay" \
  --init-mode text --detection-threshold 0.4773983955383301 --max-objects 10

for model in sam3_original sam3_native_id_tracker sam3_adapted_with_replay; do
  python "$EXP/scripts/evaluate_mot_events.py" \
    --dataset-root "$DATA" --events "$EVENTS" \
    --predictions-dir "$EXP/predictions/experiment2/$model" \
    --output "$EXP/results/experiment2/${model}.json"
done

python "$EXP/scripts/make_experiment2_tables.py" \
  --event-summary "$EXP/results/experiment2/event_mining_summary.json" \
  --results "$EXP/results/experiment2/sam3_original.json" \
            "$EXP/results/experiment2/sam3_native_id_tracker.json" \
            "$EXP/results/experiment2/sam3_adapted_with_replay.json" \
  --output-dir "$EXP/tables/experiment2"

