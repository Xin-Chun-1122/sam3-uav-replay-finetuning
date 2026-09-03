#!/usr/bin/env bash
set -u

EXP=/home/alien/sam3/Master_thesis_experiments
TOTAL=160

echo "Experiment 2 status — $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo

for model in sam3_original sam3_native_id_tracker sam3_adapted_with_replay; do
  directory="$EXP/predictions/experiment2/$model"
  completed=0
  partial=0
  latest="-"
  if [ -d "$directory" ]; then
    completed=$(find "$directory" -maxdepth 1 -type f -name '*.jsonl' | wc -l)
    partial=$(find "$directory" -maxdepth 1 -type f -name '*.partial' | wc -l)
    latest_path=$(find "$directory" -maxdepth 1 -type f -name '*.jsonl' -printf '%T@ %f\n' | sort -nr | head -1 | cut -d' ' -f2-)
    if [ -n "$latest_path" ]; then latest="$latest_path"; fi
  fi
  percent=$(awk -v value="$completed" -v total="$TOTAL" 'BEGIN { printf "%.1f", value*100/total }')
  printf '%-32s %3d/%d (%5s%%)  partial=%d  latest=%s\n' "$model" "$completed" "$TOTAL" "$percent" "$partial" "$latest"
done

echo
if pgrep -f 'infer_sam3_mot_events.py' >/dev/null; then
  echo "Inference process: RUNNING"
  pgrep -af 'infer_sam3_mot_events.py' | sed -n '1p'
else
  echo "Inference process: NOT RUNNING"
fi

if pgrep -f 'run_experiment2.sh' >/dev/null; then
  echo "Automatic continuation: ACTIVE"
else
  echo "Automatic continuation: NOT ACTIVE"
fi

echo
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true

echo
if [ -f "$EXP/tables/experiment2/experiment2_tracking_results.md" ]; then
  echo "Final table: READY"
  echo "$EXP/tables/experiment2/experiment2_tracking_results.md"
else
  echo "Final table: not ready yet"
fi
