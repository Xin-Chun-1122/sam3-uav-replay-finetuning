#!/usr/bin/env python3
"""Create thesis-ready Experiment 2 tables from event and metric JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABELS = {
    "partial_occlusion": "Partial occlusion",
    "temporary_complete_disappearance": "Temporary complete disappearance",
    "scale_decrease_and_recovery": "Scale decrease and recovery",
    "camera_motion_or_blur": "Camera motion / blur",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-summary", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(args.event_summary.read_text(encoding="utf-8"))["conditions"]
    event_rows = []
    for condition, label in LABELS.items():
        item = summary[condition]
        event_rows.append({
            "Condition": label,
            "Events": item["events"],
            "Sequences": item["sequences"],
            "Median unique targets/sequence": item["median_unique_targets_per_sequence"],
            "Median duration (frames)": item["median_duration_frames"],
            "Source": "Natural VisDrone-MOT",
        })
    event_csv = args.output_dir / "experiment2_event_statistics.csv"
    with event_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(event_rows[0]))
        writer.writeheader(); writer.writerows(event_rows)
    lines = [
        "| Condition | Events | Sequences | Median unique targets/sequence | Median duration (frames) | Source |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in event_rows:
        lines.append("| {Condition} | {Events} | {Sequences} | {Median unique targets/sequence} | {Median duration (frames)} | {Source} |".format(**row))
    (args.output_dir / "experiment2_event_statistics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not args.results:
        return
    metric_rows = []
    for path in args.results:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["missing_events"]:
            raise RuntimeError(f"Incomplete result {path}: {len(result['missing_events'])} missing events")
        model = result["prediction_metadata"]["model_name"]
        for condition, label in [("overall", "All events"), *LABELS.items()]:
            metrics = result["overall"] if condition == "overall" else result["by_condition"][condition]
            metric_rows.append({
                "Model": model,
                "Condition": label if condition == "overall" else LABELS[condition],
                "Frame Recall": metrics["frame_recall"],
                "Recovery Rate": metrics["recovery_rate"],
                "AP50": metrics["track_ap50"],
            })
    with (args.output_dir / "experiment2_tracking_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader(); writer.writerows(metric_rows)
    lines = [
        "| Model | Condition | Frame Recall | Recovery Rate | AP50 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(f"| {row['Model']} | {row['Condition']} | {row['Frame Recall']:.4f} | {row['Recovery Rate']:.4f} | {row['AP50']:.4f} |")
    (args.output_dir / "experiment2_tracking_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
