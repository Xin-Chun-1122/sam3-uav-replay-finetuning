#!/usr/bin/env python3
"""Evaluate event-target tracking with fixed SAM 3 native object IDs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def iou(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def interpolated_ap(labelled: list[tuple[float, int]], gt_count: int) -> float:
    if gt_count == 0:
        return 0.0
    tp = fp = 0
    points = []
    for score, label in sorted(labelled, reverse=True):
        tp += label
        fp += 1 - label
        points.append((tp / gt_count, tp / (tp + fp)))
    return sum(max((p for r, p in points if r >= level), default=0.0)
               for level in (index / 100 for index in range(101))) / 101


def read_gt(path: Path) -> dict[int, dict[int, dict]]:
    result: dict[int, dict[int, dict]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 10:
            continue
        frame, target_id = int(parts[0]), int(parts[1])
        left, top, width, height = map(float, parts[2:6])
        result[target_id][frame] = {
            "bbox": [left, top, left + width, top + height],
            "scale": (width * height) ** 0.5,
            "score": int(parts[6]),
            "category_id": int(parts[7]),
            "truncation": int(parts[8]),
            "occlusion": int(parts[9]),
        }
    return result


def read_prediction_file(path: Path) -> dict[int, list[dict]]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("type") != "frame":
                continue
            result[int(record["frame_index"])] = record.get("predictions", [])
    return result


def select_native_id(event: dict, predictions: dict[int, list[dict]], target_gt: dict[int, dict], init_mode: str) -> int | None:
    evidence: dict[int, list[float]] = defaultdict(list)
    for frame in range(event["clip_start_frame"], event["start_frame"]):
        gt = target_gt.get(frame)
        if gt is None:
            continue
        for pred in predictions.get(frame, []):
            evidence[int(pred["sam3_obj_id"])].append(iou(pred["bbox_xyxy"], gt["bbox"]))
    if not evidence:
        return None
    return max(evidence, key=lambda obj_id: (
        sum(overlap >= 0.5 for overlap in evidence[obj_id]),
        sum(evidence[obj_id]), len(evidence[obj_id]), -obj_id,
    ))


def evaluate_event(event: dict, predictions: dict[int, list[dict]], target_gt: dict[int, dict],
                   init_mode: str, iou_threshold: float, recovery_window: int) -> dict:
    selected_id = select_native_id(event, predictions, target_gt, init_mode)
    visible_frames = [frame for frame in range(event["clip_start_frame"], event["clip_end_frame"] + 1)
                      if frame in target_gt and target_gt[frame]["score"] != 0 and target_gt[frame]["truncation"] <= 1]
    matches = 0
    labelled = []
    phase_counts = {"pre": [0, 0], "low_reliability": [0, 0], "post": [0, 0]}
    for frame in visible_frames:
        if frame < event["start_frame"]:
            phase = "pre"
        elif frame <= event["end_frame"]:
            phase = "low_reliability"
        else:
            phase = "post"
        phase_counts[phase][1] += 1
        candidates = [pred for pred in predictions.get(frame, [])
                      if selected_id is not None and int(pred["sam3_obj_id"]) == selected_id]
        if candidates:
            pred = max(candidates, key=lambda item: item["score"])
            matched = iou(pred["bbox_xyxy"], target_gt[frame]["bbox"]) >= iou_threshold
            labelled.append((float(pred["score"]), int(matched)))
            if matched:
                matches += 1
                phase_counts[phase][0] += 1

    post_visible = [frame for frame in visible_frames if frame > event["end_frame"]][:recovery_window]
    same_id_recovered = any(
        any(selected_id is not None and int(pred["sam3_obj_id"]) == selected_id
            and iou(pred["bbox_xyxy"], target_gt[frame]["bbox"]) >= iou_threshold
            for pred in predictions.get(frame, []))
        for frame in post_visible
    )
    any_id_recovered = any(
        any(iou(pred["bbox_xyxy"], target_gt[frame]["bbox"]) >= iou_threshold
            for pred in predictions.get(frame, []))
        for frame in post_visible
    )
    return {
        "event_id": event["event_id"],
        "condition": event["condition"],
        "selected_sam3_obj_id": selected_id,
        "visible_gt_frames": len(visible_frames),
        "matched_frames": matches,
        "frame_recall": matches / len(visible_frames) if visible_frames else 0.0,
        "recovery_eligible": bool(post_visible),
        "same_id_recovered": same_id_recovered,
        "any_id_recovered": any_id_recovered,
        "labelled": labelled,
        "phase_counts": phase_counts,
    }


def aggregate(items: list[dict]) -> dict:
    visible = sum(item["visible_gt_frames"] for item in items)
    matches = sum(item["matched_frames"] for item in items)
    eligible = [item for item in items if item["recovery_eligible"]]
    labelled = [pair for item in items for pair in item["labelled"]]
    phases = {}
    for phase in ("pre", "low_reliability", "post"):
        phase_matches = sum(item["phase_counts"][phase][0] for item in items)
        phase_gt = sum(item["phase_counts"][phase][1] for item in items)
        phases[phase] = {"matches": phase_matches, "gt_frames": phase_gt,
                         "frame_recall": phase_matches / phase_gt if phase_gt else None}
    return {
        "events": len(items),
        "frame_recall": matches / visible if visible else 0.0,
        "recovery_rate": sum(item["same_id_recovered"] for item in eligible) / len(eligible) if eligible else 0.0,
        "redetection_rate_any_id": sum(item["any_id_recovered"] for item in eligible) / len(eligible) if eligible else 0.0,
        "track_ap50": interpolated_ap(labelled, visible),
        "visible_gt_frames": visible,
        "matched_frames": matches,
        "recovery_eligible_events": len(eligible),
        "phase_frame_recall": phases,
        "median_event_frame_recall": statistics.median([item["frame_recall"] for item in items]) if items else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--recovery-window", type=int, default=10)
    args = parser.parse_args()

    events = json.loads(args.events.read_text(encoding="utf-8"))
    metadata = json.loads((args.predictions_dir / "metadata.json").read_text(encoding="utf-8"))
    gt_cache = {}
    event_results = []
    missing = []
    for event in events:
        prediction_path = args.predictions_dir / f"{event['event_id']}.jsonl"
        if not prediction_path.is_file():
            missing.append(event["event_id"])
            continue
        sequence = event["sequence_id"]
        if sequence not in gt_cache:
            gt_cache[sequence] = read_gt(args.dataset_root / "annotations" / f"{sequence}.txt")
        target_gt = gt_cache[sequence][int(event["target_id"])]
        predictions = read_prediction_file(prediction_path)
        event_results.append(evaluate_event(
            event, predictions, target_gt, metadata["init_mode"], args.iou_threshold, args.recovery_window
        ))
    by_condition = {}
    for condition in sorted({event["condition"] for event in events}):
        by_condition[condition] = aggregate([item for item in event_results if item["condition"] == condition])
    result = {
        "protocol": {
            "track_id_source": "sam3_native",
            "target_id_selection": "fixed native SAM3 obj_id selected on pre-event frames; ID-track baseline uses a GT-box visual initialization",
            "frame_recall": "same native obj_id IoU>=0.5 matches / visible target GT frames",
            "recovery_rate": f"same native obj_id matches within first {args.recovery_window} visible post-event frames",
            "track_ap50": "101-point AP50 using the fixed native track prediction against the event target",
        },
        "prediction_metadata": metadata,
        "events_expected": len(events),
        "events_evaluated": len(event_results),
        "missing_events": missing,
        "overall": aggregate(event_results),
        "by_condition": by_condition,
        "event_results": event_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": result["overall"], "by_condition": by_condition,
                      "missing_events": len(missing)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
