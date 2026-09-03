#!/usr/bin/env python3
"""Evaluate class-aware VisDrone detections at IoU 0.50, including GT scale bins."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


CATEGORY_MAP = {1: "person", 2: "person", 4: "car", 5: "van", 6: "truck", 9: "bus", 10: "motorcycle"}
CATEGORIES = ("car", "person", "bus", "van", "truck", "motorcycle")
SCALES = ("tiny", "small", "regular")


def scale_name(width: float, height: float) -> str:
    size = math.sqrt(width * height)
    return "tiny" if size < 32 else "small" if size < 64 else "regular"


def iou(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def ioa_detection(box: list[float], region: list[float]) -> float:
    left, top = max(box[0], region[0]), max(box[1], region[1])
    right, bottom = min(box[2], region[2]), min(box[3], region[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return intersection / area if area > 0 else 0.0


def load_predictions(path: Path) -> tuple[dict, dict[tuple[str, str], list[dict]]]:
    metadata = {}
    predictions: dict[tuple[str, str], list[dict]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["type"] == "metadata":
                metadata = record
            elif record["type"] == "prediction":
                predictions[(record["image_name"], record["category"])] = record["predictions"]
    return metadata, predictions


def load_gt(dataset_root: Path, image_names: set[str]) -> tuple[dict[tuple[str, str], list[dict]], dict[str, np.ndarray]]:
    result: dict[tuple[str, str], list[dict]] = defaultdict(list)
    ignore_integrals: dict[str, np.ndarray] = {}
    for image_name in sorted(image_names):
        ann_path = dataset_root / "annotations" / f"{Path(image_name).stem}.txt"
        rows = []
        with ann_path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                left, top, width, height = map(float, parts[:4])
                score, category_id = int(parts[4]), int(parts[5])
                if width <= 0 or height <= 0:
                    continue
                rows.append((left, top, width, height, score, category_id))
        with Image.open(dataset_root / "images" / image_name) as image:
            image_width, image_height = image.size
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        for left, top, width, height, _, category_id in rows:
            if category_id != 0:
                continue
            x1, y1 = max(0, round(left)), max(0, round(top))
            x2, y2 = min(image_width, round(left + width)), min(image_height, round(top + height))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1
        integral = np.pad(mask.cumsum(0, dtype=np.int64).cumsum(1, dtype=np.int64), ((1, 0), (1, 0)))
        ignore_integrals[image_name] = integral
        for left, top, width, height, score, category_id in rows:
            if category_id not in CATEGORY_MAP:
                continue
            box = [left, top, left + width, top + height]
            if ignored_region_fraction(box, integral) >= 0.5:
                continue
            result[(image_name, CATEGORY_MAP[category_id])].append({
                    "bbox": box,
                    "scale": scale_name(width, height),
                    "ignored": score == 0,
                })
    return result, ignore_integrals


def ignored_region_fraction(box: list[float], integral: np.ndarray) -> float:
    height, width = integral.shape[0] - 1, integral.shape[1] - 1
    x1, y1 = max(0, min(width, round(box[0]))), max(0, min(height, round(box[1])))
    x2, y2 = max(0, min(width, round(box[2]))), max(0, min(height, round(box[3])))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    covered = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    return float(covered) / ((x2 - x1) * (y2 - y1))


def drop_predictions_in_ignored_regions(predictions: dict, ignore_integrals: dict[str, np.ndarray]) -> dict:
    return {
        key: [pred for pred in values if ignored_region_fraction(pred["bbox_xyxy"], ignore_integrals[key[0]]) < 0.5]
        for key, values in predictions.items()
    }


def limit_detections_per_image(predictions: dict, max_dets: int) -> dict:
    by_image: dict[str, list[tuple[float, tuple[str, str], dict]]] = defaultdict(list)
    for key, values in predictions.items():
        for pred in values:
            by_image[key[0]].append((float(pred["score"]), key, pred))
    limited: dict[tuple[str, str], list[dict]] = {key: [] for key in predictions}
    for items in by_image.values():
        for _, key, pred in sorted(items, key=lambda item: item[0], reverse=True)[:max_dets]:
            limited[key].append(pred)
    return limited


def label_predictions(predictions: dict, gt: dict, target_scale: str | None = None) -> tuple[list[tuple[float, int]], int]:
    labelled: list[tuple[float, int]] = []
    gt_count = 0
    keys = set(predictions) | set(gt)
    for key in keys:
        targets = gt.get(key, [])
        evaluable = [
            index for index, item in enumerate(targets)
            if not item["ignored"] and (target_scale is None or item["scale"] == target_scale)
        ]
        dataset_ignored = [index for index, item in enumerate(targets) if item["ignored"]]
        scale_ignored = [
            index for index, item in enumerate(targets)
            if not item["ignored"] and target_scale is not None and item["scale"] != target_scale
        ]
        gt_count += len(evaluable)
        matched: set[int] = set()
        for pred in sorted(predictions.get(key, []), key=lambda item: item["score"], reverse=True):
            best_index, best_iou = None, 0.5
            for index in evaluable:
                overlap = iou(pred["bbox_xyxy"], targets[index]["bbox"])
                if index not in matched and overlap >= best_iou:
                    best_index, best_iou = index, overlap
            if best_index is not None:
                matched.add(best_index)
                labelled.append((float(pred["score"]), 1))
                continue
            if any(ioa_detection(pred["bbox_xyxy"], targets[index]["bbox"]) >= 0.5 for index in dataset_ignored):
                continue
            if any(iou(pred["bbox_xyxy"], targets[index]["bbox"]) >= 0.5 for index in scale_ignored):
                continue
            labelled.append((float(pred["score"]), 0))
    return labelled, gt_count


def counts_at_threshold(labelled: list[tuple[float, int]], gt_count: int, threshold: float) -> dict:
    kept = [label for score, label in labelled if score >= threshold]
    tp, fp = sum(kept), len(kept) - sum(kept)
    fn = gt_count - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / gt_count if gt_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def best_f1_threshold(labelled: list[tuple[float, int]], gt_count: int) -> tuple[float, dict]:
    ordered = sorted(labelled, reverse=True)
    tp = fp = 0
    best = (0.5, {"tp": 0, "fp": 0, "fn": gt_count, "precision": 0.0, "recall": 0.0, "f1": 0.0})
    index = 0
    while index < len(ordered):
        threshold = ordered[index][0]
        while index < len(ordered) and ordered[index][0] == threshold:
            if ordered[index][1]:
                tp += 1
            else:
                fp += 1
            index += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / gt_count if gt_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = {"tp": tp, "fp": fp, "fn": gt_count - tp, "precision": precision, "recall": recall, "f1": f1}
        if f1 > best[1]["f1"]:
            best = (threshold, candidate)
    return best


def interpolated_ap(labelled: list[tuple[float, int]], gt_count: int) -> float:
    if gt_count == 0:
        return 0.0
    tp = fp = 0
    points = []
    for _, label in sorted(labelled, reverse=True):
        tp += label
        fp += 1 - label
        points.append((tp / gt_count, tp / (tp + fp)))
    return sum(max((precision for recall, precision in points if recall >= level), default=0.0) for level in (i / 100 for i in range(101))) / 101


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, help="Use frozen threshold; otherwise maximize micro F1 on this dataset")
    parser.add_argument("--max-dets", type=int, default=500)
    args = parser.parse_args()

    metadata, predictions = load_predictions(args.predictions)
    image_names = {key[0] for key in predictions}
    predictions = limit_detections_per_image(predictions, args.max_dets)
    gt, ignore_integrals = load_gt(args.dataset_root.resolve(), image_names)
    predictions = drop_predictions_in_ignored_regions(predictions, ignore_integrals)
    all_labelled, all_gt_count = label_predictions(predictions, gt)
    if args.threshold is None:
        threshold, overall = best_f1_threshold(all_labelled, all_gt_count)
        threshold_source = "optimized_on_this_dataset"
    else:
        threshold = args.threshold
        overall = counts_at_threshold(all_labelled, all_gt_count, threshold)
        threshold_source = "frozen_external"

    scale_metrics = {}
    for scale in SCALES:
        labelled, gt_count = label_predictions(predictions, gt, scale)
        scale_metrics[scale] = counts_at_threshold(labelled, gt_count, threshold)

    per_class_ap50 = {}
    for category in CATEGORIES:
        category_pred = {key: value for key, value in predictions.items() if key[1] == category}
        category_gt = {key: value for key, value in gt.items() if key[1] == category}
        labelled, gt_count = label_predictions(category_pred, category_gt)
        per_class_ap50[category] = interpolated_ap(labelled, gt_count)
    macro_ap50 = sum(per_class_ap50.values()) / len(CATEGORIES)
    micro_ap50 = interpolated_ap(all_labelled, all_gt_count)
    result = {
        "protocol": "class-aware greedy matching, IoU >= 0.50; maxDets=500/image; VisDrone ignored-region filtering; 101-point interpolated AP50",
        "prediction_metadata": metadata,
        "evaluated_image_count": len(image_names),
        "operating_threshold": threshold,
        "threshold_source": threshold_source,
        "overall_operating_point": overall,
        "scale_metrics": scale_metrics,
        "per_class_ap50": per_class_ap50,
        "macro_ap50": macro_ap50,
        "micro_ap50": micro_ap50,
        "fp_per_image": overall["fp"] / len(image_names),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
