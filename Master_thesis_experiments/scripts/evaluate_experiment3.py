#!/usr/bin/env python3
"""Evaluate RefDrone standard, synonym, and three-layer prompt predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


CATEGORIES = ("person", "car", "truck", "bus", "motorcycle")


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb > inter else 0.0


def operating_counts(
    units: list[dict],
    predictions: dict[str, list[dict]],
    threshold: float,
    iou_threshold: float,
) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for unit in units:
        gt = unit["gt_boxes_xyxy"]
        matched: set[int] = set()
        preds = sorted(
            (row for row in predictions.get(unit["unit_id"], []) if row["score"] >= threshold),
            key=lambda row: row["score"],
            reverse=True,
        )
        for pred in preds:
            choices = [
                (iou(pred["bbox_xyxy"], box), index)
                for index, box in enumerate(gt)
                if index not in matched
            ]
            best_iou, best_index = max(choices, default=(0.0, -1))
            if best_iou >= iou_threshold:
                tp += 1
                matched.add(best_index)
            else:
                fp += 1
        fn += len(gt) - len(matched)
    return tp, fp, fn


def ap50(
    units: list[dict],
    predictions: dict[str, list[dict]],
    iou_threshold: float,
) -> float:
    total_gt = sum(len(unit["gt_boxes_xyxy"]) for unit in units)
    if total_gt == 0:
        return 0.0
    gt_by_unit = {unit["unit_id"]: unit["gt_boxes_xyxy"] for unit in units}
    ranked = sorted(
        (
            (pred["score"], unit_id, pred["bbox_xyxy"])
            for unit_id in gt_by_unit
            for pred in predictions.get(unit_id, [])
        ),
        reverse=True,
    )
    matched: dict[str, set[int]] = defaultdict(set)
    tp_values: list[int] = []
    fp_values: list[int] = []
    for _, unit_id, box in ranked:
        choices = [
            (iou(box, gt_box), index)
            for index, gt_box in enumerate(gt_by_unit[unit_id])
            if index not in matched[unit_id]
        ]
        best_iou, best_index = max(choices, default=(0.0, -1))
        hit = best_iou >= iou_threshold
        tp_values.append(int(hit))
        fp_values.append(int(not hit))
        if hit:
            matched[unit_id].add(best_index)
    cumulative_tp = cumulative_fp = 0
    recalls, precisions = [], []
    for hit, miss in zip(tp_values, fp_values):
        cumulative_tp += hit
        cumulative_fp += miss
        recalls.append(cumulative_tp / total_gt)
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))
    return sum(
        max((p for p, r in zip(precisions, recalls) if r >= level), default=0.0)
        for level in (index / 100 for index in range(101))
    ) / 101


def summarize(
    units: list[dict],
    predictions: dict[str, list[dict]],
    threshold: float,
    iou_threshold: float,
) -> dict:
    tp, fp, fn = operating_counts(units, predictions, threshold, iou_threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap50": ap50(units, predictions, iou_threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_gt_count": sum(len(unit["gt_boxes_xyxy"]) for unit in units) / len(units),
        "mean_predicted_count": (
            sum(sum(pred["score"] >= threshold for pred in predictions.get(unit["unit_id"], [])) for unit in units)
            / len(units)
        ),
        "unit_count": len(units),
    }


def load_standard_predictions(path: Path) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") != "prediction" or row.get("category") not in CATEGORIES:
                continue
            output[f"{row['image_name']}:{row['category']}"] = row["predictions"]
    return output


def load_experiment3_predictions(path: Path) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    synonyms: dict[str, list[dict]] = {}
    specificity: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") != "prediction":
                continue
            if row["evaluation"] == "synonym":
                synonyms[f"{row['image_name']}:{row['category']}"] = row["predictions"]
            elif row["evaluation"] == "specificity":
                specificity[f"{row['example_id']}:L{row['layer']}"] = row["predictions"]
    return synonyms, specificity


def mean_metrics(rows: list[dict]) -> dict:
    fields = ("precision", "recall", "f1", "ap50")
    return {field: sum(row[field] for row in rows) / len(rows) for field in fields}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category-units", type=Path, required=True)
    parser.add_argument("--specificity-manifest", type=Path, required=True)
    parser.add_argument("--standard-predictions", type=Path, required=True)
    parser.add_argument("--experiment3-predictions", type=Path, required=True)
    parser.add_argument("--specificity-predictions", type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    category_units = json.loads(args.category_units.read_text(encoding="utf-8"))
    specificity_manifest = json.loads(args.specificity_manifest.read_text(encoding="utf-8"))
    standard_predictions = load_standard_predictions(args.standard_predictions)
    synonym_predictions, specificity_predictions = load_experiment3_predictions(
        args.experiment3_predictions
    )
    if args.specificity_predictions is not None:
        _, specificity_predictions = load_experiment3_predictions(args.specificity_predictions)

    standard_rows, synonym_rows = [], []
    for category in CATEGORIES:
        units = [unit for unit in category_units if unit["category"] == category]
        standard = summarize(
            units, standard_predictions, args.threshold, args.iou_threshold
        )
        standard_rows.append({"category": category, "prompt": category, **standard})
        synonym = units[0]["synonym_prompt"]
        if synonym is not None:
            result = summarize(
                units, synonym_predictions, args.threshold, args.iou_threshold
            )
            result["retention"] = result["f1"] / standard["f1"] if standard["f1"] else None
            synonym_rows.append({
                "category": category,
                "prompt": synonym,
                **result,
            })

    specificity_units = []
    for example in specificity_manifest:
        for layer in example["layers"]:
            specificity_units.append({
                "unit_id": f"{example['example_id']}:L{layer['layer']}",
                "example_id": example["example_id"],
                "file_name": example["file_name"],
                "category": example["category"],
                "layer": layer["layer"],
                "prompt": layer["prompt"],
                "gt_boxes_xyxy": layer["gt_boxes_xyxy"],
            })
    specificity_rows = []
    specificity_by_category = []
    for layer in (1, 2, 3):
        units = [unit for unit in specificity_units if unit["layer"] == layer]
        specificity_rows.append({
            "layer": layer,
            **summarize(units, specificity_predictions, args.threshold, args.iou_threshold),
        })
        for category in CATEGORIES:
            subset = [unit for unit in units if unit["category"] == category]
            specificity_by_category.append({
                "layer": layer,
                "category": category,
                **summarize(subset, specificity_predictions, args.threshold, args.iou_threshold),
            })

    result = {
        "model": args.model_name,
        "protocol": {
            "iou_threshold": args.iou_threshold,
            "operating_threshold": args.threshold,
            "ap50": "101-point interpolated AP at IoU 0.50",
            "matching": "one-to-one greedy matching in descending confidence order",
            "standard_prediction_source": str(args.standard_predictions.resolve()),
            "experiment3_prediction_source": str(args.experiment3_predictions.resolve()),
            "specificity_prediction_source": str(
                (args.specificity_predictions or args.experiment3_predictions).resolve()
            ),
        },
        "standard": {
            "per_category": standard_rows,
            "macro": mean_metrics(standard_rows),
            "micro": summarize(
                category_units, standard_predictions, args.threshold, args.iou_threshold
            ),
        },
        "synonym": {
            "per_category": synonym_rows,
            "macro": {
                **mean_metrics(synonym_rows),
                "retention": sum(row["retention"] for row in synonym_rows) / len(synonym_rows),
            },
            "bus_policy": "excluded because no synonym was specified",
        },
        "specificity": {
            "overall_by_layer": specificity_rows,
            "per_category_by_layer": specificity_by_category,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = args.output.with_suffix(".specificity.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "layer", "precision", "recall", "f1", "ap50",
            "mean_gt_count", "mean_predicted_count", "tp", "fp", "fn", "unit_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in specificity_rows)
    print(f"output={args.output}")
    print(f"specificity_csv={csv_path}")


if __name__ == "__main__":
    main()
