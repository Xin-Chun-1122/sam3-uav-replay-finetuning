#!/usr/bin/env python3
"""Select a SAM3 replay-anchor fusion rule on validation predictions only.

The adapted prediction is the primary output. Predictions agreeing with the
original SAM3 anchor are box-averaged and have their scores interpolated;
unmatched predictions are retained with a validation-selected score penalty.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Master_thesis_experiments.scripts.evaluate_visdrone_predictions import (  # noqa: E402
    CATEGORIES,
    SCALES,
    best_f1_threshold,
    counts_at_threshold,
    drop_predictions_in_ignored_regions,
    interpolated_ap,
    iou,
    label_predictions,
    limit_detections_per_image,
    load_gt,
    load_predictions,
)


def fuse_one(original: list[dict], adapted: list[dict], match_iou: float,
             adapted_weight: float, unmatched_penalty: float) -> list[dict]:
    pairs: list[tuple[float, int, int]] = []
    for ai, ap in enumerate(adapted):
        for oi, op in enumerate(original):
            overlap = iou(ap["bbox_xyxy"], op["bbox_xyxy"])
            if overlap >= match_iou:
                pairs.append((overlap, ai, oi))
    matched_a: set[int] = set()
    matched_o: set[int] = set()
    output: list[dict] = []
    for _, ai, oi in sorted(pairs, reverse=True):
        if ai in matched_a or oi in matched_o:
            continue
        matched_a.add(ai)
        matched_o.add(oi)
        ap, op = adapted[ai], original[oi]
        aw = adapted_weight
        output.append({
            "bbox_xyxy": [aw * a + (1.0 - aw) * o for a, o in zip(ap["bbox_xyxy"], op["bbox_xyxy"])],
            "score": aw * float(ap["score"]) + (1.0 - aw) * float(op["score"]),
        })
    for ai, ap in enumerate(adapted):
        if ai not in matched_a:
            output.append({"bbox_xyxy": ap["bbox_xyxy"], "score": float(ap["score"]) * unmatched_penalty})
    for oi, op in enumerate(original):
        if oi not in matched_o:
            output.append({"bbox_xyxy": op["bbox_xyxy"], "score": float(op["score"]) * unmatched_penalty})
    return sorted(output, key=lambda item: item["score"], reverse=True)


def fuse_all(original: dict, adapted: dict, match_iou: float,
             adapted_weight: float, unmatched_penalty: float) -> dict:
    keys = set(original) | set(adapted)
    return {
        key: fuse_one(original.get(key, []), adapted.get(key, []), match_iou,
                      adapted_weight, unmatched_penalty)
        for key in keys
    }


def evaluate(predictions: dict, gt: dict, ignore_integrals: dict, max_dets: int) -> dict:
    predictions = limit_detections_per_image(predictions, max_dets)
    predictions = drop_predictions_in_ignored_regions(predictions, ignore_integrals)
    labelled, gt_count = label_predictions(predictions, gt)
    threshold, overall = best_f1_threshold(labelled, gt_count)
    scale_metrics = {}
    for scale in SCALES:
        scale_labelled, scale_gt = label_predictions(predictions, gt, scale)
        scale_metrics[scale] = counts_at_threshold(scale_labelled, scale_gt, threshold)
    class_ap = {}
    for category in CATEGORIES:
        cp = {key: value for key, value in predictions.items() if key[1] == category}
        cg = {key: value for key, value in gt.items() if key[1] == category}
        class_labelled, class_gt = label_predictions(cp, cg)
        class_ap[category] = interpolated_ap(class_labelled, class_gt)
    image_count = len({key[0] for key in predictions})
    return {
        "threshold": threshold,
        "f1": overall["f1"],
        "recall": overall["recall"],
        "macro_ap50": sum(class_ap.values()) / len(CATEGORIES),
        "micro_ap50": interpolated_ap(labelled, gt_count),
        "fp_per_image": overall["fp"] / image_count,
        "scale_metrics": scale_metrics,
    }


def write_predictions(path: Path, predictions: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "metadata", **metadata}, ensure_ascii=False) + "\n")
        for image_name, category in sorted(predictions):
            handle.write(json.dumps({
                "type": "prediction",
                "image_name": image_name,
                "category": category,
                "prompt": category,
                "predictions": predictions[(image_name, category)],
            }, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--best-predictions", type=Path, required=True)
    parser.add_argument("--max-dets", type=int, default=500)
    args = parser.parse_args()

    original_meta, original = load_predictions(args.original)
    adapted_meta, adapted = load_predictions(args.adapted)
    image_names = {key[0] for key in original} | {key[0] for key in adapted}
    gt, ignore_integrals = load_gt(args.dataset_root.resolve(), image_names)

    trials = []
    best = None
    best_predictions = None
    for match_iou, adapted_weight, unmatched_penalty in itertools.product(
        (0.45, 0.55, 0.65), (0.4, 0.6, 0.8), (0.25, 0.5, 0.75)
    ):
        predictions = fuse_all(original, adapted, match_iou, adapted_weight, unmatched_penalty)
        metrics = evaluate(predictions, gt, ignore_integrals, args.max_dets)
        trial = {
            "match_iou": match_iou,
            "adapted_weight": adapted_weight,
            "unmatched_penalty": unmatched_penalty,
            **metrics,
        }
        trials.append(trial)
        # AP50 is primary; F1 and lower FP/image break practically equivalent ties.
        rank = (round(metrics["macro_ap50"], 4), round(metrics["micro_ap50"], 4),
                round(metrics["f1"], 4), -round(metrics["fp_per_image"], 3))
        if best is None or rank > best[0]:
            best = (rank, trial)
            best_predictions = predictions
        print(json.dumps(trial, ensure_ascii=False), flush=True)

    assert best is not None and best_predictions is not None
    summary = {
        "selection_split": str(args.dataset_root.resolve()),
        "selection_policy": "macro AP50 primary; micro AP50, F1, and lower FP/image tie-breakers",
        "original_predictions": str(args.original.resolve()),
        "adapted_predictions": str(args.adapted.resolve()),
        "best": best[1],
        "trials": trials,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_predictions(args.best_predictions, best_predictions, {
        "method": "validation-selected replay-anchor prediction fusion",
        "selection_split": str(args.dataset_root.resolve()),
        "parameters": {key: best[1][key] for key in ("match_iou", "adapted_weight", "unmatched_penalty")},
        "source_metadata": {"original": original_meta, "adapted": adapted_meta},
        "image_count": len(image_names),
        "categories": list(CATEGORIES),
    })
    print(json.dumps({"best": best[1]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
