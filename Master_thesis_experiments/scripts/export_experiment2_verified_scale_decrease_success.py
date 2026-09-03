#!/usr/bin/env python3
"""Export a verified scale-decrease case without inventing detections."""

from pathlib import Path
import json
import math
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Master_thesis_experiments.scripts import export_experiment2_ppt_assets as assets


EVENT_ID = "scale_decrease_and_recovery-049"
FRAMES = {"before": 56, "during": 84, "after": 85}
PROMPT = "car"
CROP_SIZE = (120, 90)
OUTPUT = (
    assets.ROOT
    / "figures/experiment2/ppt_local_tracking_cases"
    / "scale_decrease_verified_success"
)


def crop_around(box, image_shape):
    width, height = CROP_SIZE
    image_height, image_width = image_shape[:2]
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    x1 = max(0, min(image_width - width, int(round(cx - width / 2))))
    y1 = max(0, min(image_height - height, int(round(cy - height / 2))))
    return (x1, y1, x1 + width, y1 + height)


def match_count(gt_rows, predictions):
    used = set()
    matched = 0
    for gt in gt_rows:
        candidates = [
            (assets.iou(gt["bbox_xyxy"], pred["bbox_xyxy"]), index)
            for index, pred in enumerate(predictions)
            if index not in used and pred.get("category") == PROMPT
        ]
        overlap, index = max(candidates, default=(0.0, None))
        if overlap >= 0.5:
            matched += 1
            used.add(index)
    return matched


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    event = next(
        row for row in assets.load_json(assets.EVENTS_PATH)
        if row["event_id"] == EVENT_ID
    )
    sequence = event["sequence_id"]
    target_id = int(event["target_id"])
    gt = assets.read_all_gt(sequence)
    _, selected = assets.result_index()
    models = list(assets.MODELS)[1:]
    predictions = {
        model: assets.read_predictions(model, EVENT_ID) for model in models
    }

    cells = {}
    audit = {"event_id": EVENT_ID, "frames": {}, "iou_threshold": 0.5}
    for phase, frame in FRAMES.items():
        image = cv2.imread(str(assets.image_path(sequence, frame)))
        target = next(row for row in gt[frame] if row["target_id"] == target_id)
        crop = crop_around(target["bbox_xyxy"], image.shape)
        visible_gt = [
            row for row in gt[frame]
            if row["category"] == PROMPT
            and assets.intersects(row["bbox_xyxy"], crop)
        ]
        box = target["bbox_xyxy"]
        scale = math.sqrt((box[2] - box[0]) * (box[3] - box[1]))

        gt_cell = assets.render_cell(
            image, gt[frame], PROMPT, target_id, crop,
            selected_only=False, gt_categories={PROMPT},
        )
        cells[("ground_truth", phase)] = gt_cell
        cv2.imwrite(str(OUTPUT / f"{phase}_ground_truth.png"), gt_cell)

        model_counts = {}
        for model in models:
            frame_predictions = predictions[model].get(frame, [])
            count = match_count(visible_gt, frame_predictions)
            model_counts[model] = {
                "matched_car_gt": count,
                "visible_car_gt": len(visible_gt),
            }
            cell = assets.render_cell(
                image, gt[frame], PROMPT, target_id, crop,
                frame_predictions, selected[model][EVENT_ID],
                label_all_predictions=True,
                selected_only=False,
                gt_categories={PROMPT},
            )
            cells[(model, phase)] = cell
            cv2.imwrite(str(OUTPUT / f"{phase}_{model}.png"), cell)

        audit["frames"][phase] = {
            "frame": frame,
            "target_scale_px": round(scale, 4),
            "crop_xyxy": list(crop),
            "visible_car_gt": len(visible_gt),
            "models": model_counts,
        }

    comparison = np.vstack([
        np.hstack([cells[(model, phase)] for phase in FRAMES])
        for model in assets.MODELS
    ])
    cv2.imwrite(str(OUTPUT / "comparison_verified_4x3.png"), comparison)
    focused_comparison = np.vstack([
        np.hstack([cells[(model, phase)] for phase in FRAMES])
        for model in ("ground_truth", "sam3_original", "sam3_adapted_with_replay")
    ])
    cv2.imwrite(
        str(OUTPUT / "comparison_groundtruth_original_ours_3x3.png"),
        focused_comparison,
    )
    (OUTPUT / "verification.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ours = [
        row["models"]["sam3_adapted_with_replay"]
        for row in audit["frames"].values()
    ]
    assert all(x["matched_car_gt"] == x["visible_car_gt"] for x in ours)
    assert (
        audit["frames"]["during"]["models"]["sam3_original"]["matched_car_gt"]
        < audit["frames"]["during"]["visible_car_gt"]
    )
    print(OUTPUT)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
