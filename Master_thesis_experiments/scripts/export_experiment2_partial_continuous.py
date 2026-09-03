#!/usr/bin/env python3
"""Export a five-frame continuous partial-occlusion sequence for PPT use."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from export_experiment2_ppt_assets import (
    MODELS,
    OUTPUT,
    PREDICTIONS,
    RESULTS,
    centered_phase_crop,
    image_path,
    intersects,
    iou,
    read_all_gt,
    read_predictions,
    render_cell,
    result_index,
    target_gt,
)


EVENT_ID = "partial_occlusion-018"
SEQUENCE = "uav0000188_00000_v"
TARGET_ID = 12
PROMPT = "car"
FRAMES = [213, 217, 220, 224, 229]
CONTINUOUS_INTERVAL = [213, 229]
CROP_WIDTH = 240
OUT = OUTPUT / "partial_occlusion" / "continuous_occlusion"


def text_bar(width: int, frame: int) -> np.ndarray:
    bar = np.full((90, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        bar,
        f"Frame {frame}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        bar,
        "Official occlusion = 1",
        (18, 69),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )
    return bar


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gt = read_all_gt(SEQUENCE)
    results, selected = result_index()
    model_predictions = {
        model: read_predictions(model, EVENT_ID) for model in list(MODELS)[1:]
    }

    cells: dict[tuple[str, int], np.ndarray] = {}
    audits = []
    for frame in FRAMES:
        selected_row = next(
            row for row in gt[frame] if row["target_id"] == TARGET_ID
        )
        assert selected_row["occlusion"] == 1
        crop = centered_phase_crop(
            SEQUENCE,
            target_gt(gt, frame, TARGET_ID),
            CROP_WIDTH,
        )
        image = cv2.imread(str(image_path(SEQUENCE, frame)))
        cells[("ground_truth", frame)] = render_cell(
            image, gt[frame], PROMPT, TARGET_ID, crop
        )
        cv2.imwrite(
            str(OUT / f"frame_{frame}_ground_truth.png"),
            cells[("ground_truth", frame)],
        )

        for model in list(MODELS)[1:]:
            cell = render_cell(
                image,
                gt[frame],
                PROMPT,
                TARGET_ID,
                crop,
                model_predictions[model][frame],
                selected[model][EVENT_ID],
                label_all_predictions=(model == "sam3_adapted_with_replay"),
            )
            cells[(model, frame)] = cell
            cv2.imwrite(str(OUT / f"frame_{frame}_{model}.png"), cell)

        visible_cars = [
            row
            for row in gt[frame]
            if row["category"] == PROMPT and intersects(row["bbox_xyxy"], crop)
        ]
        ours_predictions = model_predictions["sam3_adapted_with_replay"][frame]
        matches = []
        for row in visible_cars:
            best_iou = max(
                (
                    iou(row["bbox_xyxy"], prediction["bbox_xyxy"])
                    for prediction in ours_predictions
                ),
                default=0.0,
            )
            matches.append(
                {
                    "gt_target_id": row["target_id"],
                    "best_iou": best_iou,
                    "matched_at_iou_0_5": best_iou >= 0.5,
                }
            )
        assert matches and all(row["matched_at_iou_0_5"] for row in matches)
        audits.append(
            {
                "frame": frame,
                "official_target_occlusion": selected_row["occlusion"],
                "crop_xyxy": crop,
                "cars_in_crop": len(matches),
                "cars_matched_by_ours": sum(
                    row["matched_at_iou_0_5"] for row in matches
                ),
                "matches": matches,
            }
        )

    ours_panels = [
        np.vstack([text_bar(960, frame), cells[("sam3_adapted_with_replay", frame)]])
        for frame in FRAMES
    ]
    cv2.imwrite(
        str(OUT / "continuous_partial_occlusion_ours_5frames.png"),
        np.hstack(ours_panels),
    )

    row_images = []
    for model in MODELS:
        panels = [
            np.vstack([text_bar(960, frame), cells[(model, frame)]])
            for frame in FRAMES
        ]
        row = np.hstack(panels)
        row_images.append(row)
    cv2.imwrite(
        str(OUT / "continuous_partial_occlusion_comparison_4x5.png"),
        np.vstack(row_images),
    )

    metadata = {
        "event_id": EVENT_ID,
        "sequence_id": SEQUENCE,
        "selected_target_id": TARGET_ID,
        "prompt": PROMPT,
        "official_continuous_occlusion_interval": CONTINUOUS_INTERVAL,
        "displayed_frames": FRAMES,
        "ours_selected_native_sam3_id": selected["sam3_adapted_with_replay"][
            EVENT_ID
        ],
        "ours_event_result": results["sam3_adapted_with_replay"][EVENT_ID],
        "frame_audit": audits,
    }
    (OUT / "audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
