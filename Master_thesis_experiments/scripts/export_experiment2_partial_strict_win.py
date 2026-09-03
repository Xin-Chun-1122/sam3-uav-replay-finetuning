#!/usr/bin/env python3
"""Export a measured adapted-only win during continuous partial occlusion."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from export_experiment2_ppt_assets import (
    COLORS,
    DATA,
    OUTPUT,
    PREDICTIONS,
    RESULTS,
    image_path,
    iou,
)


EVENT_ID = "partial_occlusion-007"
SEQUENCE = "uav0000120_04775_v"
TARGET_ID = 219
PROMPT = "car"
FRAMES = [664, 665, 666, 667]
DISPLAY_FRAMES = [664, 665, 667]
OUT = OUTPUT / "partial_occlusion" / "strict_adapted_only"
MODELS = [
    ("sam3_original", "Original SAM 3"),
    ("sam3_native_id_tracker", "Native-ID Refinement"),
    ("sam3_adapted_with_replay", "Ours"),
]
CELL_SIZE = (600, 720)


def selected_ids() -> dict[str, int]:
    result = {}
    for model, _ in MODELS:
        rows = json.loads((RESULTS / f"{model}.json").read_text())["event_results"]
        row = next(item for item in rows if item["event_id"] == EVENT_ID)
        result[model] = int(row["selected_sam3_obj_id"])
    return result


def read_gt() -> dict[int, list[dict]]:
    output = {}
    path = DATA / "annotations" / f"{SEQUENCE}.txt"
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split(",")
        frame = int(fields[0])
        if frame not in FRAMES or int(fields[6]) != 1 or int(fields[7]) != 4:
            continue
        x, y, width, height = map(float, fields[2:6])
        output.setdefault(frame, []).append(
            {
                "target_id": int(fields[1]),
                "bbox": [x, y, x + width, y + height],
                "occlusion": int(fields[9]),
            }
        )
    return output


def read_predictions(model: str) -> dict[int, list[dict]]:
    output = {}
    path = PREDICTIONS / model / f"{EVENT_ID}.jsonl"
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["frame_index"] in FRAMES:
            output[int(row["frame_index"])] = row["predictions"]
    return output


def crop_for_target(target_box: list[float]) -> tuple[int, int, int, int]:
    cx = (target_box[0] + target_box[2]) / 2
    cy = (target_box[1] + target_box[3]) / 2
    # Narrow ROI excludes neighbouring unmatched cars while retaining two
    # genuinely matched cars (GT 218 and selected GT 219).
    return (
        int(round(cx - 70)),
        int(round(cy - 20)),
        int(round(cx + 30)),
        int(round(cy + 100)),
    )


def dashed_box(
    image: np.ndarray, box: list[float], color: tuple[int, int, int], thickness: int
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    dash = 16
    for start in range(x1, x2, dash * 2):
        cv2.line(image, (start, y1), (min(start + dash, x2), y1), color, thickness)
        cv2.line(image, (start, y2), (min(start + dash, x2), y2), color, thickness)
    for start in range(y1, y2, dash * 2):
        cv2.line(image, (x1, start), (x1, min(start + dash, y2)), color, thickness)
        cv2.line(image, (x2, start), (x2, min(start + dash, y2)), color, thickness)


def label(
    image: np.ndarray,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
) -> None:
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2
    )
    y = max(height + 5, y)
    cv2.rectangle(
        image,
        (x, y - height - 5),
        (x + width + 8, y + baseline + 3),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 4, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def transform_box(
    box: list[float], crop: tuple[int, int, int, int]
) -> list[float]:
    sx = CELL_SIZE[0] / (crop[2] - crop[0])
    sy = CELL_SIZE[1] / (crop[3] - crop[1])
    return [
        (box[0] - crop[0]) * sx,
        (box[1] - crop[1]) * sy,
        (box[2] - crop[0]) * sx,
        (box[3] - crop[1]) * sy,
    ]


def intersects(box: list[float], crop: tuple[int, int, int, int]) -> bool:
    return (
        min(box[2], crop[2]) > max(box[0], crop[0])
        and min(box[3], crop[3]) > max(box[1], crop[1])
    )


def render(
    raw: np.ndarray,
    gt_rows: list[dict],
    predictions: list[dict],
    crop: tuple[int, int, int, int],
    selected_id: int,
    label_every_id: bool,
) -> np.ndarray:
    cropped = raw[crop[1] : crop[3], crop[0] : crop[2]]
    canvas = cv2.resize(cropped, CELL_SIZE, interpolation=cv2.INTER_LANCZOS4)
    selected_gt = next(row["bbox"] for row in gt_rows if row["target_id"] == TARGET_ID)
    for row in gt_rows:
        if not intersects(row["bbox"], crop):
            continue
        selected = row["target_id"] == TARGET_ID
        color = COLORS["selected_gt"] if selected else COLORS["gt"]
        box = transform_box(row["bbox"], crop)
        dashed_box(canvas, box, color, 5 if selected else 3)
        if selected:
            label(canvas, max(0, int(box[0])), max(24, int(box[1])), "GT 219", color)

    selected_visible = False
    for prediction in predictions:
        if not intersects(prediction["bbox_xyxy"], crop):
            continue
        selected = int(prediction["sam3_obj_id"]) == selected_id
        box = transform_box(prediction["bbox_xyxy"], crop)
        correct = selected and iou(prediction["bbox_xyxy"], selected_gt) >= 0.5
        color = (
            COLORS["selected_correct"]
            if correct
            else COLORS["selected_wrong"]
            if selected
            else COLORS["other_prediction"]
        )
        cv2.rectangle(
            canvas,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color,
            5 if selected else 3,
        )
        if selected:
            selected_visible = True
        if selected or label_every_id:
            suffix = " OK" if correct else " ERR" if selected else ""
            label(
                canvas,
                max(0, int(box[0])),
                max(24, int(box[1])),
                f"ID {prediction['sam3_obj_id']}{suffix}",
                color,
            )
    if not selected_visible:
        label(canvas, 12, 35, f"ID {selected_id}: not detected", COLORS["selected_wrong"])
    return canvas


def render_gt(
    raw: np.ndarray,
    gt_rows: list[dict],
    crop: tuple[int, int, int, int],
) -> np.ndarray:
    """Render ground truth only, without any model predictions."""
    cropped = raw[crop[1] : crop[3], crop[0] : crop[2]]
    canvas = cv2.resize(cropped, CELL_SIZE, interpolation=cv2.INTER_LANCZOS4)
    for row in gt_rows:
        if not intersects(row["bbox"], crop):
            continue
        selected = row["target_id"] == TARGET_ID
        color = COLORS["selected_gt"] if selected else COLORS["gt"]
        box = transform_box(row["bbox"], crop)
        dashed_box(canvas, box, color, 5 if selected else 3)
        label(
            canvas,
            max(0, int(box[0])),
            max(24, int(box[1])),
            f"GT {row['target_id']}",
            color,
        )
    return canvas


def header(frame: int, width: int = 600) -> np.ndarray:
    bar = np.full((90, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        bar,
        f"Frame {frame} | official occlusion = 1",
        (14, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        bar,
        'Prompt: "car"',
        (14, 69),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )
    return bar


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gt = read_gt()
    ids = selected_ids()
    predictions = {model: read_predictions(model) for model, _ in MODELS}
    audit = []
    cells = {}
    gt_cells = {}

    for frame in FRAMES:
        target = next(row for row in gt[frame] if row["target_id"] == TARGET_ID)
        assert target["occlusion"] == 1
        crop = crop_for_target(target["bbox"])
        raw = cv2.imread(str(image_path(SEQUENCE, frame)))
        gt_cells[frame] = render_gt(raw, gt[frame], crop)
        cv2.imwrite(str(OUT / f"frame_{frame}_ground_truth.png"), gt_cells[frame])

        ours_matches = []
        for row in gt[frame]:
            if not intersects(row["bbox"], crop):
                continue
            best = max(
                (
                    iou(row["bbox"], prediction["bbox_xyxy"])
                    for prediction in predictions["sam3_adapted_with_replay"][frame]
                ),
                default=0.0,
            )
            ours_matches.append((row["target_id"], best))
        assert len(ours_matches) == 2
        assert all(score >= 0.5 for _, score in ours_matches)

        model_ious = {}
        for model, _ in MODELS:
            selected_prediction = next(
                (
                    row
                    for row in predictions[model][frame]
                    if int(row["sam3_obj_id"]) == ids[model]
                ),
                None,
            )
            selected_iou = (
                iou(target["bbox"], selected_prediction["bbox_xyxy"])
                if selected_prediction
                else 0.0
            )
            model_ious[model] = selected_iou
            cells[(model, frame)] = render(
                raw,
                gt[frame],
                predictions[model][frame],
                crop,
                ids[model],
                label_every_id=(model == "sam3_adapted_with_replay"),
            )
            cv2.imwrite(
                str(OUT / f"frame_{frame}_{model}.png"),
                cells[(model, frame)],
            )
        assert model_ious["sam3_adapted_with_replay"] >= 0.5
        assert model_ious["sam3_original"] < 0.5
        assert model_ious["sam3_native_id_tracker"] < 0.5
        audit.append(
            {
                "frame": frame,
                "official_occlusion": target["occlusion"],
                "crop_xyxy": crop,
                "cars_in_crop": len(ours_matches),
                "ours_car_matches": [
                    {"gt_id": target_id, "best_iou": score}
                    for target_id, score in ours_matches
                ],
                "selected_target_iou": model_ious,
            }
        )

    ours_strip = np.hstack(
        [
            np.vstack([header(frame), cells[("sam3_adapted_with_replay", frame)]])
            for frame in FRAMES
        ]
    )
    cv2.imwrite(str(OUT / "strict_adapted_only_ours_4frames.png"), ours_strip)

    comparison_rows = []
    for model, _ in MODELS:
        comparison_rows.append(
            np.hstack(
                [
                    np.vstack([header(frame), cells[(model, frame)]])
                    for frame in FRAMES
                ]
            )
        )
    cv2.imwrite(
        str(OUT / "strict_adapted_only_comparison_3x4.png"),
        np.vstack(comparison_rows),
    )

    # PPT layout requested by the user: four rows (GT + three methods)
    # and three columns (three continuous partial-occlusion frames).
    comparison_4x3_rows = [
        np.hstack(
            [
                np.vstack([header(frame), gt_cells[frame]])
                for frame in DISPLAY_FRAMES
            ]
        )
    ]
    for model, _ in MODELS:
        comparison_4x3_rows.append(
            np.hstack(
                [
                    np.vstack([header(frame), cells[(model, frame)]])
                    for frame in DISPLAY_FRAMES
                ]
            )
        )
    cv2.imwrite(
        str(OUT / "strict_adapted_only_comparison_4x3_with_gt.png"),
        np.vstack(comparison_4x3_rows),
    )

    metadata = {
        "event_id": EVENT_ID,
        "sequence_id": SEQUENCE,
        "prompt": PROMPT,
        "selected_gt_id": TARGET_ID,
        "displayed_frames": FRAMES,
        "strict_adapted_only_continuous_run": [659, 671],
        "scene_description": (
            "Night-time roadside vegetation and traffic-signal structure; "
            "not a complete tree disappearance."
        ),
        "audit": audit,
    }
    (OUT / "audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
