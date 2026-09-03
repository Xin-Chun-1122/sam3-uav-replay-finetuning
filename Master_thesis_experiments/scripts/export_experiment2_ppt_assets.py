#!/usr/bin/env python3
"""Export audit-friendly Experiment 2 PPT panels and short videos.

All boxes are rendered from the existing VisDrone-MOT annotations and measured
SAM 3 prediction JSONL files. No detections are invented by this script.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/home/alien/sam3/Master_thesis_experiments")
DATA = ROOT / "datasets/raw/VisDrone2019-MOT-test-dev"
EVENTS_PATH = ROOT / "annotations/experiment2/events.json"
RESULTS = ROOT / "results/experiment2"
PREDICTIONS = ROOT / "predictions/experiment2"
OUTPUT = ROOT / "figures/experiment2/ppt_local_tracking_cases"
VIDEO_OUTPUT = ROOT / "videos/experiment2"

CLASS_NAMES = {
    1: "person",
    2: "person",
    4: "car",
    5: "van",
    6: "truck",
    9: "bus",
    10: "motorcycle",
}

MODELS = {
    "ground_truth": "Ground Truth",
    "sam3_original": "Original SAM 3",
    "sam3_native_id_tracker": "SAM 3 + Native-ID Refinement",
    "sam3_adapted_with_replay": "Ours",
}

CASES = {
    "partial_occlusion": {
        "event_id": "partial_occlusion-018",
        "frames": {"before": 205, "during": 220, "after": 237},
        "prompt": "car",
        "subtitle": "Official partial occlusion with complete local prompt coverage",
        "phase_crop_width": 240,
    },
    "temporary_complete_disappearance": {
        "event_id": "temporary_complete_disappearance-031",
        "frames": {"before": 103, "during": 112, "after": 123},
        "prompt": "car",
        "subtitle": "Ours recovers the same native ID after complete disappearance under an overpass",
        # Phase-specific local crops contain every visible car completely enough
        # for PPT inspection.  In the before/after crops GT 23 is the only
        # visible car and is matched by Ours; in the during crop no car GT is
        # visible because the target is fully hidden by the overpass.
        "phase_crops": {
            "before": [500, 471, 1075, 796],
            "during": [750, 350, 1070, 548],
            "after": [995, 326, 1420, 626],
        },
    },
    "scale_decrease_and_recovery": {
        "event_id": "scale_decrease_and_recovery-031",
        "frames": {"before": 5, "during": 57, "after": 190},
        "prompt": "car",
        "subtitle": "Only Ours maintains the same track through scale decrease and recovery",
        # Equal-size 4:3 local crops retain the target plus one nearby
        # vehicle, so every visible context vehicle remains legible in PPT.
        "phase_crops": {
            "before": [585, 484, 685, 559],
            "during": [780, 420, 880, 495],
            "after": [885, 515, 985, 590],
        },
        # The crop contains only a few vehicles, so retain every model output
        # instead of hiding non-target detections.
        "selected_prediction_only": False,
        "gt_categories": ["car", "van", "truck", "bus", "motorcycle"],
        "gt_legend": "Dashed boxes: visible vehicle GT; yellow: tracked target; blue: correct ID; red: error/miss",
        "phase_labels": {
            "before": "Before | Frame 5 | s=32.40",
            "during": "Tiny | Frame 57 | s=26.38",
            "after": "Recovery | Frame 190 | s=36.95",
        },
    },
    "camera_motion_or_blur": {
        "event_id": "camera_motion_or_blur-007",
        "frames": {"before": 428, "during": 438, "after": 442},
        "prompt": "truck",
        "subtitle": "Ours succeeds under camera-motion blur while Native-ID fails",
        "phase_labels": {
            "before": "Before | Frame 428 | sharpness=653.52",
            "during": "Motion blur | Frame 438 | sharpness=175.21",
            "after": "After | Frame 442 | sharpness=565.63",
        },
    },
}

VIDEO_CASES = {
    **CASES,
    "partial_occlusion": {
        "event_id": "partial_occlusion-007",
        "frames": {"before": 643, "during": 659, "after": 672},
        "prompt": "car",
        "subtitle": "Official partial occlusion near roadside vegetation and signals",
    },
}

PHASE_NAMES = {"before": "Before", "during": "During", "after": "After"}

COLORS = {
    "gt": (56, 201, 95),
    "selected_gt": (0, 210, 255),
    "other_prediction": (235, 180, 30),
    "selected_correct": (230, 130, 30),
    "selected_wrong": (45, 45, 230),
    "selected_hidden": (170, 80, 170),
    "text_bg": (18, 18, 18),
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def image_path(sequence: str, frame: int) -> Path:
    path = DATA / "sequences" / sequence / f"{frame:07d}.jpg"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_all_gt(sequence: str) -> dict[int, list[dict]]:
    rows: dict[int, list[dict]] = {}
    path = DATA / "annotations" / f"{sequence}.txt"
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split(",")
        if len(fields) < 10:
            continue
        frame, target_id = int(fields[0]), int(fields[1])
        x, y, width, height = map(float, fields[2:6])
        valid, class_id = int(fields[6]), int(fields[7])
        if not valid or class_id not in CLASS_NAMES or width <= 0 or height <= 0:
            continue
        rows.setdefault(frame, []).append(
            {
                "target_id": target_id,
                "category": CLASS_NAMES[class_id],
                "bbox_xyxy": [x, y, x + width, y + height],
                "occlusion": int(fields[9]),
            }
        )
    return rows


def read_predictions(model: str, event_id: str) -> dict[int, list[dict]]:
    path = PREDICTIONS / model / f"{event_id}.jsonl"
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        rows[int(item["frame_index"])] = item["predictions"]
    return rows


def result_index() -> tuple[dict[str, dict[str, dict]], dict[str, dict[str, int]]]:
    results, selected = {}, {}
    for model in list(MODELS)[1:]:
        data = load_json(RESULTS / f"{model}.json")
        results[model] = {row["event_id"]: row for row in data["event_results"]}
        selected[model] = {
            row["event_id"]: (
                int(row["selected_sam3_obj_id"])
                if row["selected_sam3_obj_id"] is not None
                else None
            )
            for row in data["event_results"]
        }
    return results, selected


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def intersects(box: list[float], crop: tuple[int, int, int, int]) -> bool:
    return (
        min(box[2], crop[2]) > max(box[0], crop[0])
        and min(box[3], crop[3]) > max(box[1], crop[1])
    )


def target_gt(
    gt: dict[int, list[dict]], frame: int, target_id: int
) -> list[float] | None:
    for row in gt.get(frame, []):
        if row["target_id"] == target_id:
            return row["bbox_xyxy"]
    return None


def interpolated_target(
    gt: dict[int, list[dict]], frame: int, target_id: int
) -> list[float]:
    available = sorted(
        f
        for f, rows in gt.items()
        if any(row["target_id"] == target_id for row in rows)
    )
    before = [f for f in available if f < frame]
    after = [f for f in available if f > frame]
    if before and after:
        left, right = max(before), min(after)
        a = np.asarray(target_gt(gt, left, target_id), dtype=float)
        b = np.asarray(target_gt(gt, right, target_id), dtype=float)
        alpha = (frame - left) / (right - left)
        return (a * (1 - alpha) + b * alpha).tolist()
    nearest = min(available, key=lambda value: abs(value - frame))
    return target_gt(gt, nearest, target_id)


def fixed_case_crop(
    sequence: str,
    gt: dict[int, list[dict]],
    target_id: int,
    frames: dict[str, int],
) -> tuple[int, int, int, int]:
    sample = cv2.imread(str(image_path(sequence, next(iter(frames.values())))))
    height, width = sample.shape[:2]
    references = [
        target_gt(gt, frame, target_id) or interpolated_target(gt, frame, target_id)
        for frame in frames.values()
    ]
    x1 = min(box[0] for box in references)
    y1 = min(box[1] for box in references)
    x2 = max(box[2] for box in references)
    y2 = max(box[3] for box in references)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    spread_w, spread_h = x2 - x1, y2 - y1
    crop_w = int(np.clip(max(720, spread_w + 520), 720, min(1080, width)))
    crop_h = int(np.clip(max(450, spread_h + 300), 450, min(675, height)))
    target_ratio = 16 / 10
    if crop_w / crop_h < target_ratio:
        crop_w = min(width, int(round(crop_h * target_ratio)))
    else:
        crop_h = min(height, int(round(crop_w / target_ratio)))
    left = int(np.clip(cx - crop_w / 2, 0, width - crop_w))
    top = int(np.clip(cy - crop_h / 2, 0, height - crop_h))
    return left, top, left + crop_w, top + crop_h


def centered_phase_crop(
    sequence: str,
    reference: list[float],
    crop_width: int,
) -> tuple[int, int, int, int]:
    sample = cv2.imread(str(next((DATA / "sequences" / sequence).glob("*.jpg"))))
    height, width = sample.shape[:2]
    crop_w = min(int(crop_width), width)
    crop_h = min(int(round(crop_w / 1.6)), height)
    cx = (reference[0] + reference[2]) / 2
    cy = (reference[1] + reference[3]) / 2
    left = int(np.clip(cx - crop_w / 2, 0, width - crop_w))
    top = int(np.clip(cy - crop_h / 2, 0, height - crop_h))
    return left, top, left + crop_w, top + crop_h


def dashed_rectangle(
    image: np.ndarray,
    box: list[float],
    crop: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int = 3,
    dash: int = 12,
) -> None:
    thickness = max(1, min(thickness, int(round(image.shape[1] / 240))))
    dash = max(3, min(dash, image.shape[1] // 30))
    x0, y0 = crop[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = x1 - x0, x2 - x0
    y1, y2 = y1 - y0, y2 - y0
    for start in range(x1, x2, dash * 2):
        cv2.line(image, (start, y1), (min(start + dash, x2), y1), color, thickness)
        cv2.line(image, (start, y2), (min(start + dash, x2), y2), color, thickness)
    for start in range(y1, y2, dash * 2):
        cv2.line(image, (x1, start), (x1, min(start + dash, y2)), color, thickness)
        cv2.line(image, (x2, start), (x2, min(start + dash, y2)), color, thickness)


def label(
    image: np.ndarray,
    position: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    scale: float = 0.52,
) -> None:
    # Very tight target-centric crops (used for scale-change comparison)
    # need compact labels so the annotation does not cover the object.
    scale = min(scale, max(0.08, image.shape[1] / 700 * 0.52))
    x, y = position
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
    )
    y = max(height + 5, y)
    cv2.rectangle(
        image,
        (x, y - height - 5),
        (x + width + 6, y + baseline + 2),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 3, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_gt(
    canvas: np.ndarray,
    rows: list[dict],
    prompt: str,
    target_id: int,
    crop: tuple[int, int, int, int],
    selected_only: bool = False,
    categories: set[str] | None = None,
) -> None:
    x0, y0 = crop[:2]
    for row in rows:
        allowed = categories if categories is not None else {prompt}
        if row["category"] not in allowed or not intersects(row["bbox_xyxy"], crop):
            continue
        selected = row["target_id"] == target_id
        if selected_only and not selected:
            continue
        color = COLORS["selected_gt"] if selected else COLORS["gt"]
        dashed_rectangle(
            canvas,
            row["bbox_xyxy"],
            crop,
            color,
            thickness=4 if selected else 2,
        )
        if selected:
            x, y = int(row["bbox_xyxy"][0] - x0), int(row["bbox_xyxy"][1] - y0)
            label(canvas, (max(0, x), max(18, y)), f"GT {row['target_id']}", color)
        elif categories is not None:
            # Context vehicles are explicitly named so the cropped PPT panel
            # cannot be mistaken for a single-object-only annotation.
            x, y = int(row["bbox_xyxy"][0] - x0), int(row["bbox_xyxy"][1] - y0)
            label(canvas, (max(0, x), max(18, y)), row["category"], color)


def draw_predictions(
    canvas: np.ndarray,
    predictions: list[dict],
    selected_id: int,
    selected_gt_box: list[float] | None,
    crop: tuple[int, int, int, int],
    label_all_predictions: bool = False,
    selected_only: bool = False,
) -> None:
    x0, y0 = crop[:2]
    visible = [
        row
        for row in predictions
        if intersects(row["bbox_xyxy"], crop)
    ]
    visible.sort(key=lambda row: row["sam3_obj_id"] == selected_id)
    has_id_switch = False
    for row in visible:
        box = row["bbox_xyxy"]
        selected = int(row["sam3_obj_id"]) == selected_id
        if selected_only and not selected:
            continue
        id_switch = False
        if selected:
            if selected_gt_box is None:
                correct = None
                color = COLORS["selected_hidden"]
                suffix = " HIDDEN"
            else:
                correct = iou(box, selected_gt_box) >= 0.5
                color = (
                    COLORS["selected_correct"]
                    if correct
                    else COLORS["selected_wrong"]
                )
                suffix = "" if correct else " ERR"
            thickness = 4
        else:
            # A different native ID overlapping the selected GT is an ID
            # switch, not successful same-ID recovery.
            id_switch = (
                selected_gt_box is not None
                and iou(box, selected_gt_box) >= 0.5
            )
            color = (
                COLORS["selected_wrong"]
                if id_switch
                else COLORS["other_prediction"]
            )
            thickness = 4 if id_switch else 2
            suffix = " ID SWITCH" if id_switch else ""
            has_id_switch = has_id_switch or id_switch
        x1, y1, x2, y2 = [
            int(round(box[0] - x0)),
            int(round(box[1] - y0)),
            int(round(box[2] - x0)),
            int(round(box[3] - y0)),
        ]
        thickness = max(
            1,
            min(thickness, int(round(canvas.shape[1] / 240))),
        )
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        if selected or id_switch or label_all_predictions:
            text = (
                f"ID {row['sam3_obj_id']}"
                if id_switch
                else f"ID {row['sam3_obj_id']}{suffix}"
            )
            label(
                canvas,
                (max(0, x1), max(18, y1)),
                text,
                color,
            )
    if not any(int(row["sam3_obj_id"]) == selected_id for row in visible):
        if selected_gt_box is None:
            label(
                canvas,
                (8, 24),
                f"ID {selected_id} HIDDEN",
                COLORS["selected_hidden"],
            )
        elif not has_id_switch:
            label(
                canvas,
                (8, 24),
                f"ID {selected_id} MISS",
                COLORS["selected_wrong"],
            )


def render_cell(
    image: np.ndarray,
    gt_rows: list[dict],
    prompt: str,
    target_id: int,
    crop: tuple[int, int, int, int],
    predictions: list[dict] | None = None,
    selected_id: int | None = None,
    label_all_predictions: bool = False,
    selected_only: bool = False,
    gt_categories: set[str] | None = None,
) -> np.ndarray:
    x1, y1, x2, y2 = crop
    canvas = image[y1:y2, x1:x2].copy()
    draw_gt(
        canvas,
        gt_rows,
        prompt,
        target_id,
        crop,
        # When contextual vehicle categories are requested, keep their GT
        # boxes visible in every model row as the reference for completeness.
        selected_only if gt_categories is None else False,
        gt_categories,
    )
    if predictions is not None and selected_id is not None:
        draw_predictions(
            canvas,
            predictions,
            selected_id,
            next(
                (
                    row["bbox_xyxy"]
                    for row in gt_rows
                    if row["target_id"] == target_id
                ),
                None,
            ),
            crop,
            label_all_predictions,
            selected_only,
        )
    return cv2.resize(canvas, (960, 600), interpolation=cv2.INTER_LANCZOS4)


def add_fixed_scale_inset(
    base: np.ndarray,
    image: np.ndarray,
    gt_box: list[float],
    prediction_box: list[float] | None,
) -> np.ndarray:
    """Add a fixed 64x48-source-pixel inset without changing scale by phase."""
    output = base.copy()
    height, width = image.shape[:2]
    cx = (gt_box[0] + gt_box[2]) / 2
    cy = (gt_box[1] + gt_box[3]) / 2
    crop_w, crop_h = 64, 48
    x1 = max(0, min(width - crop_w, int(round(cx - crop_w / 2))))
    y1 = max(0, min(height - crop_h, int(round(cy - crop_h / 2))))
    crop = (x1, y1, x1 + crop_w, y1 + crop_h)
    inset = image[y1 : y1 + crop_h, x1 : x1 + crop_w].copy()
    dashed_rectangle(inset, gt_box, crop, COLORS["selected_gt"], thickness=2)
    if prediction_box is not None:
        px1, py1, px2, py2 = [
            int(round(prediction_box[0] - x1)),
            int(round(prediction_box[1] - y1)),
            int(round(prediction_box[2] - x1)),
            int(round(prediction_box[3] - y1)),
        ]
        cv2.rectangle(
            inset,
            (px1, py1),
            (px2, py2),
            COLORS["selected_correct"],
            2,
        )
    inset = cv2.resize(inset, (320, 240), interpolation=cv2.INTER_NEAREST)
    ox, oy = output.shape[1] - 336, output.shape[0] - 256
    cv2.rectangle(output, (ox - 4, oy - 4), (ox + 324, oy + 244), (255, 255, 255), -1)
    output[oy : oy + 240, ox : ox + 320] = inset
    cv2.rectangle(output, (ox, oy), (ox + 319, oy + 239), (20, 20, 20), 2)
    return output


def make_contact_sheet(
    cells: dict[tuple[str, str], np.ndarray],
    case: dict,
    event: dict,
    output_path: Path,
) -> None:
    cell_w, cell_h = 640, 400
    left, top, header, gap = 220, 155, 70, 10
    rows = list(MODELS)
    phases = list(PHASE_NAMES)
    width = left + len(phases) * (cell_w + gap) + gap
    height = top + len(rows) * (cell_h + gap) + header
    sheet = np.full((height, width, 3), 255, dtype=np.uint8)
    title = f"{case['subtitle']} | Prompt: \"{case['prompt']}\" | {event['event_id']}"
    cv2.putText(
        sheet, title, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2
    )
    cv2.putText(
        sheet,
        case.get(
            "gt_legend",
            "Dashed boxes: all prompt GT; blue: correct selected native ID; red: error; purple: hidden target",
        ),
        (25, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )
    for col, phase in enumerate(phases):
        x = left + col * (cell_w + gap)
        phase_label = case.get("phase_labels", {}).get(
            phase, f"{PHASE_NAMES[phase]} | Frame {case['frames'][phase]}"
        )
        cv2.putText(
            sheet,
            phase_label,
            (x + 12, top - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    for row_index, model in enumerate(rows):
        y = top + row_index * (cell_h + gap)
        cv2.putText(
            sheet,
            MODELS[model],
            (15, y + cell_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        for col, phase in enumerate(phases):
            x = left + col * (cell_w + gap)
            resized = cv2.resize(cells[(model, phase)], (cell_w, cell_h))
            sheet[y : y + cell_h, x : x + cell_w] = resized
            cv2.rectangle(sheet, (x, y), (x + cell_w, y + cell_h), (40, 40, 40), 1)
    cv2.imwrite(str(output_path), sheet)


def export_panels() -> list[dict]:
    events = {row["event_id"]: row for row in load_json(EVENTS_PATH)}
    results, selected = result_index()
    manifest = []
    for condition, case in CASES.items():
        event = events[case["event_id"]]
        sequence = event["sequence_id"]
        target_id = int(event["target_id"])
        gt = read_all_gt(sequence)
        if "phase_crops" in case:
            crops = {
                phase: tuple(case["phase_crops"][phase])
                for phase in case["frames"]
            }
        elif "phase_crop_width" in case:
            crops = {
                phase: centered_phase_crop(
                    sequence,
                    target_gt(gt, frame, target_id)
                    or interpolated_target(gt, frame, target_id),
                    int(case["phase_crop_width"]),
                )
                for phase, frame in case["frames"].items()
            }
        else:
            shared_crop = fixed_case_crop(sequence, gt, target_id, case["frames"])
            crops = {phase: shared_crop for phase in case["frames"]}
        case_dir = OUTPUT / condition
        case_dir.mkdir(parents=True, exist_ok=True)
        model_predictions = {
            model: read_predictions(model, event["event_id"])
            for model in list(MODELS)[1:]
        }
        cells = {}
        selected_prediction_only = bool(case.get("selected_prediction_only", False))
        gt_categories = (
            set(case["gt_categories"])
            if case.get("gt_categories")
            else None
        )
        for phase, frame in case["frames"].items():
            crop = crops[phase]
            image = cv2.imread(str(image_path(sequence, frame)))
            gt_cell = render_cell(
                image,
                gt.get(frame, []),
                case["prompt"],
                target_id,
                crop,
                selected_only=False,
                gt_categories=gt_categories,
            )
            cells[("ground_truth", phase)] = gt_cell
            cv2.imwrite(str(case_dir / f"{phase}_ground_truth.png"), gt_cell)
            for model in list(MODELS)[1:]:
                cell = render_cell(
                    image,
                    gt.get(frame, []),
                    case["prompt"],
                    target_id,
                    crop,
                    model_predictions[model].get(frame, []),
                    selected[model][event["event_id"]],
                    label_all_predictions=False,
                    selected_only=selected_prediction_only,
                    gt_categories=gt_categories,
                )
                cells[(model, phase)] = cell
                cv2.imwrite(str(case_dir / f"{phase}_{model}.png"), cell)
        make_contact_sheet(
            cells,
            case,
            event,
            case_dir / "comparison_labeled.png",
        )
        clean_rows = []
        for model in MODELS:
            clean_rows.append(
                np.hstack([cells[(model, phase)] for phase in PHASE_NAMES])
            )
        cv2.imwrite(
            str(case_dir / "comparison_clean_4x3.png"),
            np.vstack(clean_rows),
        )
        if condition == "scale_decrease_and_recovery":
            inset_panels = []
            model = "sam3_adapted_with_replay"
            selected_id = selected[model][event["event_id"]]
            for phase, frame in case["frames"].items():
                image = cv2.imread(str(image_path(sequence, frame)))
                gt_row = next(
                    row
                    for row in gt.get(frame, [])
                    if row["target_id"] == target_id
                )
                prediction = next(
                    (
                        row
                        for row in model_predictions[model].get(frame, [])
                        if int(row["sam3_obj_id"]) == selected_id
                    ),
                    None,
                )
                panel = add_fixed_scale_inset(
                    cells[(model, phase)],
                    image,
                    gt_row["bbox_xyxy"],
                    prediction["bbox_xyxy"] if prediction else None,
                )
                cv2.imwrite(
                    str(case_dir / f"{phase}_ours_with_fixed_scale_inset.png"),
                    panel,
                )
                inset_panels.append(panel)
            cv2.imwrite(
                str(case_dir / "ours_scale_change_with_fixed_insets.png"),
                np.hstack(inset_panels),
            )
        case_results = {
            MODELS[model]: {
                "selected_native_sam3_id": selected[model][event["event_id"]],
                "frame_recall": results[model][event["event_id"]]["frame_recall"],
                "same_id_recovered": results[model][event["event_id"]][
                    "same_id_recovered"
                ],
                "phase_counts": results[model][event["event_id"]]["phase_counts"],
            }
            for model in list(MODELS)[1:]
        }
        manifest.append(
            {
                "condition": condition,
                "subtitle": case["subtitle"],
                "event_id": event["event_id"],
                "sequence_id": sequence,
                "target_id": target_id,
                "prompt": case["prompt"],
                "frames": case["frames"],
                "crop_xyxy": crops,
                "duration_frames": event["duration_frames"],
                "source_type": event["source_type"],
                "event_measurements": {
                    key: event[key]
                    for key in [
                        "pre_scale",
                        "minimum_scale",
                        "post_scale",
                        "median_blur",
                        "median_global_motion",
                        "blur_threshold",
                        "motion_threshold",
                    ]
                    if key in event
                },
                "measured_results": case_results,
            }
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "case_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def video_frame(
    raw: np.ndarray,
    gt_rows: list[dict],
    predictions: list[dict],
    prompt: str,
    target_id: int,
    selected_id: int,
    crop: tuple[int, int, int, int],
    frame: int,
    event: dict,
) -> np.ndarray:
    panel = render_cell(
        raw,
        gt_rows,
        prompt,
        target_id,
        crop,
        predictions,
        selected_id,
    )
    output = cv2.resize(panel, (1280, 800), interpolation=cv2.INTER_LANCZOS4)
    bar = np.full((100, 1280, 3), 245, dtype=np.uint8)
    if frame < event["start_frame"]:
        phase = "Before event"
    elif frame <= event["end_frame"]:
        phase = "During event"
    else:
        phase = "After event"
    cv2.putText(
        bar,
        f'Ours | Prompt: "{prompt}" | {phase} | Frame {frame}',
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        bar,
        "Dashed boxes: all visible prompt GT; solid boxes: all SAM 3 native track IDs",
        (25, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (55, 55, 55),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, output])


def export_video(condition: str, fps: float, filename: str) -> dict:
    events = {row["event_id"]: row for row in load_json(EVENTS_PATH)}
    case = VIDEO_CASES[condition]
    event = events[case["event_id"]]
    model = "sam3_adapted_with_replay"
    _, selected = result_index()
    selected_id = selected[model][event["event_id"]]
    gt = read_all_gt(event["sequence_id"])
    crop = fixed_case_crop(event["sequence_id"], gt, int(event["target_id"]), case["frames"])
    predictions = read_predictions(model, event["event_id"])
    VIDEO_OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = VIDEO_OUTPUT / filename
    raw_path = output_path.with_name(f"{output_path.stem}_raw.mp4")
    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (1280, 900),
    )
    for frame in range(event["clip_start_frame"], event["clip_end_frame"] + 1):
        raw = cv2.imread(str(image_path(event["sequence_id"], frame)))
        rendered = video_frame(
            raw,
            gt.get(frame, []),
            predictions.get(frame, []),
            case["prompt"],
            int(event["target_id"]),
            selected_id,
            crop,
            frame,
            event,
        )
        writer.write(rendered)
    writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output_path),
        ],
        check=True,
    )
    raw_path.unlink()
    count = event["clip_end_frame"] - event["clip_start_frame"] + 1
    return {
        "path": str(output_path),
        "frames": count,
        "fps": fps,
        "duration_seconds": count / fps,
        "event_id": event["event_id"],
        "prompt": case["prompt"],
        "selected_native_sam3_id": selected_id,
    }


def write_readme(manifest: list[dict], videos: list[dict]) -> None:
    lines = [
        "# Experiment 2 PPT local tracking cases",
        "",
        "These assets are rendered only from VisDrone-MOT GT annotations and saved",
        "SAM 3 prediction JSONL files. No box is manually invented.",
        "",
        "## Box legend",
        "",
        "- dashed green: every visible GT instance matching the prompt in the crop",
        "- dashed yellow: selected evaluation target GT",
        "- thin cyan: all other SAM 3 native track IDs returned for the same prompt",
        "- thick blue `ID n OK`: selected native SAM 3 ID, IoU >= 0.5",
        "- thick red `ID n ERR`: selected native SAM 3 ID, absent or IoU < 0.5",
        "- thick purple `ID n HIDDEN`: target is absent from GT during complete disappearance",
        "",
        "Use the 960x600 individual cells to build a PPT table. The same fixed crop",
        "is used for every model and phase within one scenario.",
        "",
        "## Selected cases",
        "",
    ]
    for row in manifest:
        measurements = row["event_measurements"]
        lines.extend(
            [
                f"### {row['condition']}",
                "",
                f"- Event: `{row['event_id']}`",
                f"- Prompt: `{row['prompt']}`",
                f"- Sequence: `{row['sequence_id']}`; target ID: {row['target_id']}",
                f"- Frames: {row['frames']}",
                f"- Natural event duration: {row['duration_frames']} frames",
                f"- Description: {row['subtitle']}",
            ]
        )
        if measurements:
            lines.append(f"- Measurements: `{json.dumps(measurements)}`")
        lines.append("")
    lines.extend(["## Videos", ""])
    for video in videos:
        lines.append(
            f"- `{video['path']}` — {video['duration_seconds']:.2f} s, "
            f'prompt `{video["prompt"]}`, event `{video["event_id"]}`'
        )
    lines.append("")
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    manifest = export_panels()
    videos = [
        export_video(
            "partial_occlusion",
            7.0,
            "experiment2_partial_occlusion_ours_recovery_8s_h264.mp4",
        ),
        export_video(
            "temporary_complete_disappearance",
            6.0,
            "experiment2_overpass_disappearance_ours_recovery_8s_h264.mp4",
        ),
    ]
    (VIDEO_OUTPUT / "video_manifest.json").write_text(
        json.dumps(videos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(manifest, videos)
    print(json.dumps({"panels": manifest, "videos": videos}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
