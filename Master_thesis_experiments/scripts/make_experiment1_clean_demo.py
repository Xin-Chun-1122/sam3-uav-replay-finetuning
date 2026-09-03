#!/usr/bin/env python3
"""Create text-free Experiment 1 demo videos from saved measured predictions."""

from __future__ import annotations

import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "videos/experiment1"
DATASET = ROOT / "datasets/raw/VisDrone2019-MOT-test-dev"
SEQUENCE = "uav0000077_00720_v"
START_FRAME = 175
END_FRAME = 214
FPS = 8.0
CACHE = VIDEO_DIR / "experiment1_car_video_detection_predictions.json"
MODEL_NAMES = (
    "Original SAM 3",
    "Adapted SAM 3 with replay (Ours)",
)
BOX_COLOR = (255, 180, 0)  # BGR: one cyan-blue color for both models.
HIGHLIGHT_COLOR = (0, 215, 255)  # BGR: yellow for Ours-only correct detections.
FOCUS_TARGET_ID = 15
FOCUS_FRAMES = range(178, 185)
FOCUS_CROP = (520, 210, 760, 390)  # left, top, right, bottom
CORRECT_DETECTION_FRAMES = range(201, 209)


def draw_predictions(frame, predictions):
    output = frame.copy()
    for row in predictions:
        x1, y1, x2, y2 = [int(round(value)) for value in row["bbox_xyxy"]]
        cv2.rectangle(output, (x1, y1), (x2, y2), BOX_COLOR, 3, cv2.LINE_AA)
    return output


def box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def load_car_gt():
    path = DATASET / "annotations" / f"{SEQUENCE}.txt"
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split(",")]
        frame, target_id, left, top, width, height, valid, category = values[:8]
        if int(category) != 4 or valid <= 0:
            continue
        output.setdefault(int(frame), []).append({
            "target_id": int(target_id),
            "bbox_xyxy": [left, top, left + width, top + height],
        })
    return output


def matched_prediction_for_target(predictions, gt, target_id):
    """Return the prediction assigned to target_id by the evaluation matcher."""
    matched = set()
    selected = None
    for prediction in sorted(predictions, key=lambda row: row["score"], reverse=True):
        choices = [
            (box_iou(prediction["bbox_xyxy"], target["bbox_xyxy"]), index)
            for index, target in enumerate(gt)
            if index not in matched
        ]
        best_iou, best_index = max(choices, default=(0.0, -1))
        if best_iou < 0.5:
            continue
        matched.add(best_index)
        if gt[best_index]["target_id"] == target_id:
            selected = prediction
    return selected


def matched_predictions(predictions, gt):
    """Return every correctly localized prediction under the locked IoU rule."""
    matched = set()
    correct = []
    for prediction in sorted(predictions, key=lambda row: row["score"], reverse=True):
        choices = [
            (box_iou(prediction["bbox_xyxy"], target["bbox_xyxy"]), index)
            for index, target in enumerate(gt)
            if index not in matched
        ]
        best_iou, best_index = max(choices, default=(0.0, -1))
        if best_iou < 0.5:
            continue
        matched.add(best_index)
        correct.append(prediction)
    return correct


def matched_predictions_by_target(predictions, gt):
    """Map each correctly localized prediction to its matched GT target ID."""
    matched = set()
    correct = {}
    for prediction in sorted(predictions, key=lambda row: row["score"], reverse=True):
        choices = [
            (box_iou(prediction["bbox_xyxy"], target["bbox_xyxy"]), index)
            for index, target in enumerate(gt)
            if index not in matched
        ]
        best_iou, best_index = max(choices, default=(0.0, -1))
        if best_iou < 0.5:
            continue
        matched.add(best_index)
        correct[gt[best_index]["target_id"]] = prediction
    return correct


def draw_target_predictions(frame, predictions_by_target, highlighted_targets=frozenset()):
    output = frame.copy()
    for target_id, row in predictions_by_target.items():
        x1, y1, x2, y2 = [int(round(value)) for value in row["bbox_xyxy"]]
        color = HIGHLIGHT_COLOR if target_id in highlighted_targets else BOX_COLOR
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    return output


def render_focus_panel(source, prediction):
    panel = source.copy()
    if prediction is not None:
        x1, y1, x2, y2 = [int(round(value)) for value in prediction["bbox_xyxy"]]
        cv2.rectangle(panel, (x1, y1), (x2, y2), BOX_COLOR, 3, cv2.LINE_AA)
    left, top, right, bottom = FOCUS_CROP
    return cv2.resize(
        panel[top:bottom, left:right], (960, 720), interpolation=cv2.INTER_CUBIC
    )


def open_writer(path: Path, size: tuple[int, int]):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer: {path}")
    return writer


def main() -> None:
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    car_gt = load_car_gt()
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    original_raw = VIDEO_DIR / "experiment1_clean_original_raw.mp4"
    ours_raw = VIDEO_DIR / "experiment1_clean_ours_raw.mp4"
    side_raw = VIDEO_DIR / "experiment1_clean_side_by_side_raw.mp4"

    first = cv2.imread(
        str(DATASET / "sequences" / SEQUENCE / f"{START_FRAME:07d}.jpg")
    )
    if first is None:
        raise FileNotFoundError("First source frame is missing")
    height, width = first.shape[:2]
    original_writer = open_writer(original_raw, (width, height))
    ours_writer = open_writer(ours_raw, (width, height))
    side_writer = open_writer(side_raw, (1920, 540))

    for frame_number in range(START_FRAME, END_FRAME + 1):
        source = cv2.imread(
            str(DATASET / "sequences" / SEQUENCE / f"{frame_number:07d}.jpg")
        )
        if source is None:
            raise FileNotFoundError(frame_number)
        panels = []
        for model_name in MODEL_NAMES:
            predictions = cache["models"][model_name]["predictions"][str(frame_number)]
            panels.append(draw_predictions(source, predictions))
        original_writer.write(panels[0])
        ours_writer.write(panels[1])
        left = cv2.resize(panels[0], (960, 540), interpolation=cv2.INTER_AREA)
        right = cv2.resize(panels[1], (960, 540), interpolation=cv2.INTER_AREA)
        side_writer.write(cv2.hconcat([left, right]))

    original_writer.release()
    ours_writer.release()
    side_writer.release()

    focus_original_raw = VIDEO_DIR / "experiment1_tiny_car_original_raw.mp4"
    focus_ours_raw = VIDEO_DIR / "experiment1_tiny_car_ours_raw.mp4"
    focus_side_raw = VIDEO_DIR / "experiment1_tiny_car_side_by_side_raw.mp4"
    focus_original_writer = open_writer(focus_original_raw, (960, 720))
    focus_ours_writer = open_writer(focus_ours_raw, (960, 720))
    focus_side_writer = open_writer(focus_side_raw, (1920, 720))
    rendered = []
    for frame_number in FOCUS_FRAMES:
        source = cv2.imread(
            str(DATASET / "sequences" / SEQUENCE / f"{frame_number:07d}.jpg")
        )
        panels = []
        for model_name in MODEL_NAMES:
            predictions = cache["models"][model_name]["predictions"][str(frame_number)]
            prediction = matched_prediction_for_target(
                predictions, car_gt[frame_number], FOCUS_TARGET_ID
            )
            panels.append(render_focus_panel(source, prediction))
        rendered.append(panels)

    # Hold each of the seven consecutive measured frames for visibility, then
    # extend the last frame to produce an exact five-second clip at 8 fps.
    output_frames = []
    for panels in rendered:
        output_frames.extend([panels] * 5)
    output_frames.extend([rendered[-1]] * 5)
    for left_panel, right_panel in output_frames:
        focus_original_writer.write(left_panel)
        focus_ours_writer.write(right_panel)
        focus_side_writer.write(cv2.hconcat([left_panel, right_panel]))
    focus_original_writer.release()
    focus_ours_writer.release()
    focus_side_writer.release()

    correct_original_raw = VIDEO_DIR / "experiment1_all_correct_cars_original_raw.mp4"
    correct_ours_raw = VIDEO_DIR / "experiment1_all_correct_cars_ours_raw.mp4"
    correct_side_raw = VIDEO_DIR / "experiment1_all_correct_cars_side_by_side_raw.mp4"
    correct_original_writer = open_writer(correct_original_raw, (960, 540))
    correct_ours_writer = open_writer(correct_ours_raw, (960, 540))
    correct_side_writer = open_writer(correct_side_raw, (1920, 540))
    correct_frames = []
    for frame_number in CORRECT_DETECTION_FRAMES:
        source = cv2.imread(
            str(DATASET / "sequences" / SEQUENCE / f"{frame_number:07d}.jpg")
        )
        panels = []
        for model_name in MODEL_NAMES:
            predictions = cache["models"][model_name]["predictions"][str(frame_number)]
            correct = matched_predictions(predictions, car_gt[frame_number])
            rendered_panel = draw_predictions(source, correct)
            panels.append(cv2.resize(
                rendered_panel, (960, 540), interpolation=cv2.INTER_AREA
            ))
        correct_frames.append(panels)

    # Eight consecutive measured frames, held for five output frames each.
    for panels in correct_frames:
        for _ in range(5):
            correct_original_writer.write(panels[0])
            correct_ours_writer.write(panels[1])
            correct_side_writer.write(cv2.hconcat(panels))
    correct_original_writer.release()
    correct_ours_writer.release()
    correct_side_writer.release()

    final_raw = VIDEO_DIR / "experiment1_final_ours_highlight_8s_raw.mp4"
    final_writer = open_writer(final_raw, (1920, 540))
    for frame_number in CORRECT_DETECTION_FRAMES:
        source = cv2.imread(
            str(DATASET / "sequences" / SEQUENCE / f"{frame_number:07d}.jpg")
        )
        original_predictions = cache["models"][MODEL_NAMES[0]]["predictions"][str(frame_number)]
        ours_predictions = cache["models"][MODEL_NAMES[1]]["predictions"][str(frame_number)]
        original_by_target = matched_predictions_by_target(
            original_predictions, car_gt[frame_number]
        )
        ours_by_target = matched_predictions_by_target(
            ours_predictions, car_gt[frame_number]
        )
        ours_only_targets = set(ours_by_target) - set(original_by_target)
        left = draw_target_predictions(source, original_by_target)
        right = draw_target_predictions(source, ours_by_target, ours_only_targets)
        left = cv2.resize(left, (960, 540), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (960, 540), interpolation=cv2.INTER_AREA)
        combined = cv2.hconcat([left, right])
        # Eight consecutive evaluated frames, each held for one second.
        for _ in range(8):
            final_writer.write(combined)
    final_writer.release()
    print(f"original_raw={original_raw}")
    print(f"ours_raw={ours_raw}")
    print(f"side_raw={side_raw}")
    print(f"focus_original_raw={focus_original_raw}")
    print(f"focus_ours_raw={focus_ours_raw}")
    print(f"focus_side_raw={focus_side_raw}")
    print(f"correct_original_raw={correct_original_raw}")
    print(f"correct_ours_raw={correct_ours_raw}")
    print(f"correct_side_raw={correct_side_raw}")
    print(f"final_raw={final_raw}")


if __name__ == "__main__":
    main()
