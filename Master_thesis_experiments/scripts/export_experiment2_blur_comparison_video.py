#!/usr/bin/env python3
"""Export an 8-second, verified blur comparison video for Experiment 2."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from export_experiment2_ppt_assets import image_path, read_all_gt
from infer_sam3_mot_events import motion_blur


ROOT = Path("/home/alien/sam3/Master_thesis_experiments")
EVENT_ID = "camera_motion_blur_fullclip_k31-001"
SEQUENCE_ID = "uav0000077_00720_v"
TARGETS = {46: 0, 5: 2}  # VisDrone GT target ID -> native SAM 3 object ID
FRAMES = list(range(414, 438))
FPS = 3.0
OUTPUT_DIR = ROOT / "videos/experiment2"
OUTPUT = OUTPUT_DIR / "experiment2_motion_blur_ours_vs_native_id_8s.mp4"
OUTPUT_H264 = OUTPUT_DIR / "experiment2_motion_blur_ours_vs_native_id_8s_h264.mp4"
PRED_ROOT = ROOT / "predictions/experiment2_camera_blur_fullclip"

# A landscape crop preserves the complete evaluated trajectory and enough UAV
# context for a 16:9 presentation without geometrically stretching the image.
CROP = (400, 60, 975, 765)
PANEL_SIZE = (590, 620)
CYAN = (255, 210, 0)       # BGR: common correct result
YELLOW = (0, 230, 255)     # BGR: Ours-only correct result


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def load_predictions(model: str) -> dict[int, list[dict]]:
    path = PRED_ROOT / model / f"{EVENT_ID}.jsonl"
    return {
        int(row["frame_index"]): row["predictions"]
        for row in (json.loads(line) for line in path.read_text().splitlines())
    }


def target_gt(gt_by_frame: dict[int, list[dict]], frame: int, target_id: int) -> dict:
    return next(item for item in gt_by_frame[frame] if int(item["target_id"]) == target_id)


def matching_prediction(predictions: list[dict], gt_box: list[float], obj_id: int) -> dict | None:
    candidates = [p for p in predictions if int(p["sam3_obj_id"]) == obj_id]
    if not candidates:
        return None
    best = max(candidates, key=lambda p: iou(p["bbox_xyxy"], gt_box))
    return best if iou(best["bbox_xyxy"], gt_box) >= 0.5 else None


def draw_panel(
    image: np.ndarray,
    rendered_predictions: list[tuple[dict, tuple[int, int, int]]],
) -> np.ndarray:
    x1, y1, x2, y2 = CROP
    crop = image[y1:y2, x1:x2].copy()
    scale = min(PANEL_SIZE[0] / (x2 - x1), PANEL_SIZE[1] / (y2 - y1))
    rendered_w, rendered_h = round((x2 - x1) * scale), round((y2 - y1) * scale)
    crop = cv2.resize(crop, (rendered_w, rendered_h), interpolation=cv2.INTER_AREA)
    panel = np.full((PANEL_SIZE[1], PANEL_SIZE[0], 3), 245, dtype=np.uint8)
    offset_x = (PANEL_SIZE[0] - rendered_w) // 2
    offset_y = (PANEL_SIZE[1] - rendered_h) // 2
    panel[offset_y:offset_y + rendered_h, offset_x:offset_x + rendered_w] = crop
    for prediction, color in rendered_predictions:
        px1, py1, px2, py2 = prediction["bbox_xyxy"]
        p1 = (offset_x + round((px1 - x1) * scale), offset_y + round((py1 - y1) * scale))
        p2 = (offset_x + round((px2 - x1) * scale), offset_y + round((py2 - y1) * scale))
        cv2.rectangle(panel, p1, p2, color, 5, cv2.LINE_AA)
        label = f"ID {int(prediction['sam3_obj_id'])}"
        cv2.putText(panel, label, (max(4, p1[0]), max(28, p1[1] - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 3, cv2.LINE_AA)
    return panel


def compose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    canvas = np.full((720, 1280, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "SAM3 + Native ID tracker", (125, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Ours", (925, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 2, cv2.LINE_AA)
    canvas[72:692, 35:625] = left
    canvas[72:692, 655:1245] = right
    return canvas


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = read_all_gt(SEQUENCE_ID)
    native = load_predictions("sam3_native_id_tracker")
    ours = load_predictions("sam3_adapted_with_replay")
    writer = cv2.VideoWriter(str(OUTPUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    assert writer.isOpened()

    audit = []
    for frame in FRAMES:
        source = cv2.imread(str(image_path(SEQUENCE_ID, frame)))
        source = motion_blur(source, 31, 12.0)
        native_rendered = []
        ours_rendered = []
        frame_audit = {"frame": frame, "targets": {}}
        for target_id, native_obj_id in TARGETS.items():
            gt_box = target_gt(gt, frame, target_id)["bbox_xyxy"]
            native_match = matching_prediction(native.get(frame, []), gt_box, native_obj_id)
            ours_match = matching_prediction(ours.get(frame, []), gt_box, native_obj_id)
            if ours_match is None:
                raise RuntimeError(f"Ours is not correct for GT {target_id} at frame {frame}")
            if native_match is not None:
                native_rendered.append((native_match, CYAN))
            ours_rendered.append((ours_match, CYAN if native_match is not None else YELLOW))
            frame_audit["targets"][str(target_id)] = {
                "native_correct": native_match is not None,
                "ours_correct": True,
                "ours_native_id": int(ours_match["sam3_obj_id"]),
            }
        writer.write(compose(
            draw_panel(source, native_rendered),
            draw_panel(source, ours_rendered),
        ))
        audit.append(frame_audit)
    writer.release()

    (OUTPUT_DIR / "experiment2_motion_blur_ours_vs_native_id_8s_audit.json").write_text(
        json.dumps({
            "event_id": EVENT_ID,
            "prompt": "truck",
            "synthetic_motion_blur": {"kernel_px": 31, "angle_deg": 12},
            "duration_seconds": len(FRAMES) / FPS,
            "crop": CROP,
            "frames": audit,
        }, indent=2), encoding="utf-8"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
