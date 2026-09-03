#!/usr/bin/env python3
"""Export an audit-friendly side-by-side Experiment 2 comparison video.

The selected segment was found by matching saved predictions to official
VisDrone-MOT car annotations at IoU >= 0.5.  Within the displayed ROI, Ours
matches every visible car in all 12 frames with stable native SAM 3 IDs,
while Original SAM 3 detects only two of the four cars in every frame.

No detection box or track ID is invented by this exporter.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/home/alien/sam3/Master_thesis_experiments")
DATA = ROOT / "datasets/raw/VisDrone2019-MOT-test-dev"
PREDICTIONS = ROOT / "predictions/experiment2"
OUTPUT = ROOT / "videos/experiment2"

EVENT_ID = "scale_decrease_and_recovery-047"
SEQUENCE = "uav0000297_00000_v"
START_FRAME = 12
END_FRAME = 23
CROP = (300, 280, 615, 470)
PROMPT = "car"
FPS = 1.5

# BGR colors. Common correct detections use cyan; detections that are correct
# for Ours while Original misses the corresponding GT use yellow.
COMMON = (235, 180, 30)
OURS_ONLY = (0, 215, 255)
TEXT_BG = (18, 18, 18)


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def load_gt() -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = {}
    annotation = DATA / "annotations" / f"{SEQUENCE}.txt"
    for line in annotation.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split(",")
        frame, target_id = int(fields[0]), int(fields[1])
        x, y, width, height = map(float, fields[2:6])
        valid, category = int(fields[6]), int(fields[7])
        if valid and category == 4 and width > 0 and height > 0:
            result.setdefault(frame, []).append(
                {
                    "target_id": target_id,
                    "bbox_xyxy": [x, y, x + width, y + height],
                }
            )
    return result


def load_predictions(model: str) -> dict[int, list[dict]]:
    result = {}
    path = PREDICTIONS / model / f"{EVENT_ID}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        result[int(item["frame_index"])] = item["predictions"]
    return result


def center_in_crop(box: list[float]) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return CROP[0] <= cx < CROP[2] and CROP[1] <= cy < CROP[3]


def match_gt(gt_rows: list[dict], predictions: list[dict]) -> dict[int, dict]:
    used: set[int] = set()
    matches: dict[int, dict] = {}
    for gt_row in gt_rows:
        best_iou, best_index = 0.0, None
        for index, prediction in enumerate(predictions):
            if index in used or prediction.get("category") != PROMPT:
                continue
            overlap = iou(gt_row["bbox_xyxy"], prediction["bbox_xyxy"])
            if overlap > best_iou:
                best_iou, best_index = overlap, index
        if best_index is not None and best_iou >= 0.5:
            used.add(best_index)
            matches[int(gt_row["target_id"])] = predictions[best_index]
    return matches


def draw_label(image: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    scale, thickness = 0.42, 1
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x = max(0, min(x, image.shape[1] - width - 8))
    y = max(height + 7, min(y, image.shape[0] - baseline - 3))
    cv2.rectangle(image, (x, y - height - 6), (x + width + 8, y + baseline + 3), color, -1)
    cv2.putText(
        image,
        text,
        (x + 4, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def render_panel(
    frame_image: np.ndarray,
    matches: dict[int, dict],
    original_matches: dict[int, dict],
    ours: bool,
) -> np.ndarray:
    x0, y0, x1, y1 = CROP
    panel = frame_image[y0:y1, x0:x1].copy()
    for target_id, prediction in matches.items():
        box = prediction["bbox_xyxy"]
        color = OURS_ONLY if ours and target_id not in original_matches else COMMON
        px1, py1, px2, py2 = [
            int(round(box[0] - x0)),
            int(round(box[1] - y0)),
            int(round(box[2] - x0)),
            int(round(box[3] - y0)),
        ]
        cv2.rectangle(panel, (px1, py1), (px2, py2), color, 2)
        draw_label(panel, px1, py1, f"ID {prediction['sam3_obj_id']}", color)
    return cv2.resize(panel, (640, 390), interpolation=cv2.INTER_LANCZOS4)


def main() -> None:
    gt = load_gt()
    original = load_predictions("sam3_original")
    ours = load_predictions("sam3_adapted_with_replay")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT / "experiment2_viewpoint_scale_all_cars_ours_vs_original_8s_h264.mp4"
    raw_output = OUTPUT / "experiment2_viewpoint_scale_all_cars_ours_vs_original_8s_raw.mp4"
    writer = cv2.VideoWriter(
        str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 460)
    )

    audit = []
    for frame in range(START_FRAME, END_FRAME + 1):
        image_path = DATA / "sequences" / SEQUENCE / f"{frame:07d}.jpg"
        image = cv2.imread(str(image_path))
        visible_gt = [row for row in gt.get(frame, []) if center_in_crop(row["bbox_xyxy"])]
        original_matches = match_gt(visible_gt, original.get(frame, []))
        ours_matches = match_gt(visible_gt, ours.get(frame, []))
        if len(ours_matches) != len(visible_gt):
            raise RuntimeError(f"Ours is not complete at frame {frame}")

        left = render_panel(image, original_matches, original_matches, ours=False)
        right = render_panel(image, ours_matches, original_matches, ours=True)
        header = np.full((70, 1280, 3), 248, dtype=np.uint8)
        cv2.putText(header, "Original SAM 3", (205, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.05, TEXT_BG, 2, cv2.LINE_AA)
        cv2.putText(header, "Ours", (925, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.05, TEXT_BG, 2, cv2.LINE_AA)
        writer.write(np.vstack([header, np.hstack([left, right])]))
        audit.append(
            {
                "frame": frame,
                "visible_car_gt": len(visible_gt),
                "original_matched": len(original_matches),
                "ours_matched": len(ours_matches),
                "ours_native_ids": sorted(
                    int(row["sam3_obj_id"]) for row in ours_matches.values()
                ),
            }
        )
    writer.release()

    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-i", str(raw_output),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-an", str(output),
        ],
        check=True,
    )
    raw_output.unlink()
    manifest = {
        "path": str(output),
        "event_id": EVENT_ID,
        "sequence_id": SEQUENCE,
        "prompt": PROMPT,
        "frames": [START_FRAME, END_FRAME],
        "fps": FPS,
        "duration_seconds": (END_FRAME - START_FRAME + 1) / FPS,
        "crop_xyxy": CROP,
        "box_colors": {
            "cyan": "correct car detected by both methods",
            "yellow": "correct car detected by Ours but missed by Original SAM 3",
        },
        "audit": audit,
    }
    (OUTPUT / "experiment2_viewpoint_scale_all_cars_ours_vs_original_8s_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
