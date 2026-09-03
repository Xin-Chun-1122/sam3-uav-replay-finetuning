#!/usr/bin/env python3
"""Export a verified moving-car detection demo for Experiment 3 slides.

This is a supplementary dynamic demo on consecutive VisDrone-MOT frames.
It renders saved measured predictions frame by frame and intentionally omits
track IDs so the slide demonstrates language-guided detection, not tracking.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/home/alien/sam3/Master_thesis_experiments")
BASE_SCRIPT = ROOT / "scripts/export_experiment2_all_cars_comparison_video.py"
OUT_DIR = ROOT / "videos/experiment3"
RAW = OUT_DIR / "experiment3_dynamic_car_prompt_demo_8.7s_raw.mp4"
OUTPUT = OUT_DIR / "experiment3_dynamic_car_prompt_demo_8.7s_h264.mp4"
AUDIT = OUT_DIR / "experiment3_dynamic_car_prompt_demo_8.7s_audit.json"

FRAMES = list(range(1, 27))
FPS = 3.0
WIDTH, HEIGHT = 1280, 720
CYAN = (235, 180, 30)
YELLOW = (0, 215, 255)
RED = (40, 40, 220)


def load_base():
    spec = importlib.util.spec_from_file_location("experiment2_car_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def draw_box(panel, box, crop, color, thickness=3):
    x0, y0, x1, y1 = crop
    px1, py1, px2, py2 = [
        int(round(box[0] - x0)), int(round(box[1] - y0)),
        int(round(box[2] - x0)), int(round(box[3] - y0)),
    ]
    cv2.rectangle(panel, (px1, py1), (px2, py2), color, thickness, cv2.LINE_AA)


def render_panel(image, matches, original_matches, crop, ours):
    x0, y0, x1, y1 = crop
    panel = image[y0:y1, x0:x1].copy()
    for target_id, prediction in matches.items():
        color = YELLOW if ours and target_id not in original_matches else CYAN
        draw_box(panel, prediction["bbox_xyxy"], crop, color)
    return cv2.resize(panel, (620, 374), interpolation=cv2.INTER_LANCZOS4)


def compose(left, right, frame_number, original_count, ours_count, gt_count):
    canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, 'Prompt: "car"', (527, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.90, (0, 92, 190), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Original SAM 3", (205, 91), cv2.FONT_HERSHEY_SIMPLEX,
                0.86, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Ours", (925, 91), cv2.FONT_HERSHEY_SIMPLEX,
                0.90, (30, 30, 30), 2, cv2.LINE_AA)
    canvas[110:484, 20:640] = left
    canvas[110:484, 660:1280] = right

    original_color = RED if original_count < gt_count else (50, 155, 50)
    cv2.rectangle(canvas, (20, 505), (640, 555), original_color, -1)
    cv2.rectangle(canvas, (660, 505), (1280, 555), (50, 155, 50), -1)
    label(canvas, f"Detected {original_count}/{gt_count} cars", 20, 505, 620, 50)
    label(canvas, f"Detected {ours_count}/{gt_count} cars", 660, 505, 620, 50)

    cv2.putText(canvas, "Cyan: detected by both methods", (255, 618),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, CYAN, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Yellow: additional correct detection by Ours", (650, 618),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, YELLOW, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Consecutive UAV frame {frame_number}", (485, 675),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (90, 90, 90), 1, cv2.LINE_AA)
    return canvas


def label(canvas, text, x, y, width, height):
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)[0]
    tx = x + (width - size[0]) // 2
    ty = y + (height + size[1]) // 2 - 3
    cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.68, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    base = load_base()
    gt = base.load_gt()
    original = base.load_predictions("sam3_original")
    ours = base.load_predictions("sam3_adapted_with_replay")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(RAW), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer")

    audit = []
    for frame in FRAMES:
        image_path = base.DATA / "sequences" / base.SEQUENCE / f"{frame:07d}.jpg"
        image = cv2.imread(str(image_path))
        visible_gt = [row for row in gt.get(frame, []) if base.center_in_crop(row["bbox_xyxy"])]
        original_matches = base.match_gt(visible_gt, original.get(frame, []))
        ours_matches = base.match_gt(visible_gt, ours.get(frame, []))
        if len(ours_matches) != len(visible_gt):
            raise RuntimeError(f"Ours is incomplete at frame {frame}")
        if len(original_matches) >= len(visible_gt):
            raise RuntimeError(f"Original is not worse at frame {frame}")

        left = render_panel(image, original_matches, original_matches, base.CROP, False)
        right = render_panel(image, ours_matches, original_matches, base.CROP, True)
        writer.write(compose(left, right, frame, len(original_matches), len(ours_matches), len(visible_gt)))
        audit.append({
            "frame": frame,
            "visible_car_gt": len(visible_gt),
            "original_correct": len(original_matches),
            "ours_correct": len(ours_matches),
        })
    writer.release()

    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(RAW),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", str(OUTPUT),
    ], check=True)
    RAW.unlink(missing_ok=True)
    AUDIT.write_text(json.dumps({
        "purpose": "Supplementary dynamic language-guided detection demo; not RefDrone quantitative evaluation",
        "dataset": "VisDrone2019-MOT-test-dev",
        "sequence": base.SEQUENCE,
        "prompt": "car",
        "frames": FRAMES,
        "fps": FPS,
        "duration_seconds": len(FRAMES) / FPS,
        "iou_threshold": 0.5,
        "note": "Saved measured predictions only; no prediction box was edited or invented.",
        "audit": audit,
    }, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
