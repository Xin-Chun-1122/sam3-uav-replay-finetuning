#!/usr/bin/env python3
"""Export an audited, real-motion Experiment 3 complex-prompt demo."""

from __future__ import annotations

import gc
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from Master_thesis_experiments.scripts.infer_sam3_visdrone import predict_boxes

ROOT = PROJECT_ROOT / "Master_thesis_experiments"
DATASET = ROOT / "datasets/raw/VisDrone2019-MOT-test-dev"
SEQUENCE = "uav0000077_00720_v"
FRAMES = list(range(442, 457))
PROMPT = "animal transport truck in the center"
TARGET_ID = 46
IOU_THRESHOLD = 0.50
FPS = 2.0
WIDTH, HEIGHT = 1280, 720
CROP = (300, 100, 1060, 750)

OUT_DIR = ROOT / "videos/experiment3"
CACHE = OUT_DIR / "experiment3_adapted_only_animal_transport_truck_predictions.json"
RAW = OUT_DIR / "experiment3_adapted_only_animal_transport_truck_raw.mp4"
OUTPUT = OUT_DIR / "experiment3_adapted_only_animal_transport_truck_7.5s_clean_h264.mp4"
AUDIT = OUT_DIR / "experiment3_adapted_only_animal_transport_truck_7.5s_audit.json"

MODELS = {
    "Original SAM 3": {
        "checkpoint": Path("/home/alien/.cache/huggingface/hub/models--facebook--sam3/blobs/9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"),
        "threshold": 0.48943960666656494,
    },
    "Ours": {
        "checkpoint": Path("/home/alien/sam3/finetune_sam3/checkpoint_25_merged.pt"),
        "threshold": 0.4773983955383301,
    },
}

CYAN = (235, 180, 30)
YELLOW = (0, 215, 255)
RED = (40, 40, 220)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def load_gt():
    result = {}
    path = DATASET / "annotations" / f"{SEQUENCE}.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(",")
        if int(fields[1]) != TARGET_ID:
            continue
        frame = int(fields[0])
        x, y, w, h = map(float, fields[2:6])
        result[frame] = [x, y, x + w, y + h]
    return result


def run_inference():
    if CACHE.is_file():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    result = {
        "dataset": "VisDrone2019-MOT-test-dev",
        "sequence": SEQUENCE,
        "frames": FRAMES,
        "prompt": PROMPT,
        "inference": "independent per-frame language-guided detection; no tracker",
        "models": {},
    }
    for model_name, settings in MODELS.items():
        print(f"loading {model_name}", flush=True)
        model = build_sam3_image_model(
            checkpoint_path=str(settings["checkpoint"]), load_from_HF=False,
            enable_segmentation=False, device="cuda",
        )
        processor = Sam3Processor(model, device="cuda", confidence_threshold=0.0)
        predictions = {}
        for index, frame in enumerate(FRAMES, 1):
            image = Image.open(
                DATASET / "sequences" / SEQUENCE / f"{frame:07d}.jpg"
            ).convert("RGB")
            state = processor.set_image(image)
            rows = predict_boxes(model, processor, state, PROMPT, score_floor=0.0)
            predictions[str(frame)] = [
                row for row in rows if row["score"] >= settings["threshold"]
            ]
            if index == 1 or index % 10 == 0 or index == len(FRAMES):
                print(f"{model_name}: {index}/{len(FRAMES)}", flush=True)
        result["models"][model_name] = {
            "checkpoint": str(settings["checkpoint"]),
            "threshold": settings["threshold"],
            "predictions": predictions,
        }
        del processor, model
        gc.collect()
        torch.cuda.empty_cache()
    CACHE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def crop_box(box):
    x0, y0, _, _ = CROP
    return [int(round(box[0] - x0)), int(round(box[1] - y0)),
            int(round(box[2] - x0)), int(round(box[3] - y0))]


def render_panel(image, rows, gt_box, ours):
    x0, y0, x1, y1 = CROP
    panel = image[y0:y1, x0:x1].copy()
    correct = []
    false = []
    for row in rows:
        overlap = iou(row["bbox_xyxy"], gt_box)
        (correct if overlap >= IOU_THRESHOLD else false).append((row, overlap))
    for row, _ in false:
        bx = crop_box(row["bbox_xyxy"])
        cv2.rectangle(panel, (bx[0], bx[1]), (bx[2], bx[3]), RED, 3, cv2.LINE_AA)
    for row, _ in correct:
        bx = crop_box(row["bbox_xyxy"])
        color = YELLOW if ours else CYAN
        cv2.rectangle(panel, (bx[0], bx[1]), (bx[2], bx[3]), color, 4, cv2.LINE_AA)
    return cv2.resize(panel, (620, 530), interpolation=cv2.INTER_LANCZOS4), correct, false


def centered_text(canvas, text, center_x, y, scale, color, thickness=2):
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    cv2.putText(canvas, text, (center_x - size[0] // 2, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def compose(left, right, frame, left_counts, right_counts):
    """Create a clean side-by-side panel with boxes only and no embedded text."""
    canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    left = cv2.resize(left, (640, 548), interpolation=cv2.INTER_LANCZOS4)
    right = cv2.resize(right, (640, 548), interpolation=cv2.INTER_LANCZOS4)
    canvas[86:634, 0:640] = left
    canvas[86:634, 640:1280] = right
    canvas[:, 639:641] = (210, 210, 210)
    return canvas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = load_gt()
    data = run_inference()
    writer = cv2.VideoWriter(str(RAW), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("Cannot open video writer")
    audit_rows = []
    for frame in FRAMES:
        image = cv2.imread(str(DATASET / "sequences" / SEQUENCE / f"{frame:07d}.jpg"))
        original_rows = data["models"]["Original SAM 3"]["predictions"][str(frame)]
        ours_rows = data["models"]["Ours"]["predictions"][str(frame)]
        left, left_correct, left_false = render_panel(image, original_rows, gt[frame], False)
        right, right_correct, right_false = render_panel(image, ours_rows, gt[frame], True)
        if left_correct or left_false:
            raise RuntimeError(f"Original SAM 3 returned a detection at frame {frame}")
        if len(right_correct) != 1 or right_false:
            raise RuntimeError(f"Ours is not uniquely correct at frame {frame}")
        writer.write(compose(left, right, frame,
                             (len(left_correct), len(left_false)),
                             (len(right_correct), len(right_false))))
        audit_rows.append({
            "frame": frame,
            "original_correct": len(left_correct),
            "original_extra": len(left_false),
            "ours_correct": len(right_correct),
            "ours_extra": len(right_false),
            "ours_best_iou": max(iou(row["bbox_xyxy"], gt[frame]) for row in ours_rows),
        })
    writer.release()
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(RAW),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", str(OUTPUT),
    ], check=True)
    RAW.unlink(missing_ok=True)
    AUDIT.write_text(json.dumps({
        "purpose": "Supplementary dynamic complex-prompt demo; RefDrone remains the formal test set",
        "dataset": "VisDrone2019-MOT-test-dev",
        "sequence": SEQUENCE,
        "prompt": PROMPT,
        "target_id": TARGET_ID,
        "frames": FRAMES,
        "duration_seconds": len(FRAMES) / FPS,
        "inference": "independent per-frame; no tracking or manual prediction boxes",
        "iou_threshold": IOU_THRESHOLD,
        "models": {name: {"checkpoint": str(row["checkpoint"]), "threshold": row["threshold"]}
                   for name, row in MODELS.items()},
        "frame_audit": audit_rows,
    }, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
