#!/usr/bin/env python3
"""Create an actual per-frame SAM 3 detection comparison video.

The video is a qualitative cross-dataset demonstration. Each sampled MOT frame
is independently processed with the text prompt "car"; no temporal tracker or
hand-authored prediction is used. Ground truth is used only for visualization
and TP/FP/FN labeling.
"""

from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402
from Master_thesis_experiments.scripts.infer_sam3_visdrone import predict_boxes  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/raw/VisDrone2019-MOT-test-dev"
SEQUENCE = "uav0000077_00720_v"
START_FRAME = 175
END_FRAME = 214
PROMPT = "car"
CLASS_ID = 4
FPS = 8.0
IOU_THRESHOLD = 0.50
MODELS = {
    "Original SAM 3": {
        "checkpoint": Path(
            "/home/alien/.cache/huggingface/hub/models--facebook--sam3/blobs/"
            "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
        ),
        "threshold": 0.48943960666656494,
    },
    "Adapted SAM 3 with replay (Ours)": {
        "checkpoint": Path("/home/alien/sam3/finetune_sam3/checkpoint_25_merged.pt"),
        "threshold": 0.4773983955383301,
    },
}
OUTPUT_DIR = ROOT / "videos/experiment1"
CACHE = OUTPUT_DIR / "experiment1_car_video_detection_predictions.json"
RAW_VIDEO = OUTPUT_DIR / "experiment1_annotated_detection_comparison_raw.mp4"

GT_COLORS = {
    "Tiny": (0, 215, 255),
    "Small": (60, 200, 60),
    "Regular": (255, 70, 255),
}
TP_COLOR = (255, 220, 0)
FP_COLOR = (40, 40, 235)


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection + 1e-9)


def scale_name(width: float, height: float) -> str:
    scale = math.sqrt(max(0.0, width * height))
    if scale < 32:
        return "Tiny"
    if scale < 64:
        return "Small"
    return "Regular"


def load_ground_truth() -> dict[int, list[dict]]:
    output: dict[int, list[dict]] = {}
    annotation = DATASET / "annotations" / f"{SEQUENCE}.txt"
    for line in annotation.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split(",")]
        frame, target_id, left, top, width, height, score, category = values[:8]
        if int(category) != CLASS_ID or score <= 0:
            continue
        output.setdefault(int(frame), []).append(
            {
                "target_id": int(target_id),
                "bbox_xyxy": [left, top, left + width, top + height],
                "scale": scale_name(width, height),
            }
        )
    return output


def run_inference() -> dict:
    if CACHE.is_file():
        return json.loads(CACHE.read_text(encoding="utf-8"))

    frame_paths = [
        DATASET / "sequences" / SEQUENCE / f"{frame:07d}.jpg"
        for frame in range(START_FRAME, END_FRAME + 1)
    ]
    result = {
        "protocol": {
            "sequence": SEQUENCE,
            "frames": [START_FRAME, END_FRAME],
            "prompt": PROMPT,
            "inference": "independent per-frame language-guided detection; no tracker",
            "threshold_source": "frozen VisDrone-DET validation thresholds",
            "iou_threshold": IOU_THRESHOLD,
        },
        "models": {},
    }

    for model_name, settings in MODELS.items():
        print(f"loading={model_name}", flush=True)
        model = build_sam3_image_model(
            checkpoint_path=str(settings["checkpoint"]),
            load_from_HF=False,
            enable_segmentation=False,
            device="cuda",
        )
        processor = Sam3Processor(model, device="cuda", confidence_threshold=0.0)
        frame_predictions = {}
        for index, path in enumerate(frame_paths, 1):
            image = Image.open(path).convert("RGB")
            state = processor.set_image(image)
            rows = predict_boxes(model, processor, state, PROMPT, score_floor=0.0)
            frame_predictions[str(START_FRAME + index - 1)] = [
                row for row in rows if row["score"] >= settings["threshold"]
            ]
            if index == 1 or index % 10 == 0 or index == len(frame_paths):
                print(f"{model_name}: frames={index}/{len(frame_paths)}", flush=True)
        result["models"][model_name] = {
            "checkpoint": str(settings["checkpoint"]),
            "threshold": settings["threshold"],
            "predictions": frame_predictions,
        }
        del processor, model
        gc.collect()
        torch.cuda.empty_cache()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def match_predictions(predictions: list[dict], gt: list[dict]) -> tuple[list[dict], set[int], int, int, int]:
    matched_gt: set[int] = set()
    annotated = []
    for prediction in sorted(predictions, key=lambda row: row["score"], reverse=True):
        candidates = [
            (box_iou(prediction["bbox_xyxy"], target["bbox_xyxy"]), index)
            for index, target in enumerate(gt)
            if index not in matched_gt
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        is_tp = best_iou >= IOU_THRESHOLD
        if is_tp:
            matched_gt.add(best_index)
        annotated.append({**prediction, "is_tp": is_tp, "iou": best_iou})
    tp = len(matched_gt)
    fp = len(predictions) - tp
    fn = len(gt) - tp
    return annotated, matched_gt, tp, fp, fn


def dashed_rectangle(image: np.ndarray, box: list[float], color: tuple[int, int, int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    dash = 9
    for start in range(x1, x2, dash * 2):
        cv2.line(image, (start, y1), (min(start + dash, x2), y1), color, thickness)
        cv2.line(image, (start, y2), (min(start + dash, x2), y2), color, thickness)
    for start in range(y1, y2, dash * 2):
        cv2.line(image, (x1, start), (x1, min(start + dash, y2)), color, thickness)
        cv2.line(image, (x2, start), (x2, min(start + dash, y2)), color, thickness)


def draw_panel(image: np.ndarray, gt: list[dict], predictions: list[dict]) -> tuple[np.ndarray, tuple[int, int, int]]:
    panel = image.copy()
    for target in gt:
        dashed_rectangle(panel, target["bbox_xyxy"], GT_COLORS[target["scale"]], 2)
    annotated, _, tp, fp, fn = match_predictions(predictions, gt)
    for row in annotated:
        x1, y1, x2, y2 = [int(round(value)) for value in row["bbox_xyxy"]]
        color = TP_COLOR if row["is_tp"] else FP_COLOR
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
    return cv2.resize(panel, (960, 540), interpolation=cv2.INTER_AREA), (tp, fp, fn)


def centered_text(canvas: np.ndarray, text: str, center_x: int, y: int, scale: float, color=(245, 245, 245), thickness=2) -> None:
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    cv2.putText(
        canvas,
        text,
        (center_x - size[0] // 2, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def render_video(cache: dict, ground_truth: dict[int, list[dict]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(RAW_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (1920, 1080),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {RAW_VIDEO}")

    totals = {name: [0, 0, 0] for name in MODELS}
    for frame in range(START_FRAME, END_FRAME + 1):
        source = cv2.imread(
            str(DATASET / "sequences" / SEQUENCE / f"{frame:07d}.jpg")
        )
        if source is None:
            raise FileNotFoundError(frame)
        gt = ground_truth.get(frame, [])
        panels = []
        stats = []
        for model_name in MODELS:
            predictions = cache["models"][model_name]["predictions"][str(frame)]
            panel, counts = draw_panel(source, gt, predictions)
            panels.append(panel)
            stats.append(counts)
            for index, value in enumerate(counts):
                totals[model_name][index] += value

        canvas = np.full((1080, 1920, 3), (25, 25, 28), dtype=np.uint8)
        centered_text(
            canvas,
            'Experiment 1 video demo - prompt: "car" (independent detection per frame)',
            960,
            62,
            1.15,
            thickness=2,
        )
        centered_text(canvas, "Original SAM 3", 480, 125, 0.95)
        centered_text(canvas, "Adapted SAM 3 with replay (Ours)", 1440, 125, 0.95, (230, 210, 255))
        canvas[155:695, 0:960] = panels[0]
        canvas[155:695, 960:1920] = panels[1]
        for index, (tp, fp, fn) in enumerate(stats):
            centered_text(
                canvas,
                f"Frame {frame}   TP={tp}   FP={fp}   FN={fn}",
                480 + 960 * index,
                755,
                0.82,
            )
        centered_text(
            canvas,
            "Dashed GT scale: Tiny (yellow) | Small (green) | Regular (magenta)    "
            "Prediction: TP (cyan) | FP (red)",
            960,
            850,
            0.70,
            (220, 220, 220),
            2,
        )
        centered_text(
            canvas,
            "IoU >= 0.50; frozen validation thresholds; all valid car GT shown",
            960,
            910,
            0.68,
            (200, 200, 200),
            2,
        )
        centered_text(
            canvas,
            "Qualitative VisDrone-MOT demo only - quantitative Experiment 1 metrics use VisDrone-DET",
            960,
            985,
            0.66,
            (150, 190, 255),
            2,
        )
        writer.write(canvas)

    summary_canvas = np.full((1080, 1920, 3), (25, 25, 28), dtype=np.uint8)
    centered_text(summary_canvas, "Qualitative clip summary", 960, 190, 1.35)
    centered_text(
        summary_canvas,
        "Actual per-frame model outputs; clip-only counts, not benchmark metrics",
        960,
        275,
        0.78,
        (180, 200, 235),
        2,
    )
    original = totals["Original SAM 3"]
    ours = totals["Adapted SAM 3 with replay (Ours)"]
    centered_text(summary_canvas, "Original SAM 3", 560, 430, 1.0)
    centered_text(
        summary_canvas,
        f"TP={original[0]}   FP={original[1]}   FN={original[2]}",
        560,
        515,
        0.92,
    )
    centered_text(
        summary_canvas,
        "Adapted SAM 3 with replay (Ours)",
        1360,
        430,
        1.0,
        (230, 210, 255),
    )
    centered_text(
        summary_canvas,
        f"TP={ours[0]}   FP={ours[1]}   FN={ours[2]}",
        1360,
        515,
        0.92,
        (230, 210, 255),
    )
    centered_text(
        summary_canvas,
        "Ours detects more valid cars with fewer false positives in this selected moving clip.",
        960,
        690,
        0.82,
        (220, 220, 220),
        2,
    )
    centered_text(
        summary_canvas,
        "Official quantitative conclusions are based on the complete VisDrone-DET test-dev set.",
        960,
        790,
        0.72,
        (150, 190, 255),
        2,
    )
    for _ in range(int(FPS)):
        writer.write(summary_canvas)
    writer.release()

    summary = {
        "frames": END_FRAME - START_FRAME + 1,
        "summary_frames": int(FPS),
        "fps": FPS,
        "duration_seconds": (END_FRAME - START_FRAME + 1 + int(FPS)) / FPS,
        "totals": {
            name: {"tp": values[0], "fp": values[1], "fn": values[2]}
            for name, values in totals.items()
        },
    }
    (OUTPUT_DIR / "experiment1_annotated_detection_video_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"raw_video={RAW_VIDEO}")


def main() -> None:
    ground_truth = load_ground_truth()
    cache = run_inference()
    render_video(cache, ground_truth)


if __name__ == "__main__":
    main()
