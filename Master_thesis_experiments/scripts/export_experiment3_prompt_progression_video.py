#!/usr/bin/env python3
"""Export a short PPT demo from measured Experiment 3 prediction panels."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PANEL_ROOT = ROOT / "figures/experiment3/ppt_table_panels/progressive_bus"
OUT_DIR = ROOT / "figures/experiment3/ppt_demo_video"
TEMP_MP4 = OUT_DIR / "experiment3_progressive_prompt_demo_temp.mp4"

WIDTH, HEIGHT = 1280, 720
FPS = 12
SECONDS_PER_LEVEL = 3

LEVELS = [
    {
        "layer": "Level 1 - Category",
        "prompt": "bus",
        "original": "7 TP, 1 FP",
        "ours": "7 TP, 0 FP",
    },
    {
        "layer": "Level 2 - Category + attribute",
        "prompt": "green bus",
        "original": "1 TP, 0 FP",
        "ours": "1 TP, 0 FP",
    },
    {
        "layer": "Level 3 - Attribute + spatial description",
        "prompt": "green bus near the bottom-right corner of the image",
        "original": "0 TP, 1 FN",
        "ours": "1 TP, 0 FN",
    },
]


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def put_centered(frame, text, y, scale, color, thickness=2):
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    x = (frame.shape[1] - size[0]) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_level(index: int) -> np.ndarray:
    info = LEVELS[index]
    layer_dir = PANEL_ROOT / f"layer{index + 1}"
    original = cv2.imread(str(layer_dir / "original_sam3.png"))
    ours = cv2.imread(str(layer_dir / "ours.png"))
    if original is None or ours is None:
        raise FileNotFoundError(f"Missing measured panels under {layer_dir}")

    frame = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    put_centered(frame, info["layer"], 42, 0.72, (25, 25, 25), 2)
    put_centered(frame, f'Prompt: "{info["prompt"]}"', 83, 0.70, (0, 92, 190), 2)

    panel_w, panel_h = 600, 400
    left_x, right_x, top_y = 25, 655, 160
    frame[top_y : top_y + panel_h, left_x : left_x + panel_w] = fit_image(original, panel_w, panel_h)
    frame[top_y : top_y + panel_h, right_x : right_x + panel_w] = fit_image(ours, panel_w, panel_h)

    cv2.putText(frame, "Original SAM 3", (185, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(frame, "Ours", (920, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)

    original_bad = index in (0, 2)
    cv2.rectangle(frame, (left_x, 580), (left_x + panel_w, 625), (40, 40, 220) if original_bad else (50, 155, 50), -1)
    cv2.rectangle(frame, (right_x, 580), (right_x + panel_w, 625), (0, 170, 230) if index == 2 else (50, 155, 50), -1)
    put_centered_in_box(frame, info["original"], left_x, 580, panel_w, 45)
    put_centered_in_box(frame, info["ours"], right_x, 580, panel_w, 45)

    legend = "Measured predictions - green dashed: GT | blue: matched prediction | red: false positive"
    put_centered(frame, legend, 670, 0.46, (85, 85, 85), 1)

    # Three-level progress indicator.
    x0, gap = 568, 72
    for i in range(3):
        color = (0, 150, 230) if i == index else (195, 195, 195)
        cv2.circle(frame, (x0 + i * gap, 698), 9, color, -1, cv2.LINE_AA)
    return frame


def put_centered_in_box(frame, text, x, y, width, height):
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)[0]
    tx = x + (width - size[0]) // 2
    ty = y + (height + size[1]) // 2 - 3
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(TEMP_MP4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer")

    frames = [render_level(i) for i in range(3)]
    count = SECONDS_PER_LEVEL * FPS
    fade = 4
    for i, base in enumerate(frames):
        for n in range(count):
            output = base
            if i > 0 and n < fade:
                alpha = (n + 1) / fade
                output = cv2.addWeighted(frames[i - 1], 1.0 - alpha, base, alpha, 0)
            writer.write(output)
    writer.release()
    print(TEMP_MP4)


if __name__ == "__main__":
    main()
