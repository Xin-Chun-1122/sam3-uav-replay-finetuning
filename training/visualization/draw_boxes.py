#!/usr/bin/env python3
"""
Visualization utilities: draw bounding boxes, labels, and tracking IDs on images.
"""

import colorsys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ── Color palette per category ────────────────────────────────
CATEGORY_COLORS = {
    "fire":   (255, 80,  0),    # Orange-red
    "smoke":  (150, 150, 150),  # Gray
    "car":    (0,   180, 255),  # Cyan
    "person": (50,  220, 50),   # Green
    "default":(200, 200, 0),    # Yellow
}


def get_color(label: str) -> Tuple[int, int, int]:
    return CATEGORY_COLORS.get(label.lower(), CATEGORY_COLORS["default"])


def draw_boxes_pil(
    image: "Image.Image",
    boxes: List[List[float]],
    labels: List[str],
    scores: Optional[List[float]] = None,
    track_ids: Optional[List[int]] = None,
    thickness: int = 3,
    font_size: int = 18,
) -> "Image.Image":
    """
    Draw bounding boxes with labels on a PIL Image.
    Returns annotated PIL Image.
    """
    assert PIL_AVAILABLE, "Pillow not installed"
    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = get_color(label)
        score = scores[i] if scores else None
        track_id = track_ids[i] if track_ids else None

        # Draw rectangle
        for t in range(thickness):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)

        # Build label text
        text_parts = [label]
        if score is not None:
            text_parts.append(f"{score:.2f}")
        if track_id is not None:
            text_parts.append(f"#{track_id}")
        text = " ".join(text_parts)

        # Draw label background
        try:
            bbox = font.getbbox(text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = len(text) * 8, 16
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - th - 3), text, fill=(255, 255, 255), font=font)

    return img


def draw_boxes_cv2(
    frame: np.ndarray,
    boxes: List[List[float]],
    labels: List[str],
    scores: Optional[List[float]] = None,
    track_ids: Optional[List[int]] = None,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw bounding boxes on a numpy array (OpenCV BGR format).
    Returns annotated frame.
    """
    assert CV2_AVAILABLE, "OpenCV not installed"
    frame = frame.copy()

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [int(v) for v in box]
        r, g, b = get_color(label)
        color_bgr = (b, g, r)  # OpenCV uses BGR

        score = scores[i] if scores else None
        track_id = track_ids[i] if track_ids else None

        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, thickness)

        text_parts = [label]
        if score is not None:
            text_parts.append(f"{score:.2f}")
        if track_id is not None:
            text_parts.append(f"#{track_id}")
        text = " ".join(text_parts)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    return frame


def save_annotated_image(
    image,
    boxes: List[List[float]],
    labels: List[str],
    output_path: str,
    scores: Optional[List[float]] = None,
    track_ids: Optional[List[int]] = None,
):
    """Save an annotated image to disk."""
    if isinstance(image, np.ndarray):
        annotated = draw_boxes_cv2(image, boxes, labels, scores, track_ids)
        if CV2_AVAILABLE:
            cv2.imwrite(str(output_path), annotated)
    else:
        annotated = draw_boxes_pil(image, boxes, labels, scores, track_ids)
        annotated.save(str(output_path))
    print(f"Saved: {output_path}")
