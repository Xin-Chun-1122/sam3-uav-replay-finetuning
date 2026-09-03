"""4-column case visualization: Input | GT | Original | Fine-tuned."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# BGR colors
COLORS = {
    "fire":   (0,   80, 255),
    "smoke":  (180, 180, 180),
    "car":    (0,  200, 255),
    "person": (50,  220,  50),
    "default": (200, 100,   0),
}

STATUS_COLORS = {
    "TP": (0, 200, 0),
    "FP": (0, 0, 255),
    "FN": (0, 140, 255),
    "GT": (255, 200, 0),
}


def _class_color(label: str) -> Tuple[int, int, int]:
    return COLORS.get(label.lower(), COLORS["default"])


def _draw_box(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y = max(y1 - 4, th + 4)
    cv2.rectangle(img, (x1, label_y - th - 6), (x1 + tw + 6, label_y + 2), color, -1)
    cv2.putText(img, label, (x1 + 3, label_y - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_gt_panel(
    frame: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: List[str],
    title: str = "Ground Truth",
) -> np.ndarray:
    panel = frame.copy()
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = STATUS_COLORS["GT"]
        _draw_box(panel, x1, y1, x2, y2, color, label)
    _add_header(panel, title, (220, 180, 0))
    return panel


def draw_prediction_panel(
    frame: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: List[str],
    pred_to_gt: Dict[int, int],
    gt_boxes: np.ndarray,
    gt_labels: List[str],
    matched_gt_indices: List[int],
    title: str,
) -> np.ndarray:
    panel = frame.copy()

    # Draw FN (unmatched GT)
    for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
        if gt_idx not in matched_gt_indices:
            x1, y1, x2, y2 = [int(v) for v in gt_box]
            _draw_box(panel, x1, y1, x2, y2, STATUS_COLORS["FN"], f"FN:{gt_label}", thickness=2)

    # Draw TP and FP predictions
    for pred_idx, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
        x1, y1, x2, y2 = [int(v) for v in box]
        gt_match = pred_to_gt.get(pred_idx, -1)
        if gt_match >= 0:
            status = "TP"
            color = STATUS_COLORS["TP"]
        else:
            status = "FP"
            color = STATUS_COLORS["FP"]
        lbl = f"{status}:{label} {score:.2f}"
        _draw_box(panel, x1, y1, x2, y2, color, lbl)

    _add_header(panel, title, (50, 220, 50))
    return panel


def _add_header(img: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
    h = 32
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    cv2.putText(img, text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def make_4col_figure(
    image_path: str,
    gt_boxes: np.ndarray,
    gt_labels: List[str],
    orig_boxes: np.ndarray,
    orig_scores: np.ndarray,
    orig_labels: List[str],
    orig_pred_to_gt: Dict[int, int],
    orig_matched_gt: List[int],
    ft_boxes: np.ndarray,
    ft_scores: np.ndarray,
    ft_labels: List[str],
    ft_pred_to_gt: Dict[int, int],
    ft_matched_gt: List[int],
    target_height: int = 480,
) -> Optional[np.ndarray]:
    """Create 4-column comparison figure."""
    frame = cv2.imread(image_path)
    if frame is None:
        logger.warning("Cannot read image: %s", image_path)
        return None

    h, w = frame.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    new_h = target_height

    frame = cv2.resize(frame, (new_w, new_h))

    # Scale boxes
    def scale_boxes(boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return boxes
        b = boxes.copy().astype(np.float32)
        b[:, [0, 2]] *= scale  # x coords
        b[:, [1, 3]] *= scale  # y coords
        return b

    gt_boxes_s = scale_boxes(gt_boxes)
    orig_boxes_s = scale_boxes(orig_boxes)
    ft_boxes_s = scale_boxes(ft_boxes)

    col0 = frame.copy()
    _add_header(col0, "Input Image", (200, 200, 200))

    col1 = draw_gt_panel(frame, gt_boxes_s, gt_labels, "Ground Truth")

    col2 = draw_prediction_panel(
        frame, orig_boxes_s, orig_scores, orig_labels,
        orig_pred_to_gt, gt_boxes_s, gt_labels, orig_matched_gt,
        "Original SAM 3",
    )

    col3 = draw_prediction_panel(
        frame, ft_boxes_s, ft_scores, ft_labels,
        ft_pred_to_gt, gt_boxes_s, gt_labels, ft_matched_gt,
        "Fine-tuned SAM 3",
    )

    figure = np.concatenate([col0, col1, col2, col3], axis=1)
    return figure


def save_figure(figure: np.ndarray, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, figure)
