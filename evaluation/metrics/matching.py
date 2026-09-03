"""IoU-based object matching and TP/FP/FN computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two xyxy boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return float(inter / union)


def compute_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between predicted and GT boxes.

    Args:
        pred_boxes: (M, 4) xyxy
        gt_boxes:   (N, 4) xyxy

    Returns:
        (M, N) IoU matrix
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)

    # Broadcast
    pb = pred_boxes[:, None, :]  # (M, 1, 4)
    gb = gt_boxes[None, :, :]    # (1, N, 4)

    x1 = np.maximum(pb[:, :, 0], gb[:, :, 0])
    y1 = np.maximum(pb[:, :, 1], gb[:, :, 1])
    x2 = np.minimum(pb[:, :, 2], gb[:, :, 2])
    y2 = np.minimum(pb[:, :, 3], gb[:, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_g = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    area_p = np.maximum(0.0, area_p)
    area_g = np.maximum(0.0, area_g)

    union = area_p[:, None] + area_g[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


@dataclass
class MatchResult:
    """Per-image matching result for one class."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    # Indices of matched GT boxes
    matched_gt_indices: List[int] = field(default_factory=list)
    # Indices of matched pred boxes
    matched_pred_indices: List[int] = field(default_factory=list)
    # Index → matched GT index (or -1)
    pred_to_gt: Dict[int, int] = field(default_factory=dict)
    # GT index → matched pred index (or -1)
    gt_to_pred: Dict[int, int] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        denom = self.tp + self.fn
        return self.fn / denom if denom > 0 else 0.0

    @property
    def object_success_rate(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0


def match_detections(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: List[str],
    gt_boxes: np.ndarray,
    gt_labels: List[str],
    iou_threshold: float = 0.50,
) -> MatchResult:
    """
    One-to-one greedy matching between predictions and GT.

    Rules:
    - Classes must match.
    - IoU >= iou_threshold required.
    - Predictions are matched in descending confidence order.
    - Each GT can only be matched once.
    - Each prediction can only be matched once.

    Returns MatchResult with TP, FP, FN counts and index maps.
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    result = MatchResult()
    result.gt_to_pred = {i: -1 for i in range(n_gt)}
    result.pred_to_gt = {i: -1 for i in range(n_pred)}

    if n_gt == 0 and n_pred == 0:
        return result

    if n_gt == 0:
        result.fp = n_pred
        return result

    if n_pred == 0:
        result.fn = n_gt
        return result

    # Sort predictions by confidence descending
    order = np.argsort(pred_scores)[::-1]

    matched_gt = set()
    matched_pred = set()

    for pred_idx in order:
        pred_cls = pred_labels[pred_idx]
        pred_box = pred_boxes[pred_idx]

        best_iou = -1.0
        best_gt_idx = -1

        for gt_idx in range(n_gt):
            if gt_idx in matched_gt:
                continue
            if gt_labels[gt_idx] != pred_cls:
                continue

            iou = compute_iou(pred_box, gt_boxes[gt_idx])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            # True positive
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)
            result.pred_to_gt[pred_idx] = best_gt_idx
            result.gt_to_pred[best_gt_idx] = pred_idx
            result.tp += 1
            result.matched_gt_indices.append(best_gt_idx)
            result.matched_pred_indices.append(pred_idx)
        else:
            # False positive
            result.fp += 1

    # False negatives: unmatched GT
    result.fn = n_gt - len(matched_gt)

    return result


@dataclass
class ImageStatus:
    """Image-level success/failure classification."""

    status: str   # 'complete_success', 'partial_failure', 'complete_failure', 'false_alarm'
    has_gt: bool
    tp: int
    fp: int
    fn: int
    n_pred: int
    n_gt: int


def classify_image(
    match_result: MatchResult,
    n_gt: int,
    n_pred: int,
) -> ImageStatus:
    """
    Classify an image into one of four categories:

    complete_success:
        - If image has GT: FN=0 AND FP=0
        - If image has no GT: n_pred=0 (successful negative)

    partial_failure:
        - TP > 0 but also FN > 0 or FP > 0

    complete_failure:
        - Image has GT but TP=0

    false_alarm:
        - Image has no GT but n_pred > 0
    """
    tp = match_result.tp
    fp = match_result.fp
    fn = match_result.fn
    has_gt = n_gt > 0

    if has_gt:
        if fn == 0 and fp == 0:
            status = "complete_success"
        elif tp > 0:
            status = "partial_failure"
        else:
            status = "complete_failure"
    else:
        if n_pred == 0:
            status = "complete_success"
        else:
            status = "false_alarm"

    return ImageStatus(
        status=status,
        has_gt=has_gt,
        tp=tp,
        fp=fp,
        fn=fn,
        n_pred=n_pred,
        n_gt=n_gt,
    )


@dataclass
class ComparisonResult:
    """Per-image fine-tuning comparison."""

    image_name: str
    original_status: str
    finetuned_status: str
    comparison: str  # 'both_success', 'improved_by_finetuning', 'both_failed', 'finetuning_regression'

    original_tp: int
    original_fp: int
    original_fn: int
    finetuned_tp: int
    finetuned_fp: int
    finetuned_fn: int
    n_gt: int


def classify_comparison(
    image_name: str,
    orig_status: ImageStatus,
    ft_status: ImageStatus,
) -> ComparisonResult:
    """Compare Original vs Fine-tuned outcome for one image."""
    os_ = orig_status.status
    fs = ft_status.status

    orig_success = os_ == "complete_success"
    ft_success = fs == "complete_success"
    orig_failed = os_ in ("complete_failure", "false_alarm")
    ft_failed = fs in ("complete_failure", "false_alarm")

    if orig_success and ft_success:
        comparison = "both_success"
    elif (not orig_success) and ft_success:
        comparison = "improved_by_finetuning"
    elif orig_failed and ft_failed:
        comparison = "both_failed"
    elif orig_success and (not ft_success):
        comparison = "finetuning_regression"
    else:
        # Both partial failure or mixed
        comparison = "both_failed"

    return ComparisonResult(
        image_name=image_name,
        original_status=os_,
        finetuned_status=fs,
        comparison=comparison,
        original_tp=orig_status.tp,
        original_fp=orig_status.fp,
        original_fn=orig_status.fn,
        finetuned_tp=ft_status.tp,
        finetuned_fp=ft_status.fp,
        finetuned_fn=ft_status.fn,
        n_gt=orig_status.n_gt,
    )
