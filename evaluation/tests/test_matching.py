"""Unit tests for detection matching logic."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.matching import match_detections, classify_image, MatchResult


def _make_box(x1, y1, x2, y2):
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ── match_detections ─────────────────────────────────────────────────────────

def test_perfect_single_detection():
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9])
    labels = ["fire"]
    gt_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    gt_labels = ["fire"]

    r = match_detections(boxes, scores, labels, gt_boxes, gt_labels, iou_threshold=0.5)
    assert r.tp == 1
    assert r.fp == 0
    assert r.fn == 0


def test_no_predictions():
    gt_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    gt_labels = ["fire"]
    r = match_detections(
        np.empty((0, 4), np.float32), np.array([]), [], gt_boxes, gt_labels
    )
    assert r.tp == 0
    assert r.fp == 0
    assert r.fn == 1


def test_no_gt():
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9])
    labels = ["fire"]
    r = match_detections(boxes, scores, labels, np.empty((0, 4), np.float32), [])
    assert r.tp == 0
    assert r.fp == 1
    assert r.fn == 0


def test_no_gt_no_pred():
    r = match_detections(
        np.empty((0, 4), np.float32), np.array([]), [],
        np.empty((0, 4), np.float32), []
    )
    assert r.tp == 0
    assert r.fp == 0
    assert r.fn == 0


def test_class_mismatch_no_match():
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.95])
    labels = ["smoke"]
    gt_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    gt_labels = ["fire"]

    r = match_detections(boxes, scores, labels, gt_boxes, gt_labels)
    assert r.tp == 0
    assert r.fp == 1
    assert r.fn == 1


def test_low_iou_no_match():
    boxes = np.array([[0, 0, 3, 3]], dtype=np.float32)
    scores = np.array([0.9])
    labels = ["fire"]
    # Barely overlapping box
    gt_boxes = np.array([[8, 8, 12, 12]], dtype=np.float32)
    gt_labels = ["fire"]

    r = match_detections(boxes, scores, labels, gt_boxes, gt_labels, iou_threshold=0.5)
    assert r.tp == 0
    assert r.fp == 1
    assert r.fn == 1


def test_one_gt_multiple_preds_only_one_tp():
    """One GT cannot match multiple predictions."""
    gt_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    gt_labels = ["fire"]

    boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=np.float32)
    scores = np.array([0.9, 0.8])
    labels = ["fire", "fire"]

    r = match_detections(boxes, scores, labels, gt_boxes, gt_labels)
    assert r.tp == 1
    assert r.fp == 1
    assert r.fn == 0


def test_one_pred_cannot_match_multiple_gt():
    """One prediction must not match multiple GTs."""
    pred_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    pred_scores = np.array([0.9])
    pred_labels = ["fire"]

    gt_boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    gt_labels = ["fire", "fire"]

    r = match_detections(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
    assert r.tp == 1
    assert r.fp == 0
    assert r.fn == 1   # Second GT has no match


def test_higher_confidence_takes_gt():
    """Higher confidence prediction should take the GT over lower confidence."""
    gt_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    gt_labels = ["fire"]

    # Two overlapping preds; higher confidence (index 0) should win
    boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=np.float32)
    scores = np.array([0.95, 0.60])
    labels = ["fire", "fire"]

    r = match_detections(boxes, scores, labels, gt_boxes, gt_labels)
    assert r.tp == 1
    assert r.fp == 1
    assert r.matched_pred_indices[0] == 0  # High confidence match


def test_multiple_classes():
    gt_boxes = np.array([[0, 0, 5, 5], [10, 10, 20, 20]], dtype=np.float32)
    gt_labels = ["fire", "smoke"]

    pred_boxes = np.array([[0, 0, 5, 5], [10, 10, 20, 20]], dtype=np.float32)
    pred_scores = np.array([0.9, 0.8])
    pred_labels = ["fire", "smoke"]

    r = match_detections(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
    assert r.tp == 2
    assert r.fp == 0
    assert r.fn == 0


# ── classify_image ────────────────────────────────────────────────────────────

def test_classify_complete_success_with_gt():
    r = MatchResult(tp=1, fp=0, fn=0)
    status = classify_image(r, n_gt=1, n_pred=1)
    assert status.status == "complete_success"


def test_classify_complete_success_no_gt_no_pred():
    r = MatchResult(tp=0, fp=0, fn=0)
    status = classify_image(r, n_gt=0, n_pred=0)
    assert status.status == "complete_success"


def test_classify_false_alarm():
    r = MatchResult(tp=0, fp=1, fn=0)
    status = classify_image(r, n_gt=0, n_pred=1)
    assert status.status == "false_alarm"


def test_classify_complete_failure():
    r = MatchResult(tp=0, fp=0, fn=2)
    status = classify_image(r, n_gt=2, n_pred=0)
    assert status.status == "complete_failure"


def test_classify_partial_failure():
    r = MatchResult(tp=1, fp=1, fn=1)
    status = classify_image(r, n_gt=2, n_pred=2)
    assert status.status == "partial_failure"


# ── MatchResult metrics ───────────────────────────────────────────────────────

def test_precision_recall_f1():
    r = MatchResult(tp=3, fp=1, fn=2)
    assert r.precision == pytest.approx(3 / 4)
    assert r.recall == pytest.approx(3 / 5)
    assert r.f1 == pytest.approx(2 * (3 / 4) * (3 / 5) / (3 / 4 + 3 / 5))


def test_zero_division_safe():
    r = MatchResult(tp=0, fp=0, fn=0)
    assert r.precision == 0.0
    assert r.recall == 0.0
    assert r.f1 == 0.0
