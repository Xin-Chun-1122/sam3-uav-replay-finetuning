"""Unit tests for evaluation metrics."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.matching import match_detections, MatchResult


def test_aggregate_precision_recall():
    """Verify aggregate computation across multiple images."""
    # Image 1: 1 TP, 0 FP, 0 FN
    r1 = MatchResult(tp=1, fp=0, fn=0)
    # Image 2: 2 TP, 1 FP, 1 FN
    r2 = MatchResult(tp=2, fp=1, fn=1)

    total_tp = r1.tp + r2.tp  # 3
    total_fp = r1.fp + r2.fp  # 1
    total_fn = r1.fn + r2.fn  # 1

    precision = total_tp / (total_tp + total_fp)  # 3/4
    recall = total_tp / (total_tp + total_fn)      # 3/4
    f1 = 2 * precision * recall / (precision + recall)

    assert precision == pytest.approx(0.75)
    assert recall == pytest.approx(0.75)
    assert f1 == pytest.approx(0.75)


def test_miss_rate_calculation():
    r = MatchResult(tp=1, fp=0, fn=3)
    # Miss rate = FN / (TP + FN) = 3/4
    assert r.miss_rate == pytest.approx(0.75)


def test_object_success_rate():
    r = MatchResult(tp=3, fp=2, fn=2)
    # OSR = TP / (TP + FN) = 3/5
    assert r.object_success_rate == pytest.approx(0.6)


def test_all_zero_safe():
    r = MatchResult(tp=0, fp=0, fn=0)
    assert r.precision == 0.0
    assert r.recall == 0.0
    assert r.f1 == 0.0
    assert r.miss_rate == 0.0
    assert r.object_success_rate == 0.0


def test_iou_threshold_50_boundary():
    """Box with IoU exactly at threshold should match."""
    # Create boxes with IoU = 0.5 exactly
    # Box A: [0, 0, 10, 10] area=100
    # Box B: [5, 0, 15, 10] area=100, inter=[5,0,10,10]=50, union=150
    # IoU = 50/150 = 0.333... → below 0.5, no match
    pred = np.array([[5, 0, 15, 10]], dtype=np.float32)
    gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
    r = match_detections(pred, np.array([0.9]), ["fire"], gt, ["fire"], iou_threshold=0.5)
    assert r.tp == 0
    assert r.fp == 1

    # Box with IoU = 0.5 exactly:
    # [0,0,20,10] and [10,0,30,10]: inter=100, area_a=200, area_b=200, union=300
    # IoU = 100/300 = 0.333...
    # For IoU=0.5: A=[0,0,10,10], B=[5,0,15,10] inter=50 union=150 → 0.333
    # Better: A=[0,0,10,10], B=[0,0,10,5] inter=50 union=100 → 0.5
    pred2 = np.array([[0, 0, 10, 5]], dtype=np.float32)
    gt2 = np.array([[0, 0, 10, 10]], dtype=np.float32)
    r2 = match_detections(pred2, np.array([0.9]), ["fire"], gt2, ["fire"], iou_threshold=0.5)
    assert r2.tp == 1  # IoU = 50/100 = 0.5 ≥ threshold
