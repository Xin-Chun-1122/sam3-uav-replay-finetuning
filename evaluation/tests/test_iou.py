"""Unit tests for IoU computation."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.matching import compute_iou, compute_iou_matrix


def test_perfect_overlap():
    a = np.array([0, 0, 10, 10], dtype=np.float32)
    assert compute_iou(a, a) == pytest.approx(1.0)


def test_no_overlap():
    a = np.array([0, 0, 5, 5], dtype=np.float32)
    b = np.array([10, 10, 20, 20], dtype=np.float32)
    assert compute_iou(a, b) == pytest.approx(0.0)


def test_half_overlap():
    a = np.array([0, 0, 4, 4], dtype=np.float32)
    b = np.array([2, 0, 6, 4], dtype=np.float32)
    # Intersection: 2x4=8, Union: 4x4+4x4-8=24
    iou = compute_iou(a, b)
    assert iou == pytest.approx(8 / 24, abs=1e-5)


def test_containment():
    outer = np.array([0, 0, 10, 10], dtype=np.float32)
    inner = np.array([2, 2, 8, 8], dtype=np.float32)
    # Intersection = 6x6=36, Union = 100
    iou = compute_iou(outer, inner)
    assert iou == pytest.approx(36 / 100, abs=1e-5)


def test_degenerate_zero_area():
    a = np.array([5, 5, 5, 5], dtype=np.float32)  # zero area
    b = np.array([0, 0, 10, 10], dtype=np.float32)
    assert compute_iou(a, b) == pytest.approx(0.0)


def test_iou_matrix_shape():
    preds = np.array([[0, 0, 5, 5], [5, 5, 10, 10]], dtype=np.float32)
    gts = np.array([[0, 0, 5, 5], [3, 3, 8, 8], [9, 9, 15, 15]], dtype=np.float32)
    mat = compute_iou_matrix(preds, gts)
    assert mat.shape == (2, 3)


def test_iou_matrix_empty_preds():
    gts = np.array([[0, 0, 5, 5]], dtype=np.float32)
    mat = compute_iou_matrix(np.empty((0, 4), np.float32), gts)
    assert mat.shape == (0, 1)


def test_iou_matrix_empty_gts():
    preds = np.array([[0, 0, 5, 5]], dtype=np.float32)
    mat = compute_iou_matrix(preds, np.empty((0, 4), np.float32))
    assert mat.shape == (1, 0)


def test_iou_matrix_values():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10]], dtype=np.float32)
    mat = compute_iou_matrix(a, b)
    assert mat[0, 0] == pytest.approx(1.0)
