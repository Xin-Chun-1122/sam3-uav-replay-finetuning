"""Unit tests for FASDD-UAV COCO dataset parser."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.fasdd_coco import FASDDCocoDataset


def _make_coco_json(tmp_dir: str) -> str:
    """Create a minimal COCO JSON with fire and smoke categories."""
    data = {
        "images": [
            {"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img002.jpg", "width": 640, "height": 480},
            {"id": 3, "file_name": "img003.jpg", "width": 640, "height": 480},
        ],
        "categories": [
            {"id": 1, "name": "fire"},
            {"id": 2, "name": "smoke"},
            {"id": 3, "name": "ignored_class"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 60], "area": 3000},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [100, 100, 80, 40], "area": 3200},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [5, 5, 30, 30], "area": 900},
            {"id": 4, "image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "area": 100},  # ignored
        ],
    }
    json_path = os.path.join(tmp_dir, "test.json")
    with open(json_path, "w") as f:
        json.dump(data, f)
    return json_path


def _make_images(tmp_dir: str) -> None:
    """Create dummy image files."""
    import cv2
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for name in ["img001.jpg", "img002.jpg"]:
        cv2.imwrite(os.path.join(tmp_dir, name), img)
    # img003.jpg intentionally missing


@pytest.fixture
def fasdd_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        json_path = _make_coco_json(tmp)
        _make_images(tmp)
        ds = FASDDCocoDataset(json_path=json_path, images_root=tmp)
        yield ds


def test_category_mapping(fasdd_dataset):
    """Categories must be read from JSON, not hardcoded."""
    assert "fire" in fasdd_dataset._cat_id_to_name.values()
    assert "smoke" in fasdd_dataset._cat_id_to_name.values()
    # 'ignored_class' should not appear
    assert "ignored_class" not in fasdd_dataset._cat_id_to_name.values()


def test_found_images(fasdd_dataset):
    found = fasdd_dataset.images()
    names = {img.file_name for img in found}
    assert "img001.jpg" in names
    assert "img002.jpg" in names
    # img003.jpg is missing from disk
    assert "img003.jpg" not in names


def test_missing_images(fasdd_dataset):
    stats = fasdd_dataset.get_stats()
    assert stats.missing_images == 1
    assert any("img003" in n for n in stats.missing_image_names)


def test_bbox_conversion_xywh_to_xyxy(fasdd_dataset):
    """COCO bbox [x,y,w,h] must be converted to [x1,y1,x2,y2]."""
    for img in fasdd_dataset.images():
        if img.file_name == "img001.jpg":
            fire_anns = [a for a in img.annotations if a.category_name == "fire"]
            assert len(fire_anns) == 1
            x1, y1, x2, y2 = fire_anns[0].bbox_xyxy
            # Original bbox: [10, 20, 50, 60] → xyxy [10, 20, 60, 80]
            assert x1 == pytest.approx(10)
            assert y1 == pytest.approx(20)
            assert x2 == pytest.approx(60)
            assert y2 == pytest.approx(80)


def test_ignored_category_excluded(fasdd_dataset):
    """Annotations with unknown category_id must be ignored."""
    for img in fasdd_dataset.images():
        for ann in img.annotations:
            assert ann.category_name in {"fire", "smoke"}


def test_per_class_counts(fasdd_dataset):
    stats = fasdd_dataset.get_stats()
    assert stats.per_class_counts.get("fire", 0) == 2
    assert stats.per_class_counts.get("smoke", 0) == 1


def test_coco_gt_dict_format(fasdd_dataset):
    gt = fasdd_dataset.get_coco_gt_dict()
    assert "images" in gt
    assert "annotations" in gt
    assert "categories" in gt

    cat_names = {c["name"] for c in gt["categories"]}
    assert "fire" in cat_names
    assert "smoke" in cat_names

    for ann in gt["annotations"]:
        assert len(ann["bbox"]) == 4
        x, y, w, h = ann["bbox"]
        assert w > 0 and h > 0


def test_images_without_annotations(fasdd_dataset):
    stats = fasdd_dataset.get_stats()
    # img003.jpg has no fire/smoke annotations in the JSON (even though it's missing from disk)
    # → images_without_annotations counts all JSON images with no GT, including missing-on-disk
    assert stats.images_without_annotations == 1  # img003 has no annotations
