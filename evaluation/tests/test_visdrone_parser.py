"""Unit tests for VisDrone dataset parser."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.visdrone import VisDroneDataset, CATEGORY_MAP, PERSON_IDS, CAR_IDS


def _make_visdrone_dataset(tmp_dir: str) -> str:
    """Create minimal VisDrone dataset structure."""
    images_dir = os.path.join(tmp_dir, "images")
    ann_dir = os.path.join(tmp_dir, "annotations")
    os.makedirs(images_dir)
    os.makedirs(ann_dir)

    # Create dummy images
    import cv2
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for name in ["img001.jpg", "img002.jpg", "img003.jpg"]:
        cv2.imwrite(os.path.join(images_dir, name), img)

    # img001: pedestrian + car + ignored region + bicycle
    with open(os.path.join(ann_dir, "img001.txt"), "w") as f:
        f.write("10,20,50,60,1,1,0,0\n")   # pedestrian (cat 1) → person
        f.write("200,300,80,40,1,2,0,0\n")  # people (cat 2) → person
        f.write("500,100,60,80,1,4,0,1\n")  # car (cat 4)
        f.write("0,0,30,30,0,1,0,0\n")      # score=0 → ignored
        f.write("100,100,40,40,1,3,0,0\n")  # bicycle (cat 3) → skip

    # img002: car only
    with open(os.path.join(ann_dir, "img002.txt"), "w") as f:
        f.write("50,50,100,100,1,4,0,0\n")  # car

    # img003: empty (no target objects)
    with open(os.path.join(ann_dir, "img003.txt"), "w") as f:
        f.write("0,0,20,20,0,0,0,0\n")  # ignored region

    return tmp_dir


@pytest.fixture
def visdrone_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        _make_visdrone_dataset(tmp)
        ds = VisDroneDataset(root=tmp)
        yield ds


def test_person_category_merge(visdrone_dataset):
    """Cat 1 (pedestrian) and cat 2 (people) must both map to 'person'."""
    assert CATEGORY_MAP[1] == "person"
    assert CATEGORY_MAP[2] == "person"


def test_car_category(visdrone_dataset):
    assert CATEGORY_MAP[4] == "car"


def test_ignored_score_zero_excluded(visdrone_dataset):
    """score=0 annotations must be excluded."""
    img = next(img for img in visdrone_dataset.images() if img.image_name == "img001.jpg")
    for ann in img.annotations:
        # If score=0 was included, there would be duplicate person entries
        pass
    # Check no degenerate boxes (the ignored box was 0,0,30,30 with score=0)
    boxes = [a.bbox_xyxy.tolist() for a in img.annotations if a.category_name == "person"]
    # Should have exactly 2 person boxes (from cat 1 and cat 2)
    assert len(boxes) == 2


def test_bicycle_excluded(visdrone_dataset):
    """Non-target categories (bicycle, truck, etc.) must be excluded."""
    for img in visdrone_dataset.images():
        for ann in img.annotations:
            assert ann.category_name in {"car", "person"}


def test_bbox_xywh_to_xyxy(visdrone_dataset):
    """bbox_left,bbox_top,bbox_width,bbox_height must convert to xyxy."""
    img = next(img for img in visdrone_dataset.images() if img.image_name == "img001.jpg")
    person_anns = [a for a in img.annotations if a.category_name == "person"]
    # First person: 10,20,50,60 → xyxy [10,20,60,80]
    ann = person_anns[0]
    x1, y1, x2, y2 = ann.bbox_xyxy
    assert x2 - x1 == pytest.approx(50)
    assert y2 - y1 == pytest.approx(60)


def test_stats(visdrone_dataset):
    stats = visdrone_dataset.get_stats()
    assert stats.total_images == 3
    assert stats.per_class_counts.get("car", 0) == 2   # img001 car + img002 car
    assert stats.per_class_counts.get("person", 0) == 2  # img001 pedestrian + people
    assert stats.images_without_target_annotations == 1   # img003


def test_coco_gt_dict(visdrone_dataset):
    gt = visdrone_dataset.get_coco_gt_dict()
    assert "images" in gt
    assert "annotations" in gt
    assert "categories" in gt

    cat_names = {c["name"] for c in gt["categories"]}
    assert "car" in cat_names
    assert "person" in cat_names

    for ann in gt["annotations"]:
        x, y, w, h = ann["bbox"]
        assert w > 0 and h > 0


def test_missing_annotation_file():
    with tempfile.TemporaryDirectory() as tmp:
        images_dir = os.path.join(tmp, "images")
        ann_dir = os.path.join(tmp, "annotations")
        os.makedirs(images_dir)
        os.makedirs(ann_dir)

        import cv2
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Image with no corresponding annotation file
        cv2.imwrite(os.path.join(images_dir, "orphan.jpg"), img)
        # Image with annotation file
        cv2.imwrite(os.path.join(images_dir, "with_ann.jpg"), img)
        with open(os.path.join(ann_dir, "with_ann.txt"), "w") as f:
            f.write("10,10,50,50,1,4,0,0\n")

        ds = VisDroneDataset(root=tmp)
        # Only with_ann.jpg should be loaded
        assert len(ds.images()) == 1
        assert ds.images()[0].image_name == "with_ann.jpg"
        stats = ds.get_stats()
        assert len(stats.missing_annotation_files) == 1
