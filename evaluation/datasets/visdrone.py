"""VisDrone2019-DET dataset parser.

Category mapping (per VisDrone spec):
  0  ignored region   → skip
  1  pedestrian       → 'person'
  2  people           → 'person'
  3  bicycle          → skip
  4  car              → 'car'
  5  van              → skip
  6  truck            → skip
  7  tricycle         → skip
  8  awning-tricycle  → skip
  9  bus              → skip
  10 motor            → skip
  11 others           → skip

score=0 means ignored region → skip
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Category IDs that map to 'person'
PERSON_IDS = {1, 2}
# Category IDs that map to 'car'
CAR_IDS = {4}

# Combined mapping
CATEGORY_MAP: Dict[int, str] = {**{i: "person" for i in PERSON_IDS}, **{i: "car" for i in CAR_IDS}}


@dataclass
class VisDroneAnnotation:
    bbox_xyxy: np.ndarray   # [x1, y1, x2, y2] pixel coords
    category_name: str       # 'car' or 'person'
    truncation: int
    occlusion: int


@dataclass
class VisDroneImage:
    image_name: str          # e.g. '0000001_00000_d_0000001.jpg'
    abs_path: str
    annotations: List[VisDroneAnnotation] = field(default_factory=list)

    @property
    def image_id(self) -> int:
        # Use integer derived from filename for COCO compat
        stem = Path(self.image_name).stem.replace("_", "")
        # Take last 10 digits if available
        digits = "".join(c for c in stem if c.isdigit())
        return int(digits[-10:]) if digits else hash(self.image_name) & 0x7FFFFFFF


@dataclass
class VisDroneDatasetStats:
    total_images: int
    images_with_annotations: int
    missing_annotation_files: List[str]
    total_annotations: int
    per_class_counts: Dict[str, int]
    images_without_target_annotations: int


class VisDroneDataset:
    """
    Loads VisDrone2019-DET dataset.

    Expected directory structure:
        root/
            images/       *.jpg
            annotations/  *.txt  (same stem as images)
    """

    TARGET_CATEGORIES = {"car", "person"}

    def __init__(
        self,
        root: str,
        target_categories: Optional[set] = None,
    ) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.annotations_dir = self.root / "annotations"
        self.target_categories = target_categories or self.TARGET_CATEGORIES

        self._images: List[VisDroneImage] = []
        self._missing_annotations: List[str] = []

        if not self.images_dir.exists():
            raise FileNotFoundError(f"VisDrone images dir not found: {self.images_dir}")
        if not self.annotations_dir.exists():
            raise FileNotFoundError(f"VisDrone annotations dir not found: {self.annotations_dir}")

        self._load()

    def _parse_annotation_line(self, line: str) -> Optional[VisDroneAnnotation]:
        """Parse one line: bbox_left,bbox_top,bbox_width,bbox_height,score,category,trunc,occ"""
        line = line.strip()
        if not line:
            return None
        parts = line.split(",")
        if len(parts) < 8:
            return None

        try:
            bbox_left = float(parts[0])
            bbox_top = float(parts[1])
            bbox_width = float(parts[2])
            bbox_height = float(parts[3])
            score = int(parts[4])
            category = int(parts[5])
            truncation = int(parts[6])
            occlusion = int(parts[7])
        except (ValueError, IndexError):
            return None

        # Filter: score=0 means ignored region
        if score == 0:
            return None

        # Filter: category 0 is ignored region
        if category == 0:
            return None

        # Map category
        if category not in CATEGORY_MAP:
            return None

        cat_name = CATEGORY_MAP[category]
        if cat_name not in self.target_categories:
            return None

        # Convert xywh → xyxy
        x1 = bbox_left
        y1 = bbox_top
        x2 = bbox_left + bbox_width
        y2 = bbox_top + bbox_height

        # Skip degenerate boxes
        if x2 <= x1 or y2 <= y1:
            return None

        return VisDroneAnnotation(
            bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
            category_name=cat_name,
            truncation=truncation,
            occlusion=occlusion,
        )

    def _load(self) -> None:
        image_paths = sorted(self.images_dir.glob("*.jpg")) + sorted(self.images_dir.glob("*.png"))
        logger.info("Loading VisDrone: %d images in %s", len(image_paths), self.images_dir)

        for img_path in image_paths:
            ann_path = self.annotations_dir / (img_path.stem + ".txt")
            if not ann_path.exists():
                self._missing_annotations.append(img_path.name)
                continue

            vdi = VisDroneImage(
                image_name=img_path.name,
                abs_path=str(img_path),
            )

            with open(ann_path, "r") as f:
                for line in f:
                    ann = self._parse_annotation_line(line)
                    if ann is not None:
                        vdi.annotations.append(ann)

            self._images.append(vdi)

        logger.info(
            "Loaded %d VisDrone images (%d missing annotations)",
            len(self._images),
            len(self._missing_annotations),
        )

    def get_stats(self) -> VisDroneDatasetStats:
        per_class: Dict[str, int] = {c: 0 for c in self.target_categories}
        no_target = 0
        total_ann = 0

        for img in self._images:
            has_target = False
            for ann in img.annotations:
                per_class[ann.category_name] = per_class.get(ann.category_name, 0) + 1
                total_ann += 1
                has_target = True
            if not has_target:
                no_target += 1

        return VisDroneDatasetStats(
            total_images=len(self._images),
            images_with_annotations=sum(1 for img in self._images if img.annotations),
            missing_annotation_files=self._missing_annotations[:50],
            total_annotations=total_ann,
            per_class_counts=per_class,
            images_without_target_annotations=no_target,
        )

    def images(self) -> List[VisDroneImage]:
        return self._images

    def get_coco_gt_dict(self) -> Dict:
        """Return COCO-format GT dict for pycocotools."""
        cat_name_to_id = {name: idx + 1 for idx, name in enumerate(sorted(self.target_categories))}
        categories = [{"id": cid, "name": name} for name, cid in cat_name_to_id.items()]

        images_list = []
        annotations_list = []
        ann_id = 1

        # Need stable image_id — use index
        for img_idx, img in enumerate(self._images):
            iid = img_idx + 1
            images_list.append({
                "id": iid,
                "file_name": img.image_name,
                "width": 0,
                "height": 0,
            })
            for ann in img.annotations:
                x1, y1, x2, y2 = ann.bbox_xyxy
                w = x2 - x1
                h = y2 - y1
                annotations_list.append({
                    "id": ann_id,
                    "image_id": iid,
                    "category_id": cat_name_to_id[ann.category_name],
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                })
                ann_id += 1

        return {
            "images": images_list,
            "annotations": annotations_list,
            "categories": categories,
        }

    @property
    def image_id_map(self) -> Dict[str, int]:
        """Map image_name → stable integer image_id."""
        return {img.image_name: idx + 1 for idx, img in enumerate(self._images)}

    @property
    def category_name_to_coco_id(self) -> Dict[str, int]:
        return {name: idx + 1 for idx, name in enumerate(sorted(self.target_categories))}
