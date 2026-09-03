"""FASDD-UAV dataset parser using COCO format annotation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FASDDAnnotation:
    ann_id: int
    image_id: int
    category_name: str   # 'fire' or 'smoke'
    bbox_xyxy: np.ndarray  # [x1, y1, x2, y2] pixel coords
    area: float


@dataclass
class FASDDImage:
    image_id: int
    file_name: str
    width: int
    height: int
    abs_path: Optional[str] = None       # resolved absolute path
    annotations: List[FASDDAnnotation] = field(default_factory=list)


@dataclass
class FASDDDatasetStats:
    total_images_in_json: int
    found_images: int
    missing_images: int
    duplicate_basenames: List[str]
    total_annotations: int
    per_class_counts: Dict[str, int]
    images_without_annotations: int
    missing_image_names: List[str]


class FASDDCocoDataset:
    """
    Loads FASDD-UAV test set from COCO JSON annotation.

    Category names are read dynamically from the JSON — no hardcoded IDs.
    Targets 'fire' and 'smoke' categories.
    Images are located by recursive search from the root directory.
    """

    TARGET_CATEGORIES = {"fire", "smoke"}

    def __init__(
        self,
        json_path: str,
        images_root: str,
        target_categories: Optional[set] = None,
    ) -> None:
        self.json_path = Path(json_path)
        self.images_root = Path(images_root)
        self.target_categories = target_categories or self.TARGET_CATEGORIES

        self._data: Dict = {}
        self._cat_id_to_name: Dict[int, str] = {}
        self._images: Dict[int, FASDDImage] = {}
        self._basename_to_path: Dict[str, Path] = {}

        self._load()

    def _build_image_index(self) -> None:
        """Recursively index all images under images_root by basename."""
        logger.info("Indexing images under %s ...", self.images_root)
        duplicates = []
        for p in self.images_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                key = p.name
                if key in self._basename_to_path:
                    if key not in duplicates:
                        duplicates.append(key)
                    logger.warning("Duplicate basename: %s → %s (prev %s)", key, p, self._basename_to_path[key])
                else:
                    self._basename_to_path[key] = p
        logger.info("Indexed %d unique image basenames (%d duplicates)", len(self._basename_to_path), len(duplicates))
        self._duplicate_basenames = duplicates

    def _load(self) -> None:
        logger.info("Loading FASDD annotation: %s", self.json_path)
        with open(self.json_path, "r") as f:
            self._data = json.load(f)

        # Build category map dynamically
        for cat in self._data.get("categories", []):
            name = cat["name"].lower().strip()
            if name in self.target_categories:
                self._cat_id_to_name[cat["id"]] = name

        target_cat_names = set(self._cat_id_to_name.values())
        logger.info("Category mapping: %s", self._cat_id_to_name)

        if not self._cat_id_to_name:
            raise ValueError(
                f"No target categories {self.target_categories} found in {self.json_path}. "
                f"Available: {[c['name'] for c in self._data.get('categories', [])]}"
            )

        # Build image index
        self._build_image_index()

        # Load images
        for img_info in self._data.get("images", []):
            iid = img_info["id"]
            fname = img_info["file_name"]
            basename = Path(fname).name
            abs_path = self._basename_to_path.get(basename)
            if abs_path is None:
                # Try full path match
                full = self.images_root / fname
                if full.exists():
                    abs_path = full

            self._images[iid] = FASDDImage(
                image_id=iid,
                file_name=fname,
                width=img_info.get("width", 0),
                height=img_info.get("height", 0),
                abs_path=str(abs_path) if abs_path else None,
            )

        # Load annotations
        for ann in self._data.get("annotations", []):
            cat_id = ann.get("category_id")
            if cat_id not in self._cat_id_to_name:
                continue
            iid = ann["image_id"]
            if iid not in self._images:
                continue

            x, y, w, h = ann["bbox"]  # COCO: [x_left, y_top, width, height]
            bbox_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)

            self._images[iid].annotations.append(
                FASDDAnnotation(
                    ann_id=ann["id"],
                    image_id=iid,
                    category_name=self._cat_id_to_name[cat_id],
                    bbox_xyxy=bbox_xyxy,
                    area=float(ann.get("area", w * h)),
                )
            )

    def get_stats(self) -> FASDDDatasetStats:
        found = sum(1 for img in self._images.values() if img.abs_path is not None)
        missing_names = [img.file_name for img in self._images.values() if img.abs_path is None]

        per_class: Dict[str, int] = {c: 0 for c in self.target_categories}
        total_ann = 0
        no_ann = 0
        for img in self._images.values():
            if not img.annotations:
                no_ann += 1
            for ann in img.annotations:
                per_class[ann.category_name] = per_class.get(ann.category_name, 0) + 1
                total_ann += 1

        return FASDDDatasetStats(
            total_images_in_json=len(self._images),
            found_images=found,
            missing_images=len(missing_names),
            duplicate_basenames=self._duplicate_basenames,
            total_annotations=total_ann,
            per_class_counts=per_class,
            images_without_annotations=no_ann,
            missing_image_names=missing_names[:50],
        )

    def images(self) -> List[FASDDImage]:
        """Return only images that were found on disk."""
        return [img for img in self._images.values() if img.abs_path is not None]

    def all_images(self) -> List[FASDDImage]:
        """Return all images (including missing)."""
        return list(self._images.values())

    def get_coco_gt_dict(self) -> Dict:
        """Return a COCO-format GT dict for pycocotools (found images only)."""
        # Build category list
        cat_name_to_id = {name: idx + 1 for idx, name in enumerate(sorted(self.target_categories))}
        categories = [{"id": cid, "name": name} for name, cid in cat_name_to_id.items()]

        images_list = []
        annotations_list = []
        ann_id = 1

        for img in self.images():
            images_list.append({
                "id": img.image_id,
                "file_name": img.file_name,
                "width": img.width,
                "height": img.height,
            })
            for ann in img.annotations:
                x1, y1, x2, y2 = ann.bbox_xyxy
                w = x2 - x1
                h = y2 - y1
                annotations_list.append({
                    "id": ann_id,
                    "image_id": img.image_id,
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
    def category_name_to_coco_id(self) -> Dict[str, int]:
        return {name: idx + 1 for idx, name in enumerate(sorted(self.target_categories))}

    def check_split_leakage(
        self,
        train_json: Optional[str],
        val_json: Optional[str],
        output_path: str,
    ) -> None:
        """Check if file_names overlap between train/val/test splits."""
        test_names = {img.file_name for img in self._images.values()}
        report_lines = ["# Data Leakage Report\n"]

        for split_name, split_path in [("train", train_json), ("val", val_json)]:
            if split_path is None or not Path(split_path).exists():
                report_lines.append(f"{split_name}: not found\n")
                continue
            with open(split_path) as f:
                split_data = json.load(f)
            split_names = {img["file_name"] for img in split_data.get("images", [])}
            overlap = test_names & split_names
            report_lines.append(f"{split_name} total: {len(split_names)}\n")
            report_lines.append(f"test ∩ {split_name}: {len(overlap)}\n")
            if overlap:
                report_lines.append(f"  LEAKED files (first 10): {sorted(overlap)[:10]}\n")
            else:
                report_lines.append("  No leakage detected.\n")
            report_lines.append("\n")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.writelines(report_lines)
        logger.info("Leakage report written to %s", output_path)
