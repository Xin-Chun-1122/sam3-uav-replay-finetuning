"""
UAV Detection Dataset Loader
Handles: FASDD_UAV (fire, smoke) + VisDrone (car, person)
Outputs SAM3-I compatible format with text prompts.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np

from .base_dataset import BaseDataset

# Category mapping: unify all source datasets into 4 classes
UNIFIED_CATEGORIES = [
    {"id": 0, "name": "fire",   "supercategory": "hazard"},
    {"id": 1, "name": "smoke",  "supercategory": "hazard"},
    {"id": 2, "name": "car",    "supercategory": "vehicle"},
    {"id": 3, "name": "person", "supercategory": "human"},
]

# Multiple text prompts per category (for open-vocabulary robustness)
CATEGORY_PROMPTS = {
    "fire":   ["fire", "flame", "burning fire", "wildfire", "fire on the ground"],
    "smoke":  ["smoke", "dark smoke", "smoke plume", "haze", "transparent smoke"],
    "car":    ["car", "vehicle", "automobile", "car from aerial view", "moving vehicle"],
    "person": ["person", "human", "pedestrian", "person from above", "people"],
}

# VisDrone category ID -> unified category name
VISDRONE_CAT_MAP = {
    1: "pedestrian",   # -> person
    2: "person",       # -> person
    4: "car",          # -> car
    5: "van",          # -> car
    6: "truck",        # -> car
    9: "motor",        # skip (motorcycle)
    10: "bicycle",     # skip
    11: "awning-tricycle", # skip
}
VISDRONE_TO_UNIFIED = {
    1: 3,   # pedestrian -> person
    2: 3,   # person -> person
    4: 2,   # car -> car
    5: 2,   # van -> car
    6: 2,   # truck -> car
}

# FASDD category name -> unified ID
FASDD_TO_UNIFIED = {
    "fire":  0,
    "smoke": 1,
}


class UAVDataset(BaseDataset):
    """
    Combined UAV dataset loader for fire, smoke, car, person.
    Supports FASDD_UAV format and VisDrone COCO-converted format.
    """

    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        source: str = "fasdd",   # "fasdd" or "visdrone"
        transforms=None,
        max_ann_per_img: int = 200,
        training: bool = True,
        use_prompt_augmentation: bool = True,
    ):
        self.source = source
        self.use_prompt_augmentation = use_prompt_augmentation
        self.cat_id_to_unified: Dict[int, int] = {}

        super().__init__(
            ann_file=ann_file,
            img_dir=img_dir,
            transforms=transforms,
            max_ann_per_img=max_ann_per_img,
            training=training,
            dataset_name=f"uav_{source}",
        )

    def _load_annotations(self):
        images, annotations, categories = self._load_coco_json(self.ann_file)
        self.images = images
        self.annotations = annotations
        self.categories = UNIFIED_CATEGORIES
        self.cat_id_to_name = {c["id"]: c["name"] for c in UNIFIED_CATEGORIES}

        # Build mapping from source cat_id -> unified cat_id
        if self.source == "visdrone":
            self.cat_id_to_unified = VISDRONE_TO_UNIFIED
        else:
            # FASDD: use category names
            for cat in categories:
                name = cat["name"].lower()
                if name in FASDD_TO_UNIFIED:
                    self.cat_id_to_unified[cat["id"]] = FASDD_TO_UNIFIED[name]

    def _get_prompt_for_category(self, cat_id: int) -> str:
        """Get a text prompt for a category ID, with optional augmentation."""
        cat_name = self.cat_id_to_name.get(cat_id, "object")
        prompts = CATEGORY_PROMPTS.get(cat_name, [cat_name])
        if self.use_prompt_augmentation and self.training:
            return random.choice(prompts)
        return prompts[0]  # Default to first (most standard) prompt

    def __getitem__(self, idx: int) -> Dict:
        img_info = self.images[idx]
        img = self.load_image(img_info)
        img_id = img_info["id"]

        anns = self.annotations.get(img_id, [])

        # Map to unified category IDs and filter unsupported categories
        boxes, labels, prompts = [], [], []
        for ann in anns[:self.max_ann_per_img]:
            src_cat_id = ann.get("category_id", -1)
            unified_cat_id = self.cat_id_to_unified.get(src_cat_id, -1)
            if unified_cat_id < 0:
                continue  # Skip unsupported categories

            bbox = ann.get("bbox", [0, 0, 0, 0])  # COCO format: [x, y, w, h]
            # Convert to [x1, y1, x2, y2]
            x1, y1, w, h = bbox
            if w <= 0 or h <= 0:
                continue
            boxes.append([x1, y1, x1 + w, y1 + h])
            labels.append(unified_cat_id)
            prompts.append(self._get_prompt_for_category(unified_cat_id))

        sample = {
            "image": img,
            "image_id": img_id,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompts": prompts,
            "width": img_info.get("width", img.width),
            "height": img_info.get("height", img.height),
            "source": self.source,
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample


class MixedUAVDataset(torch.utils.data.Dataset):
    """
    Combines multiple UAV datasets (FASDD + VisDrone) with configurable mix ratios.
    """

    def __init__(self, datasets: List[UAVDataset], weights: Optional[List[float]] = None):
        self.datasets = datasets
        if weights is None:
            weights = [1.0 / len(datasets)] * len(datasets)
        self.weights = weights

        # Build flat index with source tracking
        self.index_map = []
        for ds_idx, ds in enumerate(datasets):
            for item_idx in range(len(ds)):
                self.index_map.append((ds_idx, item_idx))

        print(f"[MixedUAVDataset] Total: {len(self.index_map)} samples from {len(datasets)} datasets")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict:
        ds_idx, item_idx = self.index_map[idx]
        return self.datasets[ds_idx][item_idx]
