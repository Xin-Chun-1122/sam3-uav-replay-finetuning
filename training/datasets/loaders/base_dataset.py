"""
Base Dataset class for all SAM3-I dataset loaders.
Provides common interface, COCO-format loading, and ID management.
"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all dataset loaders in this project.
    Subclass this to add support for new datasets.

    Required methods to implement:
        - load_annotations(): loads self.images and self.annotations
        - __getitem__(): returns a sample dict
    """

    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        transforms=None,
        max_ann_per_img: int = 500,
        training: bool = True,
        dataset_name: str = "base",
    ):
        self.ann_file = Path(ann_file)
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        self.max_ann_per_img = max_ann_per_img
        self.training = training
        self.dataset_name = dataset_name

        # Will be populated by load_annotations()
        self.images: List[Dict] = []
        self.annotations: Dict[int, List] = {}  # img_id -> list of anns
        self.categories: List[Dict] = []
        self.cat_id_to_name: Dict[int, str] = {}

        self._load_annotations()
        print(f"[{dataset_name}] Loaded {len(self.images)} images, "
              f"{sum(len(v) for v in self.annotations.values())} annotations")

    @abstractmethod
    def _load_annotations(self):
        """Load self.images, self.annotations, self.categories."""
        pass

    def _load_coco_json(self, ann_file: Path) -> Tuple[List, Dict, List]:
        """Helper to load standard COCO-format JSON."""
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        images = data.get("images", [])
        categories = data.get("categories", [])
        cat_id_to_name = {c["id"]: c["name"] for c in categories}

        # Build img_id -> annotations mapping
        annotations = {}
        for img in images:
            annotations[img["id"]] = []

        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in annotations:
                annotations[img_id].append(ann)

        return images, annotations, categories

    def load_image(self, img_info: Dict) -> Image.Image:
        """Load a PIL image from img_info dict."""
        file_name = Path(img_info["file_name"]).name
        img_path = self.img_dir / file_name
        if not img_path.exists():
            # Try with full file_name path
            img_path = self.img_dir / img_info["file_name"]
        return Image.open(img_path).convert("RGB")

    def get_category_name(self, cat_id: int) -> str:
        return self.cat_id_to_name.get(cat_id, "unknown")

    def __len__(self) -> int:
        return len(self.images)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"dataset={self.dataset_name}, "
                f"images={len(self.images)}, "
                f"categories={list(self.cat_id_to_name.values())})")
