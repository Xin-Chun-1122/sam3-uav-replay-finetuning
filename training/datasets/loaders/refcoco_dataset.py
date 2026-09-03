"""
RefCOCO / RefCOCO+ / RefCOCOg Dataset Loader
Purpose: Strengthen complex natural language prompt understanding.
Loads referring expression annotations and maps to SAM3-I format.
"""

import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

from .base_dataset import BaseDataset


class RefCOCODataset(BaseDataset):
    """
    Loader for RefCOCO / RefCOCO+ / RefCOCOg datasets.

    These datasets contain referring expressions like:
    - "the person on the left"
    - "red car near the stop sign"
    - "person wearing a blue shirt"

    This teaches SAM3-I to follow complex natural language prompts.
    """

    SPLITS = {
        "refcoco":  {"ref_file": "refs(unc).p",  "splits": ["train", "val", "testA", "testB"]},
        "refcoco+": {"ref_file": "refs(unc).p",  "splits": ["train", "val", "testA", "testB"]},
        "refcocog": {"ref_file": "refs(umd).p",  "splits": ["train", "val", "test"]},
    }

    def __init__(
        self,
        data_root: str,          # e.g. /home/nthujerry123/datasets/refcoco
        img_dir: str,            # COCO train2014 or val2014 image folder
        instances_file: str,     # COCO instances.json
        dataset_name: str = "refcoco",   # refcoco / refcoco+ / refcocog
        split: str = "train",
        transforms=None,
        training: bool = True,
        use_paraphrase: bool = True,  # Augment by randomly choosing among expressions
    ):
        self.data_root = Path(data_root)
        self.instances_file = Path(instances_file)
        self.dataset_name_ref = dataset_name
        self.split = split
        self.use_paraphrase = use_paraphrase

        # Load COCO instances for bbox lookup
        print(f"[RefCOCO] Loading COCO instances from {instances_file}...")
        with open(instances_file, "r") as f:
            instances = json.load(f)

        # Build ann_id -> bbox + image_id mapping
        self.ann_id_to_info: Dict[int, Dict] = {}
        for ann in instances.get("annotations", []):
            self.ann_id_to_info[ann["id"]] = {
                "bbox": ann["bbox"],
                "image_id": ann["image_id"],
                "category_id": ann.get("category_id", -1),
            }

        # Build image_id -> file_name mapping
        self.img_id_to_info: Dict[int, Dict] = {}
        for img in instances.get("images", []):
            self.img_id_to_info[img["id"]] = img

        super().__init__(
            ann_file=str(self.data_root / self.SPLITS[dataset_name]["ref_file"]),
            img_dir=img_dir,
            transforms=transforms,
            training=training,
            dataset_name=f"refcoco_{dataset_name}_{split}",
        )

    def _load_annotations(self):
        """Load referring expressions from .p (pickle) file."""
        ref_file = self.ann_file
        with open(ref_file, "rb") as f:
            refs = pickle.load(f)

        # Filter by split
        self.refs = [r for r in refs if r["split"] == self.split]
        self.images = self.refs  # Use refs as "images" for length

        self.categories = [{"id": 0, "name": "referred_object"}]
        self.cat_id_to_name = {0: "referred_object"}
        self.annotations = {}

        print(f"[RefCOCO] Loaded {len(self.refs)} referring expressions (split={self.split})")

    def _pick_expression(self, ref: Dict) -> str:
        """Pick one expression from the list, with optional augmentation."""
        sentences = ref.get("sentences", [])
        if not sentences:
            return "the object"
        if self.use_paraphrase and self.training:
            return random.choice(sentences)["sent"]
        return sentences[0]["sent"]

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int) -> Dict:
        ref = self.refs[idx]
        ann_id = ref["ann_id"]
        ann_info = self.ann_id_to_info.get(ann_id, {})
        img_id = ref.get("image_id") or ann_info.get("image_id")
        img_info = self.img_id_to_info.get(img_id, {})

        # Load image
        file_name = img_info.get("file_name", f"{img_id}.jpg")
        img_path = self.img_dir / file_name
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (640, 640))

        # Get bbox [x, y, w, h] -> [x1, y1, x2, y2]
        bbox = ann_info.get("bbox", [0, 0, 10, 10])
        x1, y1, w, h = bbox
        box_xyxy = [x1, y1, x1 + w, y1 + h]

        # Get natural language expression (the "prompt")
        expression = self._pick_expression(ref)

        sample = {
            "image": img,
            "image_id": img_id,
            "boxes": torch.tensor([box_xyxy], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
            "prompts": [expression],  # The key: complex natural language prompt
            "width": img_info.get("width", img.width),
            "height": img_info.get("height", img.height),
            "source": "refcoco",
            "ref_id": ref.get("ref_id"),
            "expression": expression,
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample
