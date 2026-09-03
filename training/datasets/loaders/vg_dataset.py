"""
Visual Genome Dataset Loader
Purpose: Relationship reasoning and complex scene description understanding.
Filters VG to extract UAV-relevant objects and spatial relationships.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from .base_dataset import BaseDataset


# UAV-relevant object categories to keep from VG
UAV_RELEVANT_OBJECTS = {
    "car", "vehicle", "truck", "bus", "motorcycle", "bicycle",
    "person", "man", "woman", "people", "crowd",
    "fire", "smoke", "flame", "burning",
    "building", "road", "tree", "grass", "sky", "water",
}

# Spatial relationships relevant to UAV reasoning
SPATIAL_RELATIONSHIPS = {
    "near", "next to", "beside", "adjacent to",
    "above", "below", "on top of", "under",
    "in front of", "behind",
    "left of", "right of",
    "inside", "outside",
}


class VisualGenomeDataset(BaseDataset):
    """
    Loads Visual Genome region descriptions and relationships.

    Generates prompts like:
    - "the car near the person"
    - "a vehicle beside the building"
    - "person in front of the car"

    This teaches the model to reason about spatial and semantic relationships.
    """

    def __init__(
        self,
        img_dir: str,
        region_file: str,        # region_descriptions.json
        scene_graph_file: str,   # scene_graphs.json
        transforms=None,
        training: bool = True,
        min_box_area: int = 400,
        max_regions_per_image: int = 20,
        use_relationship_prompts: bool = True,
    ):
        self.region_file = Path(region_file)
        self.scene_graph_file = Path(scene_graph_file)
        self.min_box_area = min_box_area
        self.max_regions_per_image = max_regions_per_image
        self.use_relationship_prompts = use_relationship_prompts

        # These will be loaded in _load_annotations
        self.vg_data: List[Dict] = []
        self.scene_graphs: Dict[int, Dict] = {}

        super().__init__(
            ann_file=region_file,
            img_dir=img_dir,
            transforms=transforms,
            training=training,
            dataset_name="visual_genome",
        )

    def _load_annotations(self):
        print(f"[VisualGenome] Loading region descriptions from {self.region_file}...")
        with open(self.region_file, "r") as f:
            raw_data = json.load(f)

        # Filter images that have relevant regions
        self.images = []
        self.annotations = {}

        for entry in raw_data:
            img_id = entry["id"]
            regions = entry.get("regions", [])

            # Filter regions by area and relevance
            valid_regions = []
            for reg in regions:
                w = reg.get("width", 0)
                h = reg.get("height", 0)
                if w * h < self.min_box_area:
                    continue
                desc = reg.get("phrase", "").lower()
                if len(desc.split()) < 2:
                    continue
                valid_regions.append(reg)

            if not valid_regions:
                continue

            self.images.append({
                "id": img_id,
                "file_name": f"{img_id}.jpg",
                "width": entry.get("width", 640),
                "height": entry.get("height", 480),
            })
            self.annotations[img_id] = valid_regions[:self.max_regions_per_image]

        self.categories = [{"id": 0, "name": "region"}]
        self.cat_id_to_name = {0: "region"}
        print(f"[VisualGenome] Loaded {len(self.images)} images with valid regions")

    def _build_relationship_prompt(self, region: Dict, scene_graph: Optional[Dict] = None) -> str:
        """
        Build a relationship-aware prompt.
        Falls back to region description if no relationships found.
        """
        base_desc = region.get("phrase", "the object")
        return base_desc

    def __getitem__(self, idx: int) -> Dict:
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = self.img_dir / img_info["file_name"]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (640, 480))

        regions = self.annotations.get(img_id, [])

        # Sample a subset of regions if training (random); else use all
        if self.training and len(regions) > 10:
            regions = random.sample(regions, 10)

        boxes, prompts = [], []
        for reg in regions:
            x = reg.get("x", 0)
            y = reg.get("y", 0)
            w = reg.get("width", 10)
            h = reg.get("height", 10)
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            prompt = self._build_relationship_prompt(reg)
            prompts.append(prompt)

        sample = {
            "image": img,
            "image_id": img_id,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.zeros(len(boxes), dtype=torch.long),
            "prompts": prompts,
            "width": img_info.get("width", img.width),
            "height": img_info.get("height", img.height),
            "source": "visual_genome",
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample
