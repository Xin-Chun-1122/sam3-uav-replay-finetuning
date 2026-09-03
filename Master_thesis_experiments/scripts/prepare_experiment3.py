#!/usr/bin/env python3
"""Prepare and freeze RefDrone annotations for Experiment 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


TARGET_ID_TO_CLASS = {
    1: "person",       # VisDrone pedestrian
    2: "person",       # VisDrone people
    4: "car",
    6: "truck",
    9: "bus",
    10: "motorcycle",  # VisDrone motor
}
CATEGORIES = ("person", "car", "truck", "bus", "motorcycle")
SYNONYMS = {
    "person": "human",
    "car": "automobile",
    "truck": "lorry",
    "motorcycle": "motorbike",
}
COLORS = (
    "white", "black", "red", "blue", "yellow", "green",
    "gray", "grey", "silver", "brown", "orange",
)
SPATIAL_PATTERNS = (
    "left", "right", "top", "bottom", "middle", "center", "near",
    "next to", "beside", "adjacent", "behind", "front", "between",
    "around", "along", "intersection", "road", "parking", "corner", "side",
)


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb > inter else 0.0


def deduplicate(boxes: list[list[float]], threshold: float = 0.95) -> list[list[float]]:
    kept: list[list[float]] = []
    for box in sorted(boxes):
        if not any(iou(box, old) >= threshold for old in kept):
            kept.append(box)
    return kept


def has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, flags=re.I) is not None


def stable_rank(seed: int, *parts: object) -> str:
    value = ":".join(map(str, (seed, *parts)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-category", type=int, default=10)
    args = parser.parse_args()

    source = args.dataset_root / "RefDrone_test_mdetr.json"
    images_dir = args.dataset_root / "all_image"
    data = json.loads(source.read_text(encoding="utf-8"))
    image_rows = {int(row["id"]): row for row in data["images"]}
    anns_by_query: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_query[int(ann["image_id"])].append(ann)

    records_by_file: dict[str, list[dict]] = defaultdict(list)
    for row in data["images"]:
        file_name = row["file_name"]
        if not (images_dir / file_name).is_file():
            raise FileNotFoundError(images_dir / file_name)
        records_by_file[file_name].append(row)

    def query_boxes(query_id: int, category: str) -> list[list[float]]:
        return deduplicate([
            xywh_to_xyxy(ann["bbox"])
            for ann in anns_by_query[query_id]
            if TARGET_ID_TO_CLASS.get(int(ann["category_id"])) == category
        ])

    def file_boxes(file_name: str, category: str, color: str | None = None) -> list[list[float]]:
        boxes: list[list[float]] = []
        for row in records_by_file[file_name]:
            if color is not None and not has_word(row["caption"], color):
                continue
            boxes.extend(query_boxes(int(row["id"]), category))
        return deduplicate(boxes)

    unique_files = sorted(records_by_file)
    category_gt = {
        file_name: {category: file_boxes(file_name, category) for category in CATEGORIES}
        for file_name in unique_files
    }

    category_units = []
    for file_name in unique_files:
        row = records_by_file[file_name][0]
        for category in CATEGORIES:
            category_units.append({
                "unit_id": f"{file_name}:{category}",
                "file_name": file_name,
                "width": int(row["width"]),
                "height": int(row["height"]),
                "category": category,
                "standard_prompt": category,
                "synonym_prompt": SYNONYMS.get(category),
                "gt_boxes_xyxy": category_gt[file_name][category],
            })

    candidates: dict[str, list[dict]] = defaultdict(list)
    for row in data["images"]:
        query_id = int(row["id"])
        caption = row["caption"].strip()
        lower = caption.lower()
        colors = [color for color in COLORS if has_word(lower, color)]
        if len(colors) != 1 or not any(pattern in lower for pattern in SPATIAL_PATTERNS):
            continue
        mapped_categories = {
            TARGET_ID_TO_CLASS[int(ann["category_id"])]
            for ann in anns_by_query[query_id]
            if int(ann["category_id"]) in TARGET_ID_TO_CLASS
        }
        has_non_target = any(
            int(ann["category_id"]) not in TARGET_ID_TO_CLASS
            for ann in anns_by_query[query_id]
        )
        if len(mapped_categories) != 1 or has_non_target:
            continue
        category = next(iter(mapped_categories))
        color = colors[0]
        layer1 = category_gt[row["file_name"]][category]
        layer2 = file_boxes(row["file_name"], category, color)
        layer3 = query_boxes(query_id, category)
        if not layer1 or not layer2 or not layer3:
            continue
        # The current query contributes to layer 2, so the nesting must hold.
        if any(not any(iou(box, parent) >= 0.95 for parent in layer2) for box in layer3):
            continue
        if any(not any(iou(box, parent) >= 0.95 for parent in layer1) for box in layer2):
            continue
        attribute = "gray" if color == "grey" else color
        candidates[category].append({
            "query_id": query_id,
            "file_name": row["file_name"],
            "width": int(row["width"]),
            "height": int(row["height"]),
            "category": category,
            "attribute": attribute,
            "source_expression": caption,
            "layers": [
                {"layer": 1, "prompt": category, "gt_boxes_xyxy": layer1},
                {"layer": 2, "prompt": f"{attribute} {category}", "gt_boxes_xyxy": layer2},
                {"layer": 3, "prompt": caption, "gt_boxes_xyxy": layer3},
            ],
        })

    selected = []
    used_files: set[str] = set()
    for category in CATEGORIES:
        ranked = sorted(
            candidates[category],
            key=lambda item: (
                # Prefer examples where specificity actually reduces the target set.
                -int(len(item["layers"][0]["gt_boxes_xyxy"]) > len(item["layers"][1]["gt_boxes_xyxy"])),
                -int(len(item["layers"][1]["gt_boxes_xyxy"]) > len(item["layers"][2]["gt_boxes_xyxy"])),
                stable_rank(args.seed, category, item["query_id"]),
            ),
        )
        chosen = []
        for item in ranked:
            if item["file_name"] in used_files:
                continue
            chosen.append(item)
            used_files.add(item["file_name"])
            if len(chosen) == args.per_category:
                break
        if len(chosen) != args.per_category:
            raise RuntimeError(
                f"Only {len(chosen)} unique-image specificity examples for {category}; "
                f"need {args.per_category}"
            )
        selected.extend(chosen)

    for index, item in enumerate(selected, 1):
        item["example_id"] = f"refdrone-specificity-{index:03d}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    category_path = args.output_dir / "category_prompt_units.json"
    specificity_path = args.output_dir / "specificity_50_manifest.json"
    locked_path = args.output_dir / "experiment3_locked.json"
    category_path.write_text(json.dumps(category_units, ensure_ascii=False, indent=2), encoding="utf-8")
    specificity_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    locked = {
        "dataset": "RefDrone test",
        "source_annotation": str(source.resolve()),
        "source_image_count": len(unique_files),
        "source_query_count": len(data["images"]),
        "category_id_note": "annotation IDs are VisDrone 1-based; category table is 0-based",
        "annotation_category_mapping": TARGET_ID_TO_CLASS,
        "categories": list(CATEGORIES),
        "synonyms": SYNONYMS,
        "bus_synonym_policy": "not provided; excluded from synonym evaluation and retention",
        "iou_threshold": 0.5,
        "operating_threshold_policy": "reuse each model's frozen Experiment 1 validation threshold",
        "negative_image_policy": "all unique RefDrone test images are evaluated for every category prompt",
        "gt_policy": (
            "For each physical image and thesis category, union and IoU-deduplicate all RefDrone "
            "referent boxes belonging to that category. Layer 2 unions same-color referents; "
            "Layer 3 uses the original query referents."
        ),
        "specificity_selection": {
            "seed": args.seed,
            "examples": len(selected),
            "per_category": dict(Counter(item["category"] for item in selected)),
            "unique_images": len({item["file_name"] for item in selected}),
            "color_attribute_required": True,
            "spatial_or_relational_phrase_required": True,
        },
        "files": {
            "category_units": str(category_path.resolve()),
            "specificity_manifest": str(specificity_path.resolve()),
        },
    }
    locked_path.write_text(json.dumps(locked, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(locked, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
