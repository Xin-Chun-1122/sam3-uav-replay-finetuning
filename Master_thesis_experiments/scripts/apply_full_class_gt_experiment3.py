#!/usr/bin/env python3
"""Replace broad category GT with exhaustive VisDrone GT without changing frozen prompts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


CATEGORIES = {"person", "car", "truck", "bus", "motorcycle"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances-csv", type=Path, required=True)
    parser.add_argument("--category-units", type=Path, required=True)
    parser.add_argument("--specificity-manifest", type=Path, required=True)
    parser.add_argument("--locked-config", type=Path, required=True)
    args = parser.parse_args()

    boxes: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    with args.instances_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            category = row["category"]
            if category not in CATEGORIES:
                continue
            x = float(row["bbox_left"])
            y = float(row["bbox_top"])
            w = float(row["bbox_width"])
            h = float(row["bbox_height"])
            boxes[(row["image_name"], category)].append([x, y, x + w, y + h])

    units = json.loads(args.category_units.read_text(encoding="utf-8"))
    for unit in units:
        unit["gt_boxes_xyxy"] = boxes.get((unit["file_name"], unit["category"]), [])
        unit["gt_source"] = "VisDrone2019-DET-test-dev exhaustive evaluation GT"
    args.category_units.write_text(json.dumps(units, ensure_ascii=False, indent=2), encoding="utf-8")

    specificity = json.loads(args.specificity_manifest.read_text(encoding="utf-8"))
    for example in specificity:
        layer1 = example["layers"][0]
        if int(layer1["layer"]) != 1:
            raise ValueError(f"{example['example_id']} does not begin with layer 1")
        layer1["gt_boxes_xyxy"] = boxes.get((example["file_name"], example["category"]), [])
        layer1["gt_source"] = "VisDrone2019-DET-test-dev exhaustive evaluation GT"
        for layer in example["layers"][1:]:
            layer["gt_source"] = "RefDrone referring-expression GT"
    args.specificity_manifest.write_text(
        json.dumps(specificity, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    locked = json.loads(args.locked_config.read_text(encoding="utf-8"))
    locked["gt_policy"] = (
        "Standard/synonym prompts and specificity Layer 1 use exhaustive VisDrone2019-DET "
        "evaluation GT for the same byte-identical images. Layers 2 and 3 use RefDrone "
        "attribute/referent GT. The already-frozen 50 images and prompts are unchanged."
    )
    locked["full_category_gt_source"] = str(args.instances_csv.resolve())
    locked["frozen_specificity_examples_preserved"] = True
    args.locked_config.write_text(json.dumps(locked, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = defaultdict(int)
    for unit in units:
        counts[unit["category"]] += len(unit["gt_boxes_xyxy"])
    print(json.dumps({"category_gt_counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
