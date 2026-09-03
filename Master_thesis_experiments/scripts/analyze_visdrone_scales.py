#!/usr/bin/env python3
"""Create reproducible scale statistics/manifests for VisDrone-DET."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


CATEGORY_MAP = {
    1: "person",       # pedestrian
    2: "person",       # people
    4: "car",
    5: "van",
    6: "truck",
    9: "bus",
    10: "motorcycle",  # VisDrone name: motor
}
SCALES = ("tiny", "small", "regular")


def scale_name(width: float, height: float) -> str:
    s = math.sqrt(width * height)
    if s < 32.0:
        return "tiny"
    if s < 64.0:
        return "small"
    return "regular"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    images_dir = root / "images"
    annotations_dir = root / "annotations"
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise SystemExit(f"Expected images/ and annotations/ under {root}")

    image_paths = sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.png")])
    if not image_paths:
        raise SystemExit(f"No images found under {images_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)

    instance_counts = Counter({scale: 0 for scale in SCALES})
    image_counts = Counter({scale: 0 for scale in SCALES})
    class_scale_counts: dict[str, Counter] = defaultdict(Counter)
    membership_rows: list[dict] = []
    instance_rows: list[dict] = []
    manifests: dict[str, list[str]] = {scale: [] for scale in SCALES}
    missing_annotations: list[str] = []
    malformed_lines = 0
    boundary_counts = Counter({"s_eq_32": 0, "s_eq_64": 0})

    for image_path in image_paths:
        ann_path = annotations_dir / f"{image_path.stem}.txt"
        present_scales: set[str] = set()
        per_image_counts = Counter({scale: 0 for scale in SCALES})
        if not ann_path.exists():
            missing_annotations.append(image_path.name)
        else:
            with ann_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    parts = line.strip().split(",")
                    if len(parts) < 8:
                        if line.strip():
                            malformed_lines += 1
                        continue
                    try:
                        left, top, width, height = map(float, parts[:4])
                        score, category_id = int(parts[4]), int(parts[5])
                        truncation, occlusion = int(parts[6]), int(parts[7])
                    except ValueError:
                        malformed_lines += 1
                        continue
                    if score == 0 or category_id not in CATEGORY_MAP or width <= 0 or height <= 0:
                        continue
                    size = math.sqrt(width * height)
                    if math.isclose(size, 32.0, rel_tol=0.0, abs_tol=1e-12):
                        boundary_counts["s_eq_32"] += 1
                    if math.isclose(size, 64.0, rel_tol=0.0, abs_tol=1e-12):
                        boundary_counts["s_eq_64"] += 1
                    scale = scale_name(width, height)
                    category = CATEGORY_MAP[category_id]
                    present_scales.add(scale)
                    per_image_counts[scale] += 1
                    instance_counts[scale] += 1
                    class_scale_counts[category][scale] += 1
                    instance_rows.append({
                        "image_name": image_path.name,
                        "annotation_line": line_number,
                        "category": category,
                        "visdrone_category_id": category_id,
                        "bbox_left": left,
                        "bbox_top": top,
                        "bbox_width": width,
                        "bbox_height": height,
                        "scale_s": f"{size:.6f}",
                        "scale": scale,
                        "truncation": truncation,
                        "occlusion": occlusion,
                    })
        for scale in SCALES:
            contains = scale in present_scales
            if contains:
                image_counts[scale] += 1
                manifests[scale].append(image_path.name)
        membership_rows.append({
            "image_name": image_path.name,
            "contains_tiny": int("tiny" in present_scales),
            "contains_small": int("small" in present_scales),
            "contains_regular": int("regular" in present_scales),
            "tiny_instances": per_image_counts["tiny"],
            "small_instances": per_image_counts["small"],
            "regular_instances": per_image_counts["regular"],
            "target_instances": sum(per_image_counts.values()),
        })

    for scale, names in manifests.items():
        (args.manifest_dir / f"visdrone_det_{scale}_images.txt").write_text(
            "".join(f"{name}\n" for name in names), encoding="utf-8"
        )

    scale_rows = [
        {"scale": scale, "images_containing_scale": image_counts[scale], "instances": instance_counts[scale]}
        for scale in SCALES
    ]
    class_rows = [
        {"category": category, **{scale: class_scale_counts[category][scale] for scale in SCALES},
         "total": sum(class_scale_counts[category][scale] for scale in SCALES)}
        for category in ("car", "person", "bus", "van", "truck", "motorcycle")
    ]
    write_csv(args.output_dir / "scale_counts.csv", ["scale", "images_containing_scale", "instances"], scale_rows)
    write_csv(args.output_dir / "class_scale_counts.csv", ["category", *SCALES, "total"], class_rows)
    write_csv(args.output_dir / "image_scale_membership.csv", list(membership_rows[0]), membership_rows)
    write_csv(args.output_dir / "instances.csv", list(instance_rows[0]), instance_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "dataset_image_count": len(image_paths),
        "missing_annotation_count": len(missing_annotations),
        "missing_annotations": missing_annotations,
        "malformed_annotation_lines": malformed_lines,
        "target_categories": ["car", "person", "bus", "van", "truck", "motorcycle"],
        "category_mapping": {str(key): value for key, value in CATEGORY_MAP.items()},
        "scale_formula": "sqrt(width*height)",
        "scale_ranges": {"tiny": "s < 32", "small": "32 <= s < 64", "regular": "s >= 64"},
        "image_count_semantics": "number of unique images containing at least one target instance of the scale; scale image sets may overlap",
        "images_containing_scale": dict(image_counts),
        "instances_by_scale": dict(instance_counts),
        "boundary_instance_counts": dict(boundary_counts),
        "total_target_instances": sum(instance_counts.values()),
        "annotation_manifest_sha256": sha256(args.output_dir / "instances.csv"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
