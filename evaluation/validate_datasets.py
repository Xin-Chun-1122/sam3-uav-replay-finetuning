#!/usr/bin/env python3
"""Dataset validation script: verify images, annotations and statistics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Add evaluation root to path
sys.path.insert(0, str(Path(__file__).parent))

from datasets.fasdd_coco import FASDDCocoDataset
from datasets.visdrone import VisDroneDataset


def validate_fasdd(
    json_path: str,
    images_root: str,
    output_dir: Path,
    train_json: str | None = None,
    val_json: str | None = None,
) -> None:
    print("\n" + "=" * 60)
    print("  FASDD-UAV Validation")
    print("=" * 60)

    dataset = FASDDCocoDataset(json_path=json_path, images_root=images_root)
    stats = dataset.get_stats()

    print(f"\nJSON file:              {json_path}")
    print(f"Images root:            {images_root}")
    print(f"Total images in JSON:   {stats.total_images_in_json}")
    print(f"Found images on disk:   {stats.found_images}")
    print(f"Missing images:         {stats.missing_images}")
    print(f"Duplicate basenames:    {len(stats.duplicate_basenames)}")
    if stats.duplicate_basenames:
        print(f"  (first 5) {stats.duplicate_basenames[:5]}")
    print(f"\nTotal GT annotations:   {stats.total_annotations}")
    for cls, cnt in sorted(stats.per_class_counts.items()):
        print(f"  {cls:20s}: {cnt}")
    print(f"Images without GT:      {stats.images_without_annotations}")

    if stats.missing_images > 0:
        print(f"\nMissing image examples:")
        for name in stats.missing_image_names[:10]:
            print(f"  {name}")

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "FASDD-UAV",
        "json_path": json_path,
        "images_root": images_root,
        "total_images_in_json": stats.total_images_in_json,
        "found_images": stats.found_images,
        "missing_images": stats.missing_images,
        "duplicate_basenames": stats.duplicate_basenames,
        "total_annotations": stats.total_annotations,
        "per_class_counts": stats.per_class_counts,
        "images_without_annotations": stats.images_without_annotations,
        "missing_image_names": stats.missing_image_names,
    }
    rp = output_dir / "fasdd_validation_report.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {rp}")

    # Save COCO GT
    gt_dict = dataset.get_coco_gt_dict()
    gt_path = output_dir / "fasdd_test_gt_coco.json"
    with open(gt_path, "w") as f:
        json.dump(gt_dict, f)
    print(f"COCO GT saved: {gt_path}")

    # Leakage check
    if train_json or val_json:
        leakage_path = output_dir / "data_leakage_report.txt"
        dataset.check_split_leakage(train_json, val_json, str(leakage_path))
        print(f"Leakage report: {leakage_path}")

    if stats.missing_images > 0:
        print(f"\n⚠  WARNING: {stats.missing_images} images not found on disk.")
    if stats.total_annotations == 0:
        print("\n⚠  WARNING: No GT annotations found — check category names in JSON.")
    else:
        print("\n✓ FASDD validation complete.")


def validate_visdrone(
    root: str,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 60)
    print("  VisDrone2019-DET Validation")
    print("=" * 60)

    dataset = VisDroneDataset(root=root)
    stats = dataset.get_stats()

    images_dir = Path(root) / "images"
    ann_dir = Path(root) / "annotations"

    print(f"\nDataset root:               {root}")
    print(f"images/ exists:             {images_dir.exists()}")
    print(f"annotations/ exists:        {ann_dir.exists()}")
    print(f"Total images loaded:        {stats.total_images}")
    print(f"Images with annotations:    {stats.images_with_annotations}")
    print(f"Missing annotation files:   {len(stats.missing_annotation_files)}")
    if stats.missing_annotation_files:
        print(f"  (first 5) {stats.missing_annotation_files[:5]}")
    print(f"\nTotal GT annotations:       {stats.total_annotations}")
    for cls, cnt in sorted(stats.per_class_counts.items()):
        print(f"  {cls:20s}: {cnt}")
    print(f"Images with no target GT:   {stats.images_without_target_annotations}")

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "VisDrone2019-DET",
        "root": root,
        "total_images": stats.total_images,
        "images_with_annotations": stats.images_with_annotations,
        "missing_annotation_files": stats.missing_annotation_files,
        "total_annotations": stats.total_annotations,
        "per_class_counts": stats.per_class_counts,
        "images_without_target_annotations": stats.images_without_target_annotations,
    }
    rp = output_dir / "visdrone_validation_report.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {rp}")

    # Save COCO GT
    gt_dict = dataset.get_coco_gt_dict()
    gt_path = output_dir / "visdrone_test_gt_coco.json"
    with open(gt_path, "w") as f:
        json.dump(gt_dict, f)
    print(f"COCO GT saved: {gt_path}")

    if stats.total_annotations == 0:
        print("\n⚠  WARNING: No target annotations found.")
    else:
        print("\n✓ VisDrone validation complete.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate evaluation datasets.")
    p.add_argument("--fasdd-json", type=str, help="Path to FASDD test.json")
    p.add_argument("--fasdd-images", type=str, help="FASDD images root directory")
    p.add_argument("--fasdd-train-json", type=str, default=None, help="FASDD train.json for leakage check")
    p.add_argument("--fasdd-val-json", type=str, default=None, help="FASDD val.json for leakage check")
    p.add_argument("--visdrone-root", type=str, help="VisDrone dataset root directory")
    p.add_argument("--output-dir", type=str, default="evaluation_results/dataset_validation",
                   help="Output directory for reports")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    ran_any = False

    if args.fasdd_json and args.fasdd_images:
        validate_fasdd(
            json_path=args.fasdd_json,
            images_root=args.fasdd_images,
            output_dir=output_dir / "fasdd",
            train_json=args.fasdd_train_json,
            val_json=args.fasdd_val_json,
        )
        ran_any = True

    if args.visdrone_root:
        validate_visdrone(
            root=args.visdrone_root,
            output_dir=output_dir / "visdrone",
        )
        ran_any = True

    if not ran_any:
        print("Usage: provide --fasdd-json + --fasdd-images and/or --visdrone-root")
        sys.exit(1)


if __name__ == "__main__":
    main()
