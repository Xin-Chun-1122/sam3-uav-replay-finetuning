#!/usr/bin/env python3
"""
Cross-Dataset Evaluation Script.
Tests the model across all dataset splits and reports per-class + overall AP.

Usage:
  python evaluation/cross_dataset_eval.py \
    --checkpoint /path/to/checkpoint.pt \
    --config configs/training/stage2_multidomain.yaml \
    --datasets uav refcoco \
    --output_dir experiments/stage2_eval/
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add project root and sam3 to path
PROJECT_ROOT = Path(__file__).parents[1]
SAM3_ROOT = PROJECT_ROOT.parent / "sam3"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SAM3_ROOT))


def parse_args():
    parser = argparse.ArgumentParser("Cross-Dataset Evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/training/stage2_multidomain.yaml")
    parser.add_argument("--datasets",   nargs="+", default=["uav", "refcoco"])
    parser.add_argument("--output_dir", default="experiments/stage2_uav_multidomain/eval_results")
    parser.add_argument("--iou_thresholds", nargs="+", type=float, default=[0.5, 0.75])
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--batch_size", type=int, default=2)
    return parser.parse_args()


def compute_ap_at_iou(predictions, ground_truths, iou_threshold: float) -> float:
    """Simplified AP computation. Uses pycocotools under the hood when available."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        # ... full COCO eval pipeline
        return _coco_eval(predictions, ground_truths, iou_threshold)
    except ImportError:
        print("[WARNING] pycocotools not available, using simplified AP")
        return _simple_ap(predictions, ground_truths, iou_threshold)


def _coco_eval(predictions, ground_truths, iou_threshold):
    """Run pycocotools COCO evaluation."""
    import tempfile
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(ground_truths, f)
        gt_file = f.name

    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = [iou_threshold]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[0])


def _simple_ap(predictions, ground_truths, iou_threshold):
    """Simplified AP without pycocotools."""
    return 0.0  # Placeholder - use pycocotools in practice


def format_results_table(all_results: dict) -> str:
    """Format evaluation results as a readable table."""
    lines = [
        "=" * 70,
        f"{'Dataset':<20} {'Split':<10} {'AP@50':<10} {'AP@75':<10} {'mAP':<10}",
        "-" * 70,
    ]
    for ds_name, results in all_results.items():
        for split, metrics in results.items():
            ap50 = metrics.get("AP50", 0.0)
            ap75 = metrics.get("AP75", 0.0)
            map_val = metrics.get("mAP", 0.0)
            lines.append(f"{ds_name:<20} {split:<10} {ap50:<10.4f} {ap75:<10.4f} {map_val:<10.4f}")
    lines.append("=" * 70)
    return "\n".join(lines)


def evaluate_uav_dataset(model, device, batch_size, iou_thresholds):
    """Evaluate on the merged UAV dataset (fire, smoke, car, person)."""
    from datasets.loaders.uav_dataset import UAVDataset

    results = {}
    splits = ["val", "test"]
    for split in splits:
        ann_file_key = "val" if split == "val" else "test"
        ann_file = f"/work/nthujerry123/sam3/datasets/merged/annotations/{ann_file_key}.json"
        img_dir  = "/work/nthujerry123/sam3/datasets/merged/images"

        if not Path(ann_file).exists():
            print(f"[SKIP] {ann_file} not found")
            continue

        print(f"\n[UAV] Evaluating split={split}...")
        dataset = UAVDataset(ann_file=ann_file, img_dir=img_dir, training=False)
        metrics = {f"AP{int(t*100)}": 0.0 for t in iou_thresholds}
        metrics["mAP"] = 0.0
        results[split] = metrics
        print(f"[UAV] {split}: {len(dataset)} images  (metrics placeholder - run full eval with pycocotools)")

    return results


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Cross-Dataset Evaluation")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Datasets:   {args.datasets}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Note: actual model loading requires the SAM3 build environment
    # In practice, this script is run via SLURM on the H200 nodes
    model = None  # Placeholder - replace with actual SAM3-I model loading

    all_results = {}

    if "uav" in args.datasets:
        all_results["UAV"] = evaluate_uav_dataset(model, device, args.batch_size, args.iou_thresholds)

    # Print and save results
    table = format_results_table(all_results)
    print(table)

    result_file = output_dir / "eval_results.json"
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {result_file}")

    # Compare with Stage 1 baseline if available
    baseline = {"fire": 0.894, "smoke": 0.894}  # Stage 1 AP@50
    print(f"\nStage 1 Baseline (fire+smoke): AP@50 = {baseline['fire']:.3f}")
    print(">> Ensure Stage 2 fire/smoke AP@50 >= 0.87 (max 2% regression allowed)")


if __name__ == "__main__":
    main()
