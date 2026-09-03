"""COCO AP metrics using pycocotools COCOeval."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class COCOMetrics:
    """Standard COCO detection metrics."""

    ap50: float = 0.0
    ap75: float = 0.0
    map50_95: float = 0.0
    ar: float = 0.0         # AR@100 (max 100 detections)
    ap_small: float = 0.0
    ap_medium: float = 0.0
    ap_large: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "ap50": self.ap50,
            "ap75": self.ap75,
            "map50_95": self.map50_95,
            "ar": self.ar,
            "ap_small": self.ap_small,
            "ap_medium": self.ap_medium,
            "ap_large": self.ap_large,
        }


def compute_coco_metrics(
    gt_dict: Dict,
    predictions: List[Dict],
    iou_type: str = "bbox",
) -> COCOMetrics:
    """
    Compute COCO AP metrics using pycocotools.

    Args:
        gt_dict: COCO-format GT dict with 'images', 'annotations', 'categories'.
        predictions: List of COCO-format prediction dicts:
            [{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}, ...]
        iou_type: 'bbox' or 'segm'.

    Returns:
        COCOMetrics with ap50, ap75, mAP50:95, AR, etc.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        logger.error("pycocotools not available. Install it to compute AP metrics.")
        return COCOMetrics()

    if not predictions:
        logger.warning("No predictions provided — AP will be 0.")
        return COCOMetrics()

    # Filter predictions to only include image_ids present in GT
    gt_image_ids = {img["id"] for img in gt_dict.get("images", [])}
    filtered_preds = [p for p in predictions if p["image_id"] in gt_image_ids]

    if not filtered_preds:
        logger.warning("No predictions match GT image IDs — AP will be 0.")
        return COCOMetrics()

    # Write GT to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as gt_file:
        json.dump(gt_dict, gt_file)
        gt_path = gt_file.name

    try:
        import io, contextlib
        coco_gt = COCO(gt_path)

        # Suppress pycocotools stdout
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(filtered_preds)
            coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        stats = coco_eval.stats
        return COCOMetrics(
            map50_95=float(stats[0]),   # AP  @IoU=0.50:0.95
            ap50=float(stats[1]),       # AP  @IoU=0.50
            ap75=float(stats[2]),       # AP  @IoU=0.75
            ap_small=float(stats[3]),   # AP  small
            ap_medium=float(stats[4]),  # AP  medium
            ap_large=float(stats[5]),   # AP  large
            ar=float(stats[8]),         # AR  @IoU=0.50:0.95, maxDets=100
        )
    except Exception as e:
        logger.exception("COCOeval failed: %s", e)
        return COCOMetrics()
    finally:
        Path(gt_path).unlink(missing_ok=True)


def compute_per_class_coco_metrics(
    gt_dict: Dict,
    predictions: List[Dict],
    category_name_to_id: Dict[str, int],
) -> Dict[str, COCOMetrics]:
    """
    Compute COCO metrics separately per category.

    Returns dict mapping category_name → COCOMetrics.
    """
    results: Dict[str, COCOMetrics] = {}

    for cat_name, cat_id in category_name_to_id.items():
        # Filter GT to only this category
        cat_gt = {
            "images": gt_dict["images"],
            "categories": [c for c in gt_dict["categories"] if c["id"] == cat_id],
            "annotations": [a for a in gt_dict["annotations"] if a["category_id"] == cat_id],
        }
        cat_preds = [p for p in predictions if p["category_id"] == cat_id]

        if not cat_gt["annotations"]:
            logger.warning("No GT annotations for category '%s'", cat_name)
            results[cat_name] = COCOMetrics()
            continue

        results[cat_name] = compute_coco_metrics(cat_gt, cat_preds)

    return results


def build_coco_prediction_entry(
    image_id: int,
    category_id: int,
    bbox_xyxy: np.ndarray,
    score: float,
) -> Dict:
    """Create one COCO detection prediction dict (bbox in xywh format)."""
    x1, y1, x2, y2 = bbox_xyxy
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
        "score": float(score),
    }
