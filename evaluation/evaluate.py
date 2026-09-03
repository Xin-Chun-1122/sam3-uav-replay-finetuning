#!/usr/bin/env python3
"""
SAM 3 評估主程式。

使用範例 (FASDD):
  python evaluation/evaluate.py \
    --dataset fasdd \
    --fasdd-json "/home/alien/Downloads/FASDD_UAV (1)/annotations/COCO_UAV/Annotations/test.json" \
    --fasdd-images "/home/alien/Downloads/FASDD_UAV (1)/images" \
    --base-checkpoint "/home/alien/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt" \
    --finetuned-checkpoint "/home/alien/sam3/finetune_sam3/final_best/sam3_frozen_best.pt" \
    --output-dir "evaluation_results/fasdd" \
    --confidence-threshold 0.30 \
    --iou-threshold 0.50 \
    --nms-threshold 0.50 \
    --resume

使用範例 (VisDrone):
  python evaluation/evaluate.py \
    --dataset visdrone \
    --visdrone-root "/home/alien/Downloads/VisDrone2019-DET-test-dev" \
    --base-checkpoint ... \
    --finetuned-checkpoint ... \
    --output-dir "evaluation_results/visdrone" \
    --resume
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_SAM3_ROOT = Path(__file__).parent.parent
_EVAL_ROOT = Path(__file__).parent
sys.path.insert(0, str(_SAM3_ROOT))
sys.path.insert(0, str(_EVAL_ROOT))

from datasets.fasdd_coco import FASDDCocoDataset, FASDDImage
from datasets.visdrone import VisDroneDataset, VisDroneImage
from metrics.matching import (
    match_detections, classify_image, classify_comparison,
    MatchResult, ImageStatus, ComparisonResult,
)
from metrics.coco_metrics import (
    compute_coco_metrics, compute_per_class_coco_metrics,
    build_coco_prediction_entry, COCOMetrics,
)
from model_interface import SAM3Detector, DetectionResult
from visualization.case_visualizer import make_4col_figure, save_figure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WARMUP_IMAGES = 3


# ── helpers ───────────────────────────────────────────────────────────────────

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@dataclass
class CacheConfig:
    checkpoint_sha256: str
    confidence_threshold: float
    nms_threshold: float
    iou_threshold: float
    prompts: List[str]
    dataset: str
    annotation_sha256: str

    def to_dict(self) -> Dict:
        return asdict(self)

    def matches(self, other: Dict) -> bool:
        return self.to_dict() == other


# ── per-image result ──────────────────────────────────────────────────────────

@dataclass
class ImageResult:
    image_id: int
    image_name: str
    image_path: str
    gt_boxes: np.ndarray
    gt_labels: List[str]
    pred_boxes: np.ndarray
    pred_scores: np.ndarray
    pred_labels: List[str]
    match_result: MatchResult
    image_status: ImageStatus
    inference_time_ms: float

    def to_cache_dict(self) -> Dict:
        return {
            "image_id": self.image_id,
            "image_name": self.image_name,
            "image_path": self.image_path,
            "pred_boxes": self.pred_boxes.tolist(),
            "pred_scores": self.pred_scores.tolist(),
            "pred_labels": self.pred_labels,
            "inference_time_ms": self.inference_time_ms,
        }

    @staticmethod
    def from_cache_dict(
        d: Dict,
        gt_boxes: np.ndarray,
        gt_labels: List[str],
        iou_threshold: float,
    ) -> "ImageResult":
        pred_boxes = np.array(d["pred_boxes"], dtype=np.float32).reshape(-1, 4) if d["pred_boxes"] else np.empty((0, 4), np.float32)
        pred_scores = np.array(d["pred_scores"], dtype=np.float32)
        pred_labels = d["pred_labels"]
        mr = match_detections(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold)
        status = classify_image(mr, len(gt_boxes), len(pred_boxes))
        return ImageResult(
            image_id=d["image_id"],
            image_name=d["image_name"],
            image_path=d["image_path"],
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            match_result=mr,
            image_status=status,
            inference_time_ms=d["inference_time_ms"],
        )


# ── cache manager ─────────────────────────────────────────────────────────────

class PredictionCache:
    """Incremental per-image prediction cache with metadata validation."""

    def __init__(self, cache_dir: Path, config: CacheConfig) -> None:
        self.cache_dir = cache_dir
        self.config = config
        self._meta_path = cache_dir / "cache_meta.json"
        self._data: Dict[str, Dict] = {}
        self._valid = False
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self._meta_path.exists():
            return
        with open(self._meta_path) as f:
            meta = json.load(f)
        if self.config.matches(meta):
            # Load all cached predictions
            for p in self.cache_dir.glob("*.json"):
                if p.name == "cache_meta.json":
                    continue
                with open(p) as f:
                    d = json.load(f)
                self._data[d["image_name"]] = d
            self._valid = True
            logger.info("Loaded cache: %d entries (config matches)", len(self._data))
        else:
            logger.warning("Cache config mismatch → clearing cache")
            for p in self.cache_dir.glob("*.json"):
                p.unlink()
            self._valid = False

    def _save_meta(self) -> None:
        with open(self._meta_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def has(self, image_name: str) -> bool:
        return image_name in self._data

    def get(self, image_name: str) -> Optional[Dict]:
        return self._data.get(image_name)

    def put(self, image_name: str, result: Dict) -> None:
        self._data[image_name] = result
        # Write individual file
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in image_name)
        p = self.cache_dir / f"{safe_name}.json"
        with open(p, "w") as f:
            json.dump(result, f)
        if not self._meta_path.exists():
            self._save_meta()


# ── inference loop ────────────────────────────────────────────────────────────

def run_inference_on_dataset(
    detector: SAM3Detector,
    images: List,  # FASDDImage | VisDroneImage
    prompts: List[str],
    iou_threshold: float,
    confidence_threshold: float,
    nms_threshold: float,
    cache: PredictionCache,
    image_id_fn,    # callable(img) → int
    gt_boxes_fn,    # callable(img) → (np.ndarray, List[str])
    max_images: Optional[int],
    seed: int,
    warmup_paths: List[str],
    error_log: Path,
    output_dir: Path,
) -> List[ImageResult]:
    """Run inference for one model on all images, with cache support."""

    if max_images:
        rng = random.Random(seed)
        images = rng.sample(images, min(max_images, len(images)))

    results: List[ImageResult] = []
    inference_times: List[float] = []

    # Warmup
    warmup_done = False
    for wp in warmup_paths[:WARMUP_IMAGES]:
        if Path(wp).exists():
            logger.info("Warmup: %s", wp)
            try:
                detector.predict(wp, prompts, score_threshold=0.5, warmup=True)
            except Exception as e:
                logger.warning("Warmup failed: %s", e)
            warmup_done = True

    try:
        from tqdm import tqdm
        progress = tqdm(images, desc="Inference", unit="img")
    except ImportError:
        progress = images

    for img in progress:
        img_name = getattr(img, "file_name", None) or getattr(img, "image_name", None)
        img_path = getattr(img, "abs_path", None) or getattr(img, "abs_path", None)
        iid = image_id_fn(img)
        gt_boxes, gt_labels = gt_boxes_fn(img)

        # Cache hit
        if cache.has(img_name):
            cached = cache.get(img_name)
            result = ImageResult.from_cache_dict(cached, gt_boxes, gt_labels, iou_threshold)
            results.append(result)
            inference_times.append(result.inference_time_ms)
            continue

        # Run inference
        try:
            det: DetectionResult = detector.predict(
                image_path=img_path,
                prompts=prompts,
                score_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
            )
        except Exception as e:
            logger.error("Inference error on %s: %s", img_name, e)
            with open(error_log, "a") as f:
                f.write(f"{img_name}\t{e}\n")
            # Empty result
            det = DetectionResult(
                boxes=np.empty((0, 4), np.float32),
                scores=np.empty(0, np.float32),
                labels=[],
                masks=None,
                inference_time_ms=0.0,
                image_path=img_path,
                image_width=0,
                image_height=0,
            )

        mr = match_detections(
            det.boxes, det.scores, det.labels,
            gt_boxes, gt_labels,
            iou_threshold,
        )
        status = classify_image(mr, len(gt_boxes), len(det.boxes))

        result = ImageResult(
            image_id=iid,
            image_name=img_name,
            image_path=img_path,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            pred_boxes=det.boxes,
            pred_scores=det.scores,
            pred_labels=det.labels,
            match_result=mr,
            image_status=status,
            inference_time_ms=det.inference_time_ms,
        )
        results.append(result)

        if det.inference_time_ms > 0:
            inference_times.append(det.inference_time_ms)

        # Save to cache
        cache.put(img_name, result.to_cache_dict())

        # Periodically free unused reserved VRAM to prevent fragmentation
        if len(results) % 50 == 0:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    return results


# ── aggregation ───────────────────────────────────────────────────────────────

@dataclass
class ClassMetrics:
    dataset: str
    model: str
    cls: str
    images: int
    gt: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    miss_rate: float
    object_success_rate: float
    ap50: float
    ap75: float
    map50_95: float
    average_inference_ms: float
    fps: float


def aggregate_class_metrics(
    results: List[ImageResult],
    dataset: str,
    model: str,
    category_names: List[str],
    gt_coco_dict: Dict,
    coco_predictions: List[Dict],
    category_name_to_coco_id: Dict[str, int],
) -> List[ClassMetrics]:
    from collections import defaultdict

    per_class_tp: Dict[str, int] = defaultdict(int)
    per_class_fp: Dict[str, int] = defaultdict(int)
    per_class_fn: Dict[str, int] = defaultdict(int)
    per_class_gt: Dict[str, int] = defaultdict(int)
    per_class_images: Dict[str, set] = {c: set() for c in category_names}

    for r in results:
        for label in r.gt_labels:
            per_class_gt[label] += 1
        for cls in category_names:
            cls_gt = [b for b, l in zip(r.gt_boxes, r.gt_labels) if l == cls]
            cls_pred_boxes = [b for b, l in zip(r.pred_boxes, r.pred_labels) if l == cls]
            cls_pred_scores = [s for s, l in zip(r.pred_scores, r.pred_labels) if l == cls]

            if cls_gt or cls_pred_boxes:
                per_class_images[cls].add(r.image_name)

            cls_gt_arr = np.array(cls_gt, dtype=np.float32).reshape(-1, 4) if cls_gt else np.empty((0, 4), np.float32)
            cls_pred_arr = np.array(cls_pred_boxes, dtype=np.float32).reshape(-1, 4) if cls_pred_boxes else np.empty((0, 4), np.float32)
            cls_scores_arr = np.array(cls_pred_scores, dtype=np.float32)

            mr = match_detections(cls_pred_arr, cls_scores_arr, [cls] * len(cls_pred_boxes),
                                  cls_gt_arr, [cls] * len(cls_gt))
            per_class_tp[cls] += mr.tp
            per_class_fp[cls] += mr.fp
            per_class_fn[cls] += mr.fn

    # COCO AP per class
    coco_per_class = compute_per_class_coco_metrics(gt_coco_dict, coco_predictions, category_name_to_coco_id)

    avg_times = [r.inference_time_ms for r in results if r.inference_time_ms > 0]
    avg_ms = float(np.mean(avg_times)) if avg_times else 0.0

    output = []
    for cls in category_names:
        tp = per_class_tp[cls]
        fp = per_class_fp[cls]
        fn = per_class_fn[cls]
        gt = per_class_gt[cls]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        mr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        osr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        cm = coco_per_class.get(cls, COCOMetrics())

        output.append(ClassMetrics(
            dataset=dataset,
            model=model,
            cls=cls,
            images=len(per_class_images.get(cls, set())),
            gt=gt,
            tp=tp,
            fp=fp,
            fn=fn,
            precision=prec,
            recall=rec,
            f1=f1,
            miss_rate=mr,
            object_success_rate=osr,
            ap50=cm.ap50,
            ap75=cm.ap75,
            map50_95=cm.map50_95,
            average_inference_ms=avg_ms,
            fps=1000.0 / avg_ms if avg_ms > 0 else 0.0,
        ))
    return output


@dataclass
class ImageLevelMetrics:
    dataset: str
    model: str
    total_images: int
    successful_images: int
    partial_failure_images: int
    complete_failure_images: int
    false_alarm_images: int
    negative_images: int
    image_success_rate: float
    false_alarm_rate: float


def aggregate_image_level_metrics(
    results: List[ImageResult],
    dataset: str,
    model: str,
) -> ImageLevelMetrics:
    total = len(results)
    successful = sum(1 for r in results if r.image_status.status == "complete_success")
    partial = sum(1 for r in results if r.image_status.status == "partial_failure")
    complete_fail = sum(1 for r in results if r.image_status.status == "complete_failure")
    false_alarm = sum(1 for r in results if r.image_status.status == "false_alarm")
    negatives = sum(1 for r in results if not r.image_status.has_gt)
    false_alarm_on_neg = sum(1 for r in results
                              if not r.image_status.has_gt and r.image_status.status == "false_alarm")

    return ImageLevelMetrics(
        dataset=dataset,
        model=model,
        total_images=total,
        successful_images=successful,
        partial_failure_images=partial,
        complete_failure_images=complete_fail,
        false_alarm_images=false_alarm,
        negative_images=negatives,
        image_success_rate=successful / total if total > 0 else 0.0,
        false_alarm_rate=false_alarm_on_neg / negatives if negatives > 0 else 0.0,
    )


# ── output writers ────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV: %s", path)


def save_predictions_json(path: Path, predictions: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info("Predictions: %s (%d entries)", path, len(predictions))


# ── case visualization selector ───────────────────────────────────────────────

def select_cases(
    paired_results: List[Tuple[ImageResult, ImageResult, ComparisonResult]],
    max_per_category: int,
    seed: int,
) -> Dict[str, List[Tuple[ImageResult, ImageResult, ComparisonResult]]]:
    """Select visualization cases with diverse representation."""
    rng = random.Random(seed)

    categories: Dict[str, List] = {
        "both_success": [],
        "improved_by_finetuning": [],
        "both_failed": [],
        "finetuning_regression": [],
        "false_alarm": [],
        "small_objects": [],
    }

    for orig_r, ft_r, cmp in paired_results:
        comp = cmp.comparison
        if comp in categories:
            categories[comp].append((orig_r, ft_r, cmp))

        # Small objects
        n_small = sum(
            1 for box in orig_r.gt_boxes
            if (box[2] - box[0]) * (box[3] - box[1]) < 32 * 32
        )
        if n_small > 0:
            categories["small_objects"].append((orig_r, ft_r, cmp))

        # False alarm: add to its own bucket too
        if "false_alarm" in (cmp.original_status, cmp.finetuned_status):
            if (orig_r, ft_r, cmp) not in categories["false_alarm"]:
                categories["false_alarm"].append((orig_r, ft_r, cmp))

    # Sample, prioritizing informative cases
    selected: Dict[str, List] = {}
    for cat, items in categories.items():
        rng.shuffle(items)
        selected[cat] = items[:max_per_category]

    return selected


# ── report generator ──────────────────────────────────────────────────────────

def write_markdown_report(
    path: Path,
    dataset: str,
    class_metrics: List[ClassMetrics],
    image_metrics_orig: ImageLevelMetrics,
    image_metrics_ft: ImageLevelMetrics,
    comparison: Dict,
    args,
) -> None:
    lines = [
        f"# SAM 3 Evaluation Report — {dataset.upper()}\n\n",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Dataset**: {dataset}\n",
        f"**Confidence threshold**: {args.confidence_threshold}\n",
        f"**IoU threshold**: {args.iou_threshold}\n",
        f"**NMS threshold**: {args.nms_threshold}\n",
        f"**Original checkpoint**: {args.base_checkpoint}\n",
        f"**Fine-tuned checkpoint**: {args.finetuned_checkpoint}\n\n",
        "## Object-level Metrics\n\n",
        "| Dataset | Class | Model | GT | TP | FP | FN | Precision | Recall | F1 | AP50 | mAP50:95 | FPS |\n",
        "|---------|-------|-------|----|----|----|----|-----------|--------|----|------|----------|-----|\n",
    ]
    for m in class_metrics:
        lines.append(
            f"| {m.dataset} | {m.cls} | {m.model} | {m.gt} | {m.tp} | {m.fp} | {m.fn} "
            f"| {m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} "
            f"| {m.ap50:.3f} | {m.map50_95:.3f} | {m.fps:.1f} |\n"
        )

    lines += [
        "\n## Image-level Success/Failure\n\n",
        "| Dataset | Model | Total | Success | Partial Fail | Complete Fail | False Alarm | Success Rate | FA Rate |\n",
        "|---------|-------|-------|---------|--------------|---------------|-------------|--------------|---------||\n",
    ]
    for ilm in [image_metrics_orig, image_metrics_ft]:
        lines.append(
            f"| {ilm.dataset} | {ilm.model} | {ilm.total_images} "
            f"| {ilm.successful_images} | {ilm.partial_failure_images} "
            f"| {ilm.complete_failure_images} | {ilm.false_alarm_images} "
            f"| {ilm.image_success_rate:.3f} | {ilm.false_alarm_rate:.3f} |\n"
        )

    if comparison:
        lines += [
            "\n## Fine-tuning Comparison\n\n",
            f"| Both Success | Improved by FT | Both Failed | FT Regression | Net Improvement |\n",
            f"|-------------|----------------|-------------|---------------|-----------------|\n",
            f"| {comparison.get('both_success',0)} | {comparison.get('improved_by_finetuning',0)} "
            f"| {comparison.get('both_failed',0)} | {comparison.get('finetuning_regression',0)} "
            f"| {comparison.get('net_improvement',0)} |\n",
        ]

    lines += ["\n## Notes\n\n",
        "- Threshold was set via CLI, not tuned on test set.\n",
        "- AP computed with pycocotools COCOeval.\n",
        "- TP/FP/FN use IoU ≥ 0.50 with confidence threshold from CLI.\n",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(lines)
    logger.info("Markdown report: %s", path)


def write_latex_table(
    path: Path,
    class_metrics: List[ClassMetrics],
) -> None:
    lines = [
        "% Auto-generated LaTeX table — SAM 3 Evaluation\n",
        "\\begin{table}[htbp]\n",
        "\\centering\n",
        "\\caption{Object-level detection metrics comparison}\n",
        "\\label{tab:object_metrics}\n",
        "\\begin{tabular}{llrrrrrrr}\n",
        "\\hline\n",
        "Dataset & Class & Model & GT & Precision & Recall & F1 & AP@50 & mAP@50:95 \\\\\n",
        "\\hline\n",
    ]
    for m in class_metrics:
        model_str = m.model.replace("_", " ")
        lines.append(
            f"{m.dataset} & {m.cls} & {model_str} & {m.gt} "
            f"& {m.precision:.3f} & {m.recall:.3f} & {m.f1:.3f} "
            f"& {m.ap50:.3f} & {m.map50_95:.3f} \\\\\n"
        )
    lines += [
        "\\hline\n",
        "\\end{tabular}\n",
        "\\end{table}\n",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(lines)
    logger.info("LaTeX table: %s", path)


# ── charts ────────────────────────────────────────────────────────────────────

def generate_charts(
    class_metrics: List[ClassMetrics],
    image_metrics_list: List[ImageLevelMetrics],
    comparison: Dict,
    output_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping charts")
        return

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    orig_metrics = [m for m in class_metrics if "original" in m.model.lower() or "zero" in m.model.lower() or "base" in m.model.lower()]
    ft_metrics = [m for m in class_metrics if m not in orig_metrics]

    # Sort both lists to match class order
    all_classes = sorted({m.cls for m in class_metrics})

    def _get_vals(metrics_list, attr):
        by_cls = {m.cls: getattr(m, attr) for m in metrics_list}
        return [by_cls.get(c, 0.0) for c in all_classes]

    def _bar_chart(title: str, fname: str, metric_attr: str, ylabel: str) -> None:
        if not all_classes:
            return

        x = np.arange(len(all_classes))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))

        if orig_metrics:
            orig_vals = _get_vals(orig_metrics, metric_attr)
            ax.bar(x - width/2 if ft_metrics else x, orig_vals, width, label="Original SAM 3", color="#1f77b4")
        if ft_metrics:
            ft_vals = _get_vals(ft_metrics, metric_attr)
            ax.bar(x + width/2 if orig_metrics else x, ft_vals, width, label="Fine-tuned SAM 3", color="#ff7f0e")

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(all_classes, fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            p = charts_dir / f"{fname}.{suffix}"
            dpi = 300 if suffix == "png" else 150
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    for metric, ylabel, fname in [
        ("precision", "Precision", "precision_comparison"),
        ("recall", "Recall", "recall_comparison"),
        ("f1", "F1-score", "f1_comparison"),
        ("ap50", "AP@50", "ap50_comparison"),
        ("map50_95", "mAP@50:95", "map5095_comparison"),
    ]:
        _bar_chart(f"{metric.upper()} Comparison", fname, metric, ylabel)

    # Fine-tuning outcome distribution
    if not comparison:
        return
    labels = ["Both Success", "Improved", "Both Failed", "Regression"]
    vals = [
        comparison.get("both_success", 0),
        comparison.get("improved_by_finetuning", 0),
        comparison.get("both_failed", 0),
        comparison.get("finetuning_regression", 0),
    ]
    if sum(vals) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ["#2ecc71", "#3498db", "#e74c3c", "#e67e22"]
        ax.bar(labels, vals, color=colors)
        ax.set_ylabel("Number of Images")
        ax.set_title("Fine-tuning Outcome Distribution")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(charts_dir / f"finetuning_outcomes.{suffix}", dpi=300 if suffix == "png" else 150)
        plt.close(fig)

    logger.info("Charts saved to %s", charts_dir)


# ── main evaluation orchestrator ──────────────────────────────────────────────

def run_evaluation(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    error_log = output_dir / "errors.log"
    missing_log = output_dir / "missing_images.log"

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset_name = args.dataset.lower()

    if dataset_name == "fasdd":
        prompts = args.prompts or ["fire", "smoke"]
        logger.info("Loading FASDD dataset...")
        ds = FASDDCocoDataset(
            json_path=args.fasdd_json,
            images_root=args.fasdd_images,
        )
        images_list = ds.images()
        gt_coco_dict = ds.get_coco_gt_dict()
        category_name_to_coco_id = ds.category_name_to_coco_id

        def image_id_fn(img: FASDDImage) -> int:
            return img.image_id

        def gt_boxes_fn(img: FASDDImage):
            boxes = np.array([a.bbox_xyxy for a in img.annotations], dtype=np.float32).reshape(-1, 4)
            labels = [a.category_name for a in img.annotations]
            return boxes, labels

        ann_json_path = args.fasdd_json

    elif dataset_name == "visdrone":
        prompts = args.prompts or ["car", "person"]
        logger.info("Loading VisDrone dataset...")
        ds = VisDroneDataset(root=args.visdrone_root)
        images_list = ds.images()
        gt_coco_dict = ds.get_coco_gt_dict()
        category_name_to_coco_id = ds.category_name_to_coco_id
        id_map = ds.image_id_map

        def image_id_fn(img: VisDroneImage) -> int:
            return id_map[img.image_name]

        def gt_boxes_fn(img: VisDroneImage):
            boxes = np.array([a.bbox_xyxy for a in img.annotations], dtype=np.float32).reshape(-1, 4)
            labels = [a.category_name for a in img.annotations]
            return boxes, labels

        ann_json_path = str(Path(args.visdrone_root) / "annotations")
    else:
        logger.error("Unknown dataset: %s", dataset_name)
        sys.exit(1)

    category_names = sorted(category_name_to_coco_id.keys())

    # Log missing images
    if hasattr(ds, "get_stats"):
        stats = ds.get_stats()
        missing = getattr(stats, "missing_image_names", []) or getattr(stats, "missing_annotation_files", [])
        if missing:
            with open(missing_log, "w") as f:
                f.write("\n".join(missing))

    logger.info("Dataset: %d images, prompts: %s", len(images_list), prompts)

    # Warmup images (first few from dataset)
    warmup_paths = [
        (getattr(img, "abs_path", None) or getattr(img, "abs_path", None))
        for img in images_list[:WARMUP_IMAGES]
        if (getattr(img, "abs_path", None))
    ]

    # ── Cache setup ────────────────────────────────────────────────────────────
    ann_sha256 = sha256_str(ann_json_path)

    def make_cache(model_key: str, ckpt_path: str) -> PredictionCache:
        ckpt_sha = sha256_file(ckpt_path)
        cfg = CacheConfig(
            checkpoint_sha256=ckpt_sha,
            confidence_threshold=args.confidence_threshold,
            nms_threshold=args.nms_threshold,
            iou_threshold=args.iou_threshold,
            prompts=sorted(prompts),
            dataset=dataset_name,
            annotation_sha256=ann_sha256,
        )
        cache_dir = output_dir / "cache" / model_key
        if args.overwrite:
            import shutil
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
        return PredictionCache(cache_dir, cfg)

    # ── Models ────────────────────────────────────────────────────────────────
    run_original = args.model in ("original", "both")
    run_finetuned = args.model in ("finetuned", "both")

    orig_results: List[ImageResult] = []
    ft_results: List[ImageResult] = []

    if run_original:
        logger.info("=== Original SAM 3 ===")
        orig_cache = make_cache("original", args.base_checkpoint)
        orig_detector = SAM3Detector(
            checkpoint_path=args.base_checkpoint,
            device=args.device,
            strict_state_dict_loading=True,
        )
        orig_results = run_inference_on_dataset(
            detector=orig_detector,
            images=images_list,
            prompts=prompts,
            iou_threshold=args.iou_threshold,
            confidence_threshold=args.confidence_threshold,
            nms_threshold=args.nms_threshold,
            cache=orig_cache,
            image_id_fn=image_id_fn,
            gt_boxes_fn=gt_boxes_fn,
            max_images=args.max_images,
            seed=args.seed,
            warmup_paths=warmup_paths,
            error_log=error_log,
            output_dir=output_dir,
        )
        logger.info("Original: processed %d images", len(orig_results))
        orig_detector.free()
        gc.collect()

    if run_finetuned:
        logger.info("=== Fine-tuned SAM 3 ===")
        ft_cache = make_cache("finetuned", args.finetuned_checkpoint)
        ft_detector = SAM3Detector(
            checkpoint_path=args.finetuned_checkpoint,
            device=args.device,
            strict_state_dict_loading=False,
        )
        ft_results = run_inference_on_dataset(
            detector=ft_detector,
            images=images_list,
            prompts=prompts,
            iou_threshold=args.iou_threshold,
            confidence_threshold=args.confidence_threshold,
            nms_threshold=args.nms_threshold,
            cache=ft_cache,
            image_id_fn=image_id_fn,
            gt_boxes_fn=gt_boxes_fn,
            max_images=args.max_images,
            seed=args.seed,
            warmup_paths=warmup_paths,
            error_log=error_log,
            output_dir=output_dir,
        )
        logger.info("Fine-tuned: processed %d images", len(ft_results))
        ft_detector.free()
        gc.collect()

    # ── COCO Predictions JSON ─────────────────────────────────────────────────
    def results_to_coco_preds(results: List[ImageResult]) -> List[Dict]:
        preds = []
        for r in results:
            for box, score, label in zip(r.pred_boxes, r.pred_scores, r.pred_labels):
                cat_id = category_name_to_coco_id.get(label)
                if cat_id is None:
                    continue
                preds.append(build_coco_prediction_entry(r.image_id, cat_id, box, score))
        return preds

    if orig_results:
        orig_preds = results_to_coco_preds(orig_results)
        save_predictions_json(output_dir / "predictions_original.json", orig_preds)
    if ft_results:
        ft_preds = results_to_coco_preds(ft_results)
        save_predictions_json(output_dir / "predictions_finetuned.json", ft_preds)

    # ── Class metrics ─────────────────────────────────────────────────────────
    all_class_metrics: List[ClassMetrics] = []

    if orig_results:
        orig_cm = aggregate_class_metrics(
            orig_results, dataset_name, "original_sam3", category_names,
            gt_coco_dict, orig_preds if orig_results else [], category_name_to_coco_id,
        )
        all_class_metrics.extend(orig_cm)

    if ft_results:
        ft_cm = aggregate_class_metrics(
            ft_results, dataset_name, "finetuned_sam3", category_names,
            gt_coco_dict, ft_preds if ft_results else [], category_name_to_coco_id,
        )
        all_class_metrics.extend(ft_cm)

    if all_class_metrics:
        write_csv(output_dir / "metrics_by_class.csv",
                  [asdict(m) for m in all_class_metrics],
                  list(asdict(all_class_metrics[0]).keys()))

    # ── Image-level metrics ───────────────────────────────────────────────────
    ilm_list = []
    if orig_results:
        orig_ilm = aggregate_image_level_metrics(orig_results, dataset_name, "original_sam3")
        ilm_list.append(orig_ilm)
    if ft_results:
        ft_ilm = aggregate_image_level_metrics(ft_results, dataset_name, "finetuned_sam3")
        ilm_list.append(ft_ilm)

    if ilm_list:
        write_csv(output_dir / "image_level_metrics.csv",
                  [asdict(m) for m in ilm_list],
                  list(asdict(ilm_list[0]).keys()))

    # ── Fine-tuning comparison ────────────────────────────────────────────────
    comparison_results: List[ComparisonResult] = []
    comparison_summary = {}

    if orig_results and ft_results:
        orig_by_name = {r.image_name: r for r in orig_results}
        ft_by_name = {r.image_name: r for r in ft_results}
        common = set(orig_by_name.keys()) & set(ft_by_name.keys())

        for name in sorted(common):
            or_ = orig_by_name[name]
            fr = ft_by_name[name]
            cmp = classify_comparison(name, or_.image_status, fr.image_status)
            comparison_results.append(cmp)

        counts = {"both_success": 0, "improved_by_finetuning": 0,
                  "both_failed": 0, "finetuning_regression": 0}
        for c in comparison_results:
            counts[c.comparison] = counts.get(c.comparison, 0) + 1

        total = len(comparison_results)
        comparison_summary = {
            **counts,
            "net_improvement": counts["improved_by_finetuning"] - counts["finetuning_regression"],
            "improvement_rate": counts["improved_by_finetuning"] / total if total > 0 else 0.0,
            "total_compared": total,
        }

        # Write comparison CSV
        write_csv(output_dir / "finetuning_comparison.csv",
                  [asdict(c) for c in comparison_results],
                  list(asdict(comparison_results[0]).keys()) if comparison_results else [])

        # Write comparison summary CSV
        comp_rows = [{
            "dataset": dataset_name,
            **comparison_summary,
        }]
        write_csv(output_dir / "finetuning_summary.csv", comp_rows, list(comp_rows[0].keys()))

    # ── Per-image results CSV ─────────────────────────────────────────────────
    per_image_rows = []
    for model_tag, results_list in [("original_sam3", orig_results), ("finetuned_sam3", ft_results)]:
        for r in results_list:
            per_image_rows.append({
                "dataset": dataset_name,
                "image_id": r.image_id,
                "image_name": r.image_name,
                "model": model_tag,
                "gt_count": r.image_status.n_gt,
                "prediction_count": r.image_status.n_pred,
                "tp": r.match_result.tp,
                "fp": r.match_result.fp,
                "fn": r.match_result.fn,
                "precision": r.match_result.precision,
                "recall": r.match_result.recall,
                "f1": r.match_result.f1,
                "image_status": r.image_status.status,
                "inference_time_ms": r.inference_time_ms,
            })

    if per_image_rows:
        write_csv(output_dir / "per_image_results.csv", per_image_rows, list(per_image_rows[0].keys()))

    # ── Visualizations ────────────────────────────────────────────────────────
    if not args.no_visualizations and orig_results and ft_results:
        logger.info("Generating case visualizations...")
        cases_dir = output_dir / "cases"
        case_index_rows = []

        orig_by_name = {r.image_name: r for r in orig_results}
        ft_by_name = {r.image_name: r for r in ft_results}

        paired = []
        for cr in comparison_results:
            name = cr.image_name
            if name in orig_by_name and name in ft_by_name:
                paired.append((orig_by_name[name], ft_by_name[name], cr))

        selected = select_cases(paired, args.max_case_images, args.seed)

        for category, triplets in selected.items():
            cat_dir = cases_dir / category
            for idx, (orig_r, ft_r, cr) in enumerate(triplets):
                if not orig_r.image_path or not Path(orig_r.image_path).exists():
                    continue

                fig = make_4col_figure(
                    image_path=orig_r.image_path,
                    gt_boxes=orig_r.gt_boxes,
                    gt_labels=orig_r.gt_labels,
                    orig_boxes=orig_r.pred_boxes,
                    orig_scores=orig_r.pred_scores,
                    orig_labels=orig_r.pred_labels,
                    orig_pred_to_gt=orig_r.match_result.pred_to_gt,
                    orig_matched_gt=orig_r.match_result.matched_gt_indices,
                    ft_boxes=ft_r.pred_boxes,
                    ft_scores=ft_r.pred_scores,
                    ft_labels=ft_r.pred_labels,
                    ft_pred_to_gt=ft_r.match_result.pred_to_gt,
                    ft_matched_gt=ft_r.match_result.matched_gt_indices,
                )
                if fig is None:
                    continue

                out_path = str(cat_dir / f"{idx:03d}_{orig_r.image_name}")
                save_figure(fig, out_path)

                n_small = sum(
                    1 for box in orig_r.gt_boxes
                    if (box[2] - box[0]) * (box[3] - box[1]) < 32 * 32
                )
                case_index_rows.append({
                    "dataset": dataset_name,
                    "image_name": orig_r.image_name,
                    "comparison_category": category,
                    "original_tp": cr.original_tp,
                    "original_fp": cr.original_fp,
                    "original_fn": cr.original_fn,
                    "finetuned_tp": cr.finetuned_tp,
                    "finetuned_fp": cr.finetuned_fp,
                    "finetuned_fn": cr.finetuned_fn,
                    "gt_count": cr.n_gt,
                    "gt_small_count": n_small,
                    "saved_figure_path": out_path,
                })

        if case_index_rows:
            write_csv(output_dir / "case_index.csv", case_index_rows, list(case_index_rows[0].keys()))

    # ── Charts ────────────────────────────────────────────────────────────────
    if all_class_metrics:
        generate_charts(all_class_metrics, ilm_list, comparison_summary, output_dir)

    # ── Reports ───────────────────────────────────────────────────────────────
    write_markdown_report(
        output_dir / "evaluation_report.md",
        dataset=dataset_name,
        class_metrics=all_class_metrics,
        image_metrics_orig=ilm_list[0] if len(ilm_list) > 0 else ImageLevelMetrics(dataset_name, "original_sam3", 0,0,0,0,0,0,0.0,0.0),
        image_metrics_ft=ilm_list[1] if len(ilm_list) > 1 else ImageLevelMetrics(dataset_name, "finetuned_sam3", 0,0,0,0,0,0,0.0,0.0),
        comparison=comparison_summary,
        args=args,
    )
    if all_class_metrics:
        write_latex_table(output_dir / "evaluation_report.tex", all_class_metrics)

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "dataset": dataset_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_images_evaluated": len(orig_results) or len(ft_results),
        "prompts": prompts,
        "confidence_threshold": args.confidence_threshold,
        "iou_threshold": args.iou_threshold,
        "nms_threshold": args.nms_threshold,
        "base_checkpoint": args.base_checkpoint,
        "finetuned_checkpoint": args.finetuned_checkpoint,
        "class_metrics": [asdict(m) for m in all_class_metrics],
        "image_level_metrics": [asdict(m) for m in ilm_list],
        "finetuning_comparison": comparison_summary,
    }
    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  Evaluation Complete — {dataset_name.upper()}")
    print("=" * 70)
    for m in all_class_metrics:
        print(f"  {m.model:20s} [{m.cls:8s}]  P={m.precision:.3f}  R={m.recall:.3f}  "
              f"F1={m.f1:.3f}  AP50={m.ap50:.3f}  mAP={m.map50_95:.3f}")
    if comparison_summary:
        print(f"\n  Fine-tuning Comparison:")
        print(f"    Both Success:     {comparison_summary.get('both_success', 0)}")
        print(f"    Improved by FT:   {comparison_summary.get('improved_by_finetuning', 0)}")
        print(f"    Both Failed:      {comparison_summary.get('both_failed', 0)}")
        print(f"    FT Regression:    {comparison_summary.get('finetuning_regression', 0)}")
        print(f"    Net Improvement:  {comparison_summary.get('net_improvement', 0)}")
    print(f"\n  Results: {output_dir}")
    print("=" * 70)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAM 3 evaluation: Original vs Fine-tuned",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset
    p.add_argument("--dataset", required=True, choices=["fasdd", "visdrone"],
                   help="Dataset to evaluate")
    p.add_argument("--fasdd-json", type=str, help="FASDD test.json path")
    p.add_argument("--fasdd-images", type=str, help="FASDD images root")
    p.add_argument("--visdrone-root", type=str, help="VisDrone dataset root")

    # Model checkpoints
    p.add_argument("--base-checkpoint", required=True,
                   help="Original SAM 3 checkpoint path")
    p.add_argument("--finetuned-checkpoint", required=True,
                   help="Fine-tuned SAM 3 checkpoint path")
    p.add_argument("--model", choices=["original", "finetuned", "both"], default="both",
                   help="Which model(s) to run")

    # Output
    p.add_argument("--output-dir", default="evaluation_results",
                   help="Output directory")

    # Thresholds
    p.add_argument("--confidence-threshold", type=float, default=0.30,
                   help="Detection confidence threshold")
    p.add_argument("--iou-threshold", type=float, default=0.50,
                   help="IoU threshold for TP matching")
    p.add_argument("--nms-threshold", type=float, default=0.50,
                   help="NMS IoU threshold")

    # Prompts
    p.add_argument("--prompts", nargs="+", default=None,
                   help="Text prompts (default: fire smoke for FASDD, car person for VisDrone)")

    # Run control
    p.add_argument("--max-images", type=int, default=None,
                   help="Limit number of images (for smoke/small test)")
    p.add_argument("--image-list", type=str, default=None,
                   help="File with image names to evaluate (one per line)")
    p.add_argument("--max-case-images", type=int, default=20,
                   help="Max visualizations per comparison category")
    p.add_argument("--device", default="cuda", help="Compute device")
    p.add_argument("--image-size", type=int, default=1008, help="Input image size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Cache
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing predictions cache")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing cache")

    # Visualization
    p.add_argument("--no-visualizations", action="store_true",
                   help="Skip case visualizations")
    p.add_argument("--save-all-visualizations", action="store_true",
                   help="Save visualizations for all images (slow)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset == "fasdd" and (not args.fasdd_json or not args.fasdd_images):
        logger.error("--fasdd-json and --fasdd-images are required for FASDD dataset")
        sys.exit(1)
    if args.dataset == "visdrone" and not args.visdrone_root:
        logger.error("--visdrone-root is required for VisDrone dataset")
        sys.exit(1)

    run_evaluation(args)


if __name__ == "__main__":
    main()
