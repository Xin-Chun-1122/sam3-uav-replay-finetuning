"""SAM3Detector: unified interface wrapping Sam3VideoPredictor for image evaluation."""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Add sam3 root to path
_SAM3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_SAM3_ROOT))


@dataclass
class DetectionResult:
    """Unified detection output for one image."""

    boxes: np.ndarray          # Nx4, xyxy pixel coords
    scores: np.ndarray         # N confidence scores
    labels: List[str]          # N class label strings
    masks: Optional[np.ndarray]  # NxHxW binary, may be None
    inference_time_ms: float
    image_path: str
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        n = len(self.scores)
        assert self.boxes.shape == (n, 4) or (n == 0 and self.boxes.shape == (0, 4)), (
            f"boxes shape {self.boxes.shape} inconsistent with {n} scores"
        )
        assert len(self.labels) == n


def _boxes_xywh_norm_to_xyxy_pixel(
    boxes_xywh: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Convert model output [x_left_norm, y_top_norm, w_norm, h_norm] → xyxy pixel."""
    if len(boxes_xywh) == 0:
        return np.empty((0, 4), dtype=np.float32)
    b = boxes_xywh.copy().astype(np.float32)
    x1 = b[:, 0] * img_w
    y1 = b[:, 1] * img_h
    x2 = (b[:, 0] + b[:, 2]) * img_w
    y2 = (b[:, 1] + b[:, 3]) * img_h
    # clamp to image boundary
    x1 = np.clip(x1, 0, img_w)
    y1 = np.clip(y1, 0, img_h)
    x2 = np.clip(x2, 0, img_w)
    y2 = np.clip(y2, 0, img_h)
    return np.stack([x1, y1, x2, y2], axis=1)


def _class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: List[str],
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Apply NMS separately within each class."""
    if len(scores) == 0:
        return boxes, scores, labels

    unique_classes = list(dict.fromkeys(labels))
    keep_boxes, keep_scores, keep_labels = [], [], []

    for cls in unique_classes:
        idx = [i for i, l in enumerate(labels) if l == cls]
        if not idx:
            continue
        cls_boxes = boxes[idx]
        cls_scores = scores[idx]

        order = cls_scores.argsort()[::-1]
        suppressed = set()
        for rank, i in enumerate(order):
            if i in suppressed:
                continue
            keep_boxes.append(cls_boxes[i])
            keep_scores.append(cls_scores[i])
            keep_labels.append(cls)
            bx = cls_boxes[i]
            for j in order[rank + 1 :]:
                if j in suppressed:
                    continue
                bj = cls_boxes[j]
                ix1 = max(bx[0], bj[0])
                iy1 = max(bx[1], bj[1])
                ix2 = min(bx[2], bj[2])
                iy2 = min(bx[3], bj[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                area_bx = max(0.0, bx[2] - bx[0]) * max(0.0, bx[3] - bx[1])
                area_bj = max(0.0, bj[2] - bj[0]) * max(0.0, bj[3] - bj[1])
                union = area_bx + area_bj - inter
                if union > 0 and inter / union > iou_threshold:
                    suppressed.add(j)

    if not keep_boxes:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), []

    return (
        np.array(keep_boxes, dtype=np.float32),
        np.array(keep_scores, dtype=np.float32),
        keep_labels,
    )


def _parse_propagate_output(
    outputs: Dict,
    img_w: int,
    img_h: int,
    score_threshold: float,
    prompt_label: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Parse one propagate_in_video output dict for a single prompt."""
    if not outputs or "out_obj_ids" not in outputs:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), None

    obj_ids = outputs["out_obj_ids"]
    probs = outputs.get("out_probs", None)
    boxes_xywh = outputs.get("out_boxes_xywh", None)
    binary_masks = outputs.get("out_binary_masks", None)

    if obj_ids is None or len(obj_ids) == 0:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), None

    n = len(obj_ids)
    keep_boxes, keep_scores, keep_mask_list = [], [], []

    for i in range(n):
        score = float(probs[i]) if probs is not None and i < len(probs) else 1.0
        if score < score_threshold:
            continue

        if boxes_xywh is not None and i < len(boxes_xywh):
            bxywh = np.array(boxes_xywh[i], dtype=np.float32)
        else:
            # Fall back to deriving from mask
            if binary_masks is None or i >= len(binary_masks):
                continue
            m = binary_masks[i]
            if isinstance(m, torch.Tensor):
                m = m.squeeze().cpu().numpy()
            else:
                m = np.array(m).squeeze()
            if not m.any():
                continue
            ys, xs = np.where(m)
            x1p, y1p, x2p, y2p = xs.min(), ys.min(), xs.max(), ys.max()
            bxywh = np.array(
                [x1p / img_w, y1p / img_h, (x2p - x1p) / img_w, (y2p - y1p) / img_h],
                dtype=np.float32,
            )

        # Convert to pixel xyxy
        xyxy = _boxes_xywh_norm_to_xyxy_pixel(bxywh[None], img_w, img_h)[0]

        # Skip degenerate boxes
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            continue
        if np.any(np.isnan(xyxy)) or np.any(np.isinf(xyxy)):
            continue

        keep_boxes.append(xyxy)
        keep_scores.append(score)

        if binary_masks is not None and i < len(binary_masks):
            m = binary_masks[i]
            if isinstance(m, torch.Tensor):
                m = m.squeeze().cpu().numpy().astype(bool)
            else:
                m = np.array(m, dtype=bool).squeeze()
            if m.shape == (img_h, img_w):
                keep_mask_list.append(m)

    if not keep_boxes:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), None

    boxes_out = np.array(keep_boxes, dtype=np.float32)
    scores_out = np.array(keep_scores, dtype=np.float32)
    masks_out = np.stack(keep_mask_list, axis=0) if len(keep_mask_list) == len(keep_boxes) else None

    return boxes_out, scores_out, masks_out


MAX_INFERENCE_SIDE = 1920   # pixels; larger images are resized to this before inference


class SAM3Detector:
    """
    Unified wrapper around Sam3VideoPredictor for per-image evaluation.

    Runs one text prompt per session (separate sessions per prompt) to avoid
    obj_id collisions and ensure clean per-class outputs.

    Images larger than max_inference_side are proportionally resized before
    writing the fake video. Because out_boxes_xywh is normalized [0,1], the
    predicted bounding boxes remain correct in the original image coordinate space.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        strict_state_dict_loading: bool = False,
        apply_temporal_disambiguation: bool = False,
        max_inference_side: int = MAX_INFERENCE_SIDE,
    ) -> None:
        self.max_inference_side = max_inference_side
        self.checkpoint_path = str(checkpoint_path)
        self.device = device

        logger.info("Loading SAM3 checkpoint: %s", self.checkpoint_path)
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor

        self._predictor = Sam3VideoPredictor(
            checkpoint_path=self.checkpoint_path,
            strict_state_dict_loading=strict_state_dict_loading,
            apply_temporal_disambiguation=apply_temporal_disambiguation,
        )
        # score_threshold_detection is set dynamically in predict() to match the
        # user's confidence threshold, because it directly controls NMS candidate
        # filtering: nms_prob_thresh = score_threshold_detection.  With a low value
        # (e.g., 0.0), the detector can produce 150+ candidates per image, and the
        # NMS IoU matrix [N, N, H*W] causes OOM (19+ GiB on a 720p image).
        inner = self._predictor.model
        inner.hotstart_delay = 0
        inner.max_num_objects = 32    # hard cap on tracked objects per session
        self._inner = inner
        logger.info("SAM3Detector ready (device=%s)", device)

    @property
    def checkpoint_sha256(self) -> str:
        h = hashlib.sha256()
        with open(self.checkpoint_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    @torch.inference_mode()
    def predict(
        self,
        image_path: str,
        prompts: List[str],
        score_threshold: float = 0.25,
        nms_threshold: float = 0.50,
        return_masks: bool = False,
        warmup: bool = False,
    ) -> DetectionResult:
        """
        Run detection on a single image.

        Args:
            image_path: Path to the image file.
            prompts: List of text prompts (e.g. ['fire', 'smoke']).
            score_threshold: Minimum detection confidence.
            nms_threshold: IoU threshold for class-aware NMS.
            return_masks: Whether to return binary masks.
            warmup: If True, don't count time in benchmarks (still returns result).

        Returns:
            DetectionResult with xyxy pixel boxes, scores, labels.
        """
        # Set detection threshold before each inference to control NMS candidate count.
        # nms_prob_thresh = score_threshold_detection, so this prevents [N×N×H×W] OOM.
        self._inner.score_threshold_detection = max(0.01, score_threshold * 0.9)
        self._inner.new_det_thresh = max(0.01, score_threshold * 0.9)

        frame_bgr = cv2.imread(image_path)
        if frame_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        orig_h, orig_w = frame_bgr.shape[:2]

        # Resize large images to avoid OOM (boxes remain in original coord space
        # because out_boxes_xywh is normalized by the video dimensions)
        if max(orig_w, orig_h) > self.max_inference_side:
            scale = self.max_inference_side / max(orig_w, orig_h)
            vid_w = int(orig_w * scale)
            vid_h = int(orig_h * scale)
            frame_bgr = cv2.resize(frame_bgr, (vid_w, vid_h))
        else:
            vid_w, vid_h = orig_w, orig_h

        all_boxes: List[np.ndarray] = []
        all_scores: List[float] = []
        all_labels: List[str] = []
        all_masks: List[np.ndarray] = []

        tmp_path = None
        t_start = time.perf_counter()

        try:
            tmp_fd, tmp_path = tempfile.mkstemp(prefix="sam3eval_", suffix=".mp4")
            os.close(tmp_fd)

            writer = cv2.VideoWriter(
                tmp_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                1,
                (vid_w, vid_h),
            )
            writer.write(frame_bgr)
            writer.write(frame_bgr)
            writer.release()

            for prompt in prompts:
                sid = None
                try:
                    sess_resp = self._predictor.handle_request({
                        "type": "start_session",
                        "resource_path": tmp_path,
                    })
                    sid = sess_resp["session_id"]

                    self._predictor.handle_request({
                        "type": "add_prompt",
                        "session_id": sid,
                        "frame_index": 0,
                        "text": prompt,
                        "obj_id": 1,
                    })

                    results = list(self._predictor.handle_stream_request({
                        "type": "propagate_in_video",
                        "session_id": sid,
                        "propagation_direction": "forward",
                        "start_frame_index": 0,
                        "max_frame_num_to_track": 1,
                    }))

                    for r in results:
                        if r.get("frame_index", -1) != 0:
                            continue
                        # Parse with orig dims: normalized boxes map to original coords
                        b, s, m = _parse_propagate_output(
                            r.get("outputs", {}),
                            orig_w,
                            orig_h,
                            score_threshold,
                            prompt,
                        )
                        all_boxes.append(b)
                        all_scores.extend(s.tolist())
                        all_labels.extend([prompt] * len(s))
                        if return_masks and m is not None:
                            all_masks.append(m)
                finally:
                    if sid is not None:
                        try:
                            self._predictor.handle_request({
                                "type": "close_session",
                                "session_id": sid,
                            })
                        except Exception:
                            pass

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        inference_time_ms = (t_end - t_start) * 1000.0

        # Merge all prompts
        if all_boxes:
            merged_boxes = np.concatenate(all_boxes, axis=0)
        else:
            merged_boxes = np.empty((0, 4), dtype=np.float32)
        merged_scores = np.array(all_scores, dtype=np.float32)

        # Class-aware NMS
        if len(merged_scores) > 0:
            merged_boxes, merged_scores, all_labels = _class_aware_nms(
                merged_boxes, merged_scores, all_labels, nms_threshold
            )

        masks_out: Optional[np.ndarray] = None
        if return_masks and all_masks:
            # masks per class-aware NMS is complex — skip for now
            masks_out = None

        return DetectionResult(
            boxes=merged_boxes,
            scores=merged_scores,
            labels=all_labels,
            masks=masks_out,
            inference_time_ms=inference_time_ms,
            image_path=image_path,
            image_width=orig_w,
            image_height=orig_h,
        )

    def warmup(self, image_path: str, prompts: List[str]) -> None:
        """Run a warmup pass (result discarded)."""
        logger.info("Warmup inference on: %s", image_path)
        self.predict(image_path, prompts, score_threshold=0.5, warmup=True)

    def free(self) -> None:
        """Release GPU memory."""
        del self._predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
