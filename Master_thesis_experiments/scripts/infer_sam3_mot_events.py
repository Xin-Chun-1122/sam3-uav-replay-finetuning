#!/usr/bin/env python3
"""Run SAM 3 native-ID tracking on selected VisDrone-MOT event clips."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def install_cpu_mask_iou_patch() -> None:
    """Avoid quadratic GPU-memory spikes in dense UAV scenes."""
    def cpu_mask_iou(pred, gt, chunk=32):
        n_a, n_b = pred.shape[0], gt.shape[0]
        device = pred.device
        pa = pred.bool().cpu().reshape(n_a, -1).float()
        pb = gt.bool().cpu().reshape(n_b, -1).float()
        a_areas, b_areas = pa.sum(1), pb.sum(1)
        result = torch.zeros(n_a, n_b)
        for start in range(0, n_a, chunk):
            current = pa[start:start + chunk]
            intersection = torch.mm(current, pb.t())
            union = a_areas[start:start + chunk, None] + b_areas[None, :] - intersection
            result[start:start + chunk] = intersection / union.clamp(min=1.0)
        return result.to(device)

    import sam3.perflib.masks_ops as masks_ops
    import sam3.perflib.nms as nms
    import sam3.model.sam3_video_base as video_base
    import sam3.perflib.associate_det_trk as associate
    masks_ops.mask_iou = cpu_mask_iou
    nms.mask_iou = cpu_mask_iou
    video_base.mask_iou = cpu_mask_iou
    associate.mask_iou = cpu_mask_iou


def read_gt(path: Path) -> dict[int, dict[int, dict]]:
    by_target: dict[int, dict[int, dict]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 10:
            continue
        frame, target_id = int(parts[0]), int(parts[1])
        left, top, width, height = map(float, parts[2:6])
        by_target[target_id][frame] = {
            "bbox_xywh": [left, top, width, height],
            "category_id": int(parts[7]),
            "truncation": int(parts[8]),
            "occlusion": int(parts[9]),
        }
    return by_target


def frame_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as image:
        return image.size


def box_iou(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def motion_blur(image: np.ndarray, kernel_size: int, angle_degrees: float) -> np.ndarray:
    """Apply deterministic linear motion blur without changing image geometry."""
    if kernel_size <= 1:
        return image
    if kernel_size % 2 == 0:
        raise ValueError("motion-blur kernel size must be odd")
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    center = (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5)
    rotation = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (kernel_size, kernel_size))
    kernel /= max(float(kernel.sum()), 1e-12)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT101)


def prepare_clip(
    sequence_dir: Path,
    start: int,
    end: int,
    max_side: int,
    event: dict | None = None,
) -> Path:
    temporary = Path(tempfile.mkdtemp(prefix="sam3_mot_event_"))
    for local_index, global_frame in enumerate(range(start, end + 1)):
        source = sequence_dir / f"{global_frame:07d}.jpg"
        if not source.is_file():
            # Some mirrors use six digits; preserve deterministic local ordering.
            matches = sorted(sequence_dir.glob(f"*{global_frame}.jpg"))
            if not matches:
                shutil.rmtree(temporary)
                raise FileNotFoundError(source)
            source = matches[0]
        image = cv2.imread(str(source))
        if image is None:
            shutil.rmtree(temporary)
            raise RuntimeError(f"Cannot decode {source}")
        if event is not None and event.get("synthetic_motion_blur_kernel", 0) > 1:
            blur_entire_clip = bool(event.get("synthetic_motion_blur_entire_clip", False))
            if blur_entire_clip or event["start_frame"] <= global_frame <= event["end_frame"]:
                image = motion_blur(
                    image,
                    int(event["synthetic_motion_blur_kernel"]),
                    float(event.get("synthetic_motion_blur_angle_degrees", 0.0)),
                )
        height, width = image.shape[:2]
        scale = min(1.0, max_side / max(height, width))
        if scale < 1.0:
            image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(temporary / f"{local_index:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            shutil.rmtree(temporary)
            raise RuntimeError("Failed to materialize resized clip")
    return temporary


def output_rows(outputs: dict, event: dict, local_frame: int, width: int, height: int,
                model_name: str, init_mode: str) -> list[dict]:
    obj_ids = np.asarray(outputs.get("out_obj_ids", []))
    probs = np.asarray(outputs.get("out_probs", []))
    boxes = np.asarray(outputs.get("out_boxes_xywh", []))
    rows = []
    for index, obj_id in enumerate(obj_ids.tolist()):
        if index >= len(boxes) or index >= len(probs):
            continue
        left, top, box_width, box_height = map(float, boxes[index])
        # SAM 3 video outputs normalized xywh boxes.
        x1, y1 = left * width, top * height
        x2, y2 = (left + box_width) * width, (top + box_height) * height
        score = float(probs[index])
        if not math.isfinite(score) or x2 <= x1 or y2 <= y1:
            continue
        rows.append({
            "event_id": event["event_id"],
            "condition": event["condition"],
            "sequence_id": event["sequence_id"],
            "frame_index": event["clip_start_frame"] + local_frame,
            "local_frame_index": local_frame,
            "sam3_obj_id": int(obj_id),
            "global_linked_id": None,
            "track_id_source": "sam3_native",
            "bbox_xyxy": [x1, y1, x2, y2],
            "score": score,
            "category": event["category"],
            "model": model_name,
            "initialization": init_mode,
        })
    return rows


def find_point_initialization(event: dict, target_frames: dict[int, dict], width: int, height: int) -> tuple[int, list[list[float]]]:
    preferred_end = max(event["clip_start_frame"], event["start_frame"] - 1)
    candidates = [frame for frame in target_frames if event["clip_start_frame"] <= frame <= preferred_end]
    if not candidates:
        candidates = [frame for frame in target_frames if event["clip_start_frame"] <= frame <= event["clip_end_frame"]]
    if not candidates:
        raise RuntimeError(f"No GT initialization frame for {event['event_id']}")
    global_frame = max(candidates) if max(candidates) <= preferred_end else min(candidates)
    left, top, box_width, box_height = target_frames[global_frame]["bbox_xywh"]
    point = [[(left + box_width / 2) / width, (top + box_height / 2) / height]]
    return global_frame - event["clip_start_frame"], point


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-mode", choices=("text", "native_id_refine"), default="text")
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument("--new-detection-threshold", type=float, default=0.7)
    parser.add_argument("--max-objects", type=int, default=50)
    parser.add_argument("--input-max-side", type=int, default=1008)
    parser.add_argument("--event-limit", type=int)
    parser.add_argument("--event-id")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    install_cpu_mask_iou_patch()
    from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU

    root = args.dataset_root.resolve()
    events = json.loads(args.events.read_text(encoding="utf-8"))
    if args.event_id:
        events = [item for item in events if item["event_id"] == args.event_id]
    if args.event_limit is not None:
        events = events[:args.event_limit]
    if not events:
        raise SystemExit("No selected events")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictor = Sam3VideoPredictorMultiGPU(
        gpus_to_use=[0], checkpoint_path=str(args.checkpoint.resolve()),
        strict_state_dict_loading=True, apply_temporal_disambiguation=True,
    )
    predictor.model.score_threshold_detection = args.detection_threshold
    predictor.model.new_det_thresh = args.new_detection_threshold
    predictor.model.max_num_objects = args.max_objects

    completed = 0
    try:
        for event_index, selected_event in enumerate(events, 1):
            output_path = args.output_dir / f"{selected_event['event_id']}.jsonl"
            if output_path.is_file() and output_path.stat().st_size > 0:
                print(f"skip {event_index}/{len(events)} {selected_event['event_id']}", flush=True)
                completed += 1
                continue
            sequence_dir = root / "sequences" / selected_event["sequence_id"]
            first_frame = next(iter(sorted(sequence_dir.glob("*.jpg"))))
            width, height = frame_size(first_frame)
            gt = read_gt(root / "annotations" / f"{selected_event['sequence_id']}.txt")
            clip_dir = prepare_clip(
                sequence_dir, selected_event["clip_start_frame"], selected_event["clip_end_frame"],
                args.input_max_side, selected_event,
            )
            temporary_output = output_path.with_suffix(".jsonl.partial")
            session_id = None
            try:
                response = predictor.handle_request({"type": "start_session", "resource_path": str(clip_dir)})
                session_id = response["session_id"]
                if args.init_mode == "text":
                    prompt_frame = 0
                    prompt_request = {
                        "type": "add_prompt", "session_id": session_id, "frame_index": prompt_frame,
                        "text": selected_event["category"],
                    }
                else:
                    prompt_frame, points = find_point_initialization(
                        selected_event, gt[int(selected_event["target_id"])], width, height
                    )
                    target_row = gt[int(selected_event["target_id"])][
                        selected_event["clip_start_frame"] + prompt_frame
                    ]
                    left, top, box_width, box_height = target_row["bbox_xywh"]
                    # First create object IDs through SAM 3's own text grounding path.
                    text_response = predictor.handle_request({
                        "type": "add_prompt", "session_id": session_id, "frame_index": prompt_frame,
                        "text": selected_event["category"],
                    })
                    text_rows = output_rows(
                        text_response["outputs"], selected_event, prompt_frame, width, height,
                        args.model_name, args.init_mode,
                    )
                    if not text_rows:
                        raise RuntimeError(f"SAM 3 produced no native ID to refine for {selected_event['event_id']}")
                    target_box = [left, top, left + box_width, top + box_height]
                    native_obj_id = int(max(text_rows, key=lambda row: box_iou(row["bbox_xyxy"], target_box))["sam3_obj_id"])
                    # Instance refinement requires the normal native-ID propagation cache.
                    for _ in predictor.handle_stream_request({
                        "type": "propagate_in_video", "session_id": session_id,
                        "propagation_direction": "both", "start_frame_index": prompt_frame,
                    }):
                        pass
                    prompt_request = {
                        "type": "add_prompt", "session_id": session_id, "frame_index": prompt_frame,
                        "points": points, "point_labels": [1], "obj_id": native_obj_id,
                    }
                prompt_response = predictor.handle_request(prompt_request)
                collected: dict[int, list[dict]] = {}
                collected[prompt_frame] = output_rows(
                    prompt_response["outputs"], selected_event, prompt_frame, width, height,
                    args.model_name, args.init_mode,
                )
                direction = "forward" if prompt_frame == 0 else "both"
                for response in predictor.handle_stream_request({
                    "type": "propagate_in_video", "session_id": session_id,
                    "propagation_direction": direction, "start_frame_index": prompt_frame,
                }):
                    local_frame = int(response["frame_index"])
                    collected[local_frame] = output_rows(
                        response["outputs"], selected_event, local_frame, width, height,
                        args.model_name, args.init_mode,
                    )
                with temporary_output.open("w", encoding="utf-8") as handle:
                    for local_frame in range(selected_event["clip_end_frame"] - selected_event["clip_start_frame"] + 1):
                        # Frame markers preserve evaluated frames even when no objects are returned.
                        handle.write(json.dumps({
                            "type": "frame", "event_id": selected_event["event_id"],
                            "sequence_id": selected_event["sequence_id"],
                            "frame_index": selected_event["clip_start_frame"] + local_frame,
                            "predictions": collected.get(local_frame, []),
                        }, ensure_ascii=False) + "\n")
                temporary_output.replace(output_path)
                completed += 1
                print(f"done {event_index}/{len(events)} {selected_event['event_id']} rows={sum(map(len, collected.values()))}", flush=True)
            finally:
                if session_id is not None:
                    predictor.handle_request({"type": "close_session", "session_id": session_id})
                shutil.rmtree(clip_dir, ignore_errors=True)
                if temporary_output.exists():
                    temporary_output.unlink()
                torch.cuda.empty_cache()
    finally:
        predictor.shutdown()
    metadata = {
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "init_mode": args.init_mode,
        "track_id_source": "sam3_native",
        "events_requested": len(events),
        "events_completed": completed,
        "detection_threshold": args.detection_threshold,
        "new_detection_threshold": args.new_detection_threshold,
        "max_objects": args.max_objects,
        "input_max_side": args.input_max_side,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
