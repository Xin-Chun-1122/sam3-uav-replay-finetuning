#!/usr/bin/env python3
"""
Video Inference Script with Complex Prompt Support.

Usage:
  # Simple detection
  python inference/infer_video.py \
    --checkpoint experiments/stage2_uav_multidomain/checkpoints/checkpoint.pt \
    --video test_fire_video/fire_test.mp4 \
    --prompt "fire"

  # Complex temporal prompt
  python inference/infer_video.py \
    --checkpoint experiments/stage2_uav_multidomain/checkpoints/checkpoint.pt \
    --video test_fire_video/fire_test.mp4 \
    --prompt "find vehicles, then check if there is smoke near any detected vehicle" \
    --output_dir output_fire_video/
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parents[1]
SAM3_ROOT = PROJECT_ROOT.parent / "sam3"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SAM3_ROOT))

from inference.temporal_reasoner import Detection, TemporalReasoner
from prompts.prompt_engine import PromptEngine, classify_prompt
from visualization.draw_boxes import draw_boxes_cv2, save_annotated_image


def parse_args():
    parser = argparse.ArgumentParser("SAM3-I Video Inference")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--video",      required=True, help="Path to input video")
    parser.add_argument("--prompt",     required=True, help="Detection prompt (can be complex)")
    parser.add_argument("--output_dir", default="output_fire_video")
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--max_frames",   type=int,   default=-1)
    parser.add_argument("--save_frames",  action="store_true", help="Save individual frames")
    parser.add_argument("--save_video",   action="store_true", help="Save output video")
    parser.add_argument("--save_json",    action="store_true", help="Save JSON results")
    parser.add_argument("--device",       default="cuda")
    return parser.parse_args()


def load_sam3_model(checkpoint_path: str, device: str):
    """Load SAM3-I model from checkpoint."""
    print(f"Loading SAM3-I from {checkpoint_path}...")
    try:
        from hydra import compose, initialize_config_dir
        from sam3.model_builder import build_sam3_image_model

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = checkpoint.get("model", checkpoint)

        cfg_dir = str(PROJECT_ROOT / "configs" / "training")
        with initialize_config_dir(config_dir=cfg_dir, version_base=None):
            cfg = compose(config_name="stage2_multidomain")

        model = build_sam3_image_model(
            bpe_path=cfg.paths.bpe_path,
            device="cpu",
            eval_mode=True,
        )
        model.load_state_dict(model_state, strict=False)
        model = model.to(device).eval()
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"[WARNING] Could not load model: {e}")
        print("Running in DEMO mode (no actual inference)")
        return None


def run_inference(model, image: np.ndarray, prompt: str, device: str, score_thresh: float) -> List[Detection]:
    """Run SAM3-I inference on a single frame."""
    if model is None:
        # Demo mode: return random detections
        import random
        n = random.randint(0, 2)
        h, w = image.shape[:2]
        dets = []
        for _ in range(n):
            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, h // 2)
            x2 = x1 + random.randint(50, 200)
            y2 = y1 + random.randint(50, 200)
            label = prompt.split()[0] if prompt.split() else "object"
            dets.append(Detection(
                box=[x1, y1, min(x2, w), min(y2, h)],
                label=label,
                score=random.uniform(0.4, 0.95),
                prompt=prompt,
                frame_idx=0,
            ))
        return dets

    # Real inference
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(image[..., ::-1])  # BGR -> RGB

    with torch.no_grad():
        results = model.predict(pil_img, text=prompt)

    detections = []
    for box, score, label in zip(results.get("boxes", []), results.get("scores", []), results.get("labels", [])):
        if score >= score_thresh:
            detections.append(Detection(
                box=box.tolist(),
                label=label,
                score=float(score),
                prompt=prompt,
                frame_idx=0,
            ))
    return detections


def main():
    args = parse_args()

    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV not installed. Run: pip install opencv-python")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_sam3_model(args.checkpoint, device)

    # Classify the prompt type
    intent, targets = classify_prompt(args.prompt)
    print(f"\nPrompt:  {args.prompt}")
    print(f"Intent:  {intent}  |  Targets: {targets}")

    # Setup temporal reasoner for complex prompts
    reasoner = TemporalReasoner()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nVideo: {w}x{h} @ {fps:.1f}fps, {total_frames} frames")

    # Setup output video writer
    out_writer = None
    if args.save_video:
        out_path = output_dir / f"output_{Path(args.video).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    all_results = []
    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        # Run detection
        dets = run_inference(model, frame, args.prompt, device, args.score_thresh)
        tracked = reasoner.update(frame_idx, dets)

        # Build frame result
        frame_result = {
            "frame_idx": frame_idx,
            "timestamp_sec": frame_idx / fps,
            "detections": [
                {"box": d.box, "label": d.label, "score": d.score, "track_id": d.track_id}
                for d in tracked
            ],
        }

        # Apply complex reasoning
        if intent == "detect_new":
            new_events = reasoner.compare_with_previous(frame_idx, lookback_frames=5)
            frame_result["new_events"] = new_events
            if new_events:
                print(f"Frame {frame_idx:4d}: {len(new_events)} NEW event(s)!")

        elif intent == "no_duplicate":
            new_unique = reasoner.get_new_events(frame_idx)
            frame_result["unique_events"] = new_unique
            if new_unique:
                print(f"Frame {frame_idx:4d}: {len(new_unique)} unique event(s)!")

        elif intent == "find_near" and len(targets) >= 2:
            nearby = reasoner.get_objects_near(targets[0], targets[1], frame_idx)
            frame_result["nearby_events"] = nearby
            if nearby:
                print(f"Frame {frame_idx:4d}: {targets[1]} near {targets[0]}: {len(nearby)} match(es)")

        all_results.append(frame_result)

        # Draw and save
        boxes  = [d.box   for d in tracked]
        labels = [d.label for d in tracked]
        scores = [d.score for d in tracked]
        tids   = [d.track_id for d in tracked]

        annotated = draw_boxes_cv2(frame, boxes, labels, scores, tids)

        # Add frame info overlay
        info_text = f"Frame {frame_idx} | {args.prompt[:50]}"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if out_writer:
            out_writer.write(annotated)

        if args.save_frames and frame_idx % 30 == 0:
            frame_path = output_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(frame_path), annotated)

        frame_idx += 1

    cap.release()
    if out_writer:
        out_writer.release()

    elapsed = time.time() - t_start
    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS)")

    if args.save_json:
        json_path = output_dir / "detections.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved: {json_path}")

    # Summary
    total_dets = sum(len(r["detections"]) for r in all_results)
    print(f"\nSummary: {total_dets} total detections across {frame_idx} frames")
    print(f"Active tracks: {len(reasoner.tracks)}")


if __name__ == "__main__":
    main()
