#!/usr/bin/env python3
"""Run the frozen adapted SAM 3 model on self-collected UAV images.

This script intentionally keeps the VisDrone-validation operating threshold
frozen.  It saves every prediction above a permissive score floor so that the
displayed qualitative result remains auditable.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam3.model import box_ops  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402


PROMPTS_BY_NUMBER = {
    98: ("person",),
    99: ("person",),
    100: ("person",),
    101: ("car", "bus", "person"),
    102: ("car", "bus", "person"),
    103: ("car", "person"),
    104: ("car", "person"),
    105: ("car", "person"),
    106: ("car", "person"),
    107: ("car", "bus", "person"),
}


def image_number(path: Path) -> int:
    match = re.search(r"\((\d+)\)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse image number from {path.name}")
    return int(match.group(1))


@torch.inference_mode()
def predict_boxes(model, processor: Sam3Processor, state: dict, prompt: str, score_floor: float) -> list[dict]:
    text_outputs = model.backbone.forward_text([prompt], device=processor.device)
    prompt_backbone = dict(state["backbone_out"])
    prompt_backbone.update(text_outputs)
    outputs = model.forward_grounding(
        backbone_out=prompt_backbone,
        find_input=processor.find_stage,
        geometric_prompt=model._get_dummy_prompt(),
        find_target=None,
    )
    scores = (
        outputs["pred_logits"].sigmoid()
        * outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
    ).squeeze(-1)[0]
    boxes = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][0])
    scale = torch.tensor(
        [state["original_width"], state["original_height"]] * 2,
        device=boxes.device,
        dtype=boxes.dtype,
    )
    boxes = boxes * scale
    keep = torch.nonzero(scores >= score_floor, as_tuple=False).flatten()
    if keep.numel() == 0:
        return []
    order = keep[torch.argsort(scores[keep], descending=True)]
    records = []
    for box, score in zip(boxes[order].float().cpu().tolist(), scores[order].float().cpu().tolist()):
        if math.isfinite(score) and all(math.isfinite(value) for value in box):
            records.append({"bbox_xyxy": box, "score": score})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operating-threshold", type=float, default=0.4773983955383301)
    parser.add_argument("--model-name", default="Adapted SAM 3 with replay (Ours)")
    parser.add_argument("--score-floor", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    image_paths = sorted(args.images_dir.glob("*.png"), key=image_number)
    if not image_paths:
        raise SystemExit("No PNG images found")
    unknown = [path.name for path in image_paths if image_number(path) not in PROMPTS_BY_NUMBER]
    if unknown:
        raise ValueError(f"No fixed prompt manifest for: {unknown}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint.resolve()),
        load_from_HF=False,
        enable_segmentation=False,
        device=args.device,
    )
    processor = Sam3Processor(model, device=args.device, confidence_threshold=0.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    started = time.time()
    metadata = {
        "type": "metadata",
        "evaluation_type": "qualitative_generalization_without_ground_truth",
        "images_dir": str(args.images_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "model_name": args.model_name,
        "operating_threshold": args.operating_threshold,
        "threshold_source": "frozen_VisDrone_DET_validation",
        "score_floor": args.score_floor,
        "image_count": len(image_paths),
        "prompt_manifest": {
            path.name: list(PROMPTS_BY_NUMBER[image_number(path)]) for path in image_paths
        },
    }
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for index, image_path in enumerate(image_paths, 1):
            image = Image.open(image_path).convert("RGB")
            state = processor.set_image(image)
            for prompt in PROMPTS_BY_NUMBER[image_number(image_path)]:
                predictions = predict_boxes(model, processor, state, prompt, args.score_floor)
                handle.write(
                    json.dumps(
                        {
                            "type": "prediction",
                            "image_name": image_path.name,
                            "width": image.width,
                            "height": image.height,
                            "prompt": prompt,
                            "predictions": predictions,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(
                f"images={index}/{len(image_paths)} name={image_path.name} "
                f"elapsed_s={time.time() - started:.1f}",
                flush=True,
            )
    temporary.replace(args.output)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
