#!/usr/bin/env python3
"""Run reproducible language-guided SAM 3 box inference on VisDrone images."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


CATEGORIES = ("car", "person", "bus", "van", "truck", "motorcycle")


def read_images(images_dir: Path, manifest: Path | None, sample_size: int | None, seed: int) -> list[Path]:
    if manifest is None:
        paths = sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.png")])
    else:
        names = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        paths = [images_dir / name for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest images are missing; first={missing[0]}")
    if sample_size is not None and sample_size < len(paths):
        paths = sorted(
            paths,
            key=lambda path: hashlib.sha256(f"{seed}:{path.name}".encode()).digest(),
        )[:sample_size]
    return paths


@torch.inference_mode()
def predict_boxes(model, processor: Sam3Processor, state: dict, prompt: str, score_floor: float) -> list[dict]:
    text_outputs = model.backbone.forward_text([prompt], device=processor.device)
    # forward_grounding removes backbone_fpn when segmentation is disabled. A shallow
    # per-prompt mapping preserves the cached image features for the next category.
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
    boxes_cpu = boxes[order].float().cpu()
    scores_cpu = scores[order].float().cpu()
    result = []
    for box, score in zip(boxes_cpu.tolist(), scores_cpu.tolist()):
        if not math.isfinite(score) or not all(math.isfinite(value) for value in box):
            continue
        result.append({"bbox_xyxy": box, "score": score})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-floor", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    images_dir = args.dataset_root.resolve() / "images"
    image_paths = read_images(images_dir, args.image_manifest, args.sample_size, args.seed)
    if not image_paths:
        raise SystemExit("No input images")
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
    with temporary.open("w", encoding="utf-8") as handle:
        metadata = {
            "type": "metadata",
            "checkpoint": str(args.checkpoint.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "image_count": len(image_paths),
            "categories": list(CATEGORIES),
            "score_floor": args.score_floor,
            "selection_seed": args.seed,
            "image_names": [path.name for path in image_paths],
        }
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for index, image_path in enumerate(image_paths, 1):
            image = Image.open(image_path).convert("RGB")
            state = processor.set_image(image)
            for category in CATEGORIES:
                predictions = predict_boxes(model, processor, state, category, args.score_floor)
                record = {
                    "type": "prediction",
                    "image_name": image_path.name,
                    "category": category,
                    "prompt": category,
                    "predictions": predictions,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index == 1 or index % 10 == 0 or index == len(image_paths):
                elapsed = time.time() - started
                print(f"images={index}/{len(image_paths)} elapsed_s={elapsed:.1f}", flush=True)
    temporary.replace(args.output)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
