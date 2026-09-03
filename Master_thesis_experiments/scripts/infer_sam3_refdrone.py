#!/usr/bin/env python3
"""Run resumable SAM 3 synonym and specificity inference on RefDrone."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam3.model import box_ops  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402


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
    rows = []
    for box, score in zip(
        boxes[order].float().cpu().tolist(),
        scores[order].float().cpu().tolist(),
    ):
        if math.isfinite(score) and all(math.isfinite(value) for value in box):
            rows.append({"bbox_xyxy": box, "score": score})
    return rows


def build_tasks(
    category_units_path: Path,
    specificity_path: Path,
    evaluation: str,
) -> dict[str, list[dict]]:
    tasks: dict[str, list[dict]] = defaultdict(list)
    category_units = json.loads(category_units_path.read_text(encoding="utf-8"))
    if evaluation in {"all", "synonym"}:
        for unit in category_units:
            if unit["synonym_prompt"] is None:
                continue
            tasks[unit["file_name"]].append({
                "task_id": f"synonym:{unit['unit_id']}",
                "evaluation": "synonym",
                "category": unit["category"],
                "prompt": unit["synonym_prompt"],
            })
    specificity = json.loads(specificity_path.read_text(encoding="utf-8"))
    if evaluation in {"all", "specificity"}:
        for example in specificity:
            for layer in example["layers"]:
                tasks[example["file_name"]].append({
                    "task_id": f"specificity:{example['example_id']}:L{layer['layer']}",
                    "evaluation": "specificity",
                    "example_id": example["example_id"],
                    "category": example["category"],
                    "layer": int(layer["layer"]),
                    "prompt": layer["prompt"],
                })
    return dict(tasks)


def read_completed(output: Path) -> set[str]:
    completed: set[str] = set()
    if not output.is_file():
        return completed
    with output.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") == "image_complete":
                completed.add(row["image_name"])
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--category-units", type=Path, required=True)
    parser.add_argument("--specificity-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-floor", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-images", type=int)
    parser.add_argument(
        "--evaluation", choices=("all", "synonym", "specificity"), default="all"
    )
    args = parser.parse_args()

    tasks = build_tasks(args.category_units, args.specificity_manifest, args.evaluation)
    images_dir = args.dataset_root / "all_image"
    missing = [name for name in tasks if not (images_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} task images missing; first={missing[0]}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(args.output)
    remaining = [name for name in sorted(tasks) if name not in completed]
    if args.max_new_images is not None:
        remaining = remaining[:args.max_new_images]
    print(
        f"model={args.model_name} total_images={len(tasks)} completed={len(completed)} "
        f"remaining={len(remaining)} total_prompts={sum(len(tasks[n]) for n in remaining)}",
        flush=True,
    )
    if not remaining:
        return

    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint.resolve()),
        load_from_HF=False,
        enable_segmentation=False,
        device=args.device,
    )
    processor = Sam3Processor(model, device=args.device, confidence_threshold=0.0)
    new_file = not args.output.exists() or args.output.stat().st_size == 0
    started = time.time()
    with args.output.open("a", encoding="utf-8") as handle:
        if new_file:
            handle.write(json.dumps({
                "type": "metadata",
                "model_name": args.model_name,
                "checkpoint": str(args.checkpoint.resolve()),
                "dataset_root": str(args.dataset_root.resolve()),
                "category_units": str(args.category_units.resolve()),
                "specificity_manifest": str(args.specificity_manifest.resolve()),
                "image_count": len(tasks),
                "prompt_count": sum(map(len, tasks.values())),
                "score_floor": args.score_floor,
            }, ensure_ascii=False) + "\n")
            handle.flush()
        for index, image_name in enumerate(remaining, 1):
            state = processor.set_image(Image.open(images_dir / image_name).convert("RGB"))
            for task in tasks[image_name]:
                predictions = predict_boxes(model, processor, state, task["prompt"], args.score_floor)
                handle.write(json.dumps({
                    "type": "prediction",
                    "image_name": image_name,
                    **task,
                    "predictions": predictions,
                }, ensure_ascii=False) + "\n")
            handle.write(json.dumps({
                "type": "image_complete",
                "image_name": image_name,
                "task_count": len(tasks[image_name]),
            }) + "\n")
            handle.flush()
            if index == 1 or index % 10 == 0 or index == len(remaining):
                elapsed = time.time() - started
                rate = index / elapsed if elapsed else 0.0
                eta = (len(remaining) - index) / rate if rate else 0.0
                print(
                    f"images={len(completed)+index}/{len(tasks)} new={index}/{len(remaining)} "
                    f"elapsed_s={elapsed:.1f} eta_s={eta:.1f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
