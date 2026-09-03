#!/usr/bin/env python3
"""Run YOLO or YOLO-World on VisDrone and emit the common raw-prediction JSONL format."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO, YOLOWorld


CATEGORIES = ("car", "person", "bus", "van", "truck", "motorcycle")
COCO_TO_THESIS = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def read_images(images_dir: Path, manifest: Path | None) -> list[Path]:
    if manifest is None:
        return sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.png")])
    names = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    paths = [images_dir / name for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest images are missing; first={missing[0]}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", choices=("yolo", "yolo_world"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    images = read_images(args.dataset_root.resolve() / "images", args.image_manifest)
    if args.model_type == "yolo_world":
        model = YOLOWorld(str(args.checkpoint.resolve()))
        model.set_classes(list(CATEGORIES))
    else:
        model = YOLO(str(args.checkpoint.resolve()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    started = time.time()
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "type": "metadata",
            "checkpoint": str(args.checkpoint.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "image_count": len(images),
            "categories": list(CATEGORIES),
            "model_type": args.model_type,
            "imgsz": args.imgsz,
            "score_floor": args.score_floor,
            "max_det": args.max_det,
            "image_names": [path.name for path in images],
        }, ensure_ascii=False) + "\n")
        for start in range(0, len(images), args.batch_size):
            batch = images[start:start + args.batch_size]
            results = model.predict(
                source=[str(path) for path in batch],
                imgsz=args.imgsz,
                conf=args.score_floor,
                iou=0.7,
                max_det=args.max_det,
                device=args.device,
                verbose=False,
            )
            for image_path, result in zip(batch, results):
                grouped = {category: [] for category in CATEGORIES}
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.float().cpu().tolist()
                    scores = result.boxes.conf.float().cpu().tolist()
                    classes = result.boxes.cls.int().cpu().tolist()
                    for box, score, class_id in zip(boxes, scores, classes):
                        if args.model_type == "yolo_world":
                            if not 0 <= class_id < len(CATEGORIES):
                                continue
                            category = CATEGORIES[class_id]
                        else:
                            category = COCO_TO_THESIS.get(class_id)
                            if category is None:
                                continue
                        grouped[category].append({"bbox_xyxy": box, "score": score})
                for category in CATEGORIES:
                    handle.write(json.dumps({
                        "type": "prediction",
                        "image_name": image_path.name,
                        "category": category,
                        "prompt": category if args.model_type == "yolo_world" else None,
                        "predictions": grouped[category],
                    }, ensure_ascii=False) + "\n")
            completed = min(start + len(batch), len(images))
            if completed == len(images) or completed % 40 == 0:
                print(f"images={completed}/{len(images)} elapsed_s={time.time() - started:.1f}", flush=True)
    temporary.replace(args.output)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
