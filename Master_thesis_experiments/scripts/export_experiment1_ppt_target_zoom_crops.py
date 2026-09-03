#!/usr/bin/env python3
"""Export highly localized, verified strict-win crops for the Experiment 1 PPT."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path("/home/alien/sam3/Master_thesis_experiments")
DATASET = Path("/home/alien/Downloads/VisDrone2019-DET-test-dev")
PREDICTIONS = ROOT / "predictions/experiment1/test"
OUTPUT = (
    ROOT
    / "figures/experiment1/ppt_clean_panels_detection_baseline"
    / "ppt_table_target_zoom_crops"
)
EVALUATOR_PATH = ROOT / "scripts/evaluate_visdrone_predictions.py"
OUTPUT_SIZE = (960, 720)

CASES = [
    {
        "scale": "tiny",
        "image_name": "9999973_00000_d_0000058.jpg",
        "category": "car",
        "gt": [1167.0, 32.0, 1200.0, 57.0],
    },
    {
        "scale": "small",
        "image_name": "0000259_03000_d_0000007.jpg",
        "category": "truck",
        "gt": [675.0, 133.0, 706.0, 179.0],
    },
    {
        "scale": "regular",
        "image_name": "9999938_00000_d_0000354.jpg",
        "category": "bus",
        "gt": [779.0, 288.0, 818.0, 426.0],
    },
]

MODELS = [
    {
        "slug": "yolov8x",
        "name": "YOLOv8x",
        "file": "yolo_test1610.jsonl",
        "threshold": 0.06027992069721222,
    },
    {
        "slug": "yolo_worldv2",
        "name": "YOLOv8x-WorldV2",
        "file": "yolo_world_test1610.jsonl",
        "threshold": 0.052561935037374496,
    },
    {
        "slug": "original_sam3",
        "name": "Original SAM 3",
        "file": "sam3_original_test1610.jsonl",
        "threshold": 0.48943960666656494,
    },
    {
        "slug": "without_replay",
        "name": "Adapted SAM 3 without replay",
        "file": "sam3_adapted_no_replay_test1610.jsonl",
        "threshold": 2.187017789090362e-10,
    },
    {
        "slug": "ours",
        "name": "Adapted SAM 3 with replay (Ours)",
        "file": "sam3_adapted_with_replay_test1610.jsonl",
        "threshold": 0.4773983955383301,
    },
]

COLORS = {
    "tiny": "#f0a202",
    "small": "#2ca02c",
    "regular": "#9467bd",
    "gt": "#00a878",
    "tp": "#1f77b4",
    "fp": "#d62728",
}


def load_evaluator():
    specification = importlib.util.spec_from_file_location("vis_eval", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def load_selected_predictions(path: Path, image_names: set[str]) -> dict:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "prediction" and row["image_name"] in image_names:
                result[(row["image_name"], row["category"])] = row["predictions"]
    return result


def crop_for_target(
    target: list[float], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = target
    target_height = y2 - y1
    crop_height = max(120.0, target_height * 2.25)
    crop_width = crop_height * 4 / 3
    target_width = x2 - x1
    if crop_width < target_width * 2.25:
        crop_width = target_width * 2.25
        crop_height = crop_width * 3 / 4
    image_width, image_height = image_size
    crop_width = min(crop_width, float(image_width))
    crop_height = min(crop_height, float(image_height))
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    left = max(0.0, min(image_width - crop_width, center_x - crop_width / 2))
    top = max(0.0, min(image_height - crop_height, center_y - crop_height / 2))
    return (
        int(round(left)),
        int(round(top)),
        int(round(left + crop_width)),
        int(round(top + crop_height)),
    )


def intersects(box: list[float], roi: tuple[int, int, int, int]) -> bool:
    return (
        min(box[2], roi[2]) > max(box[0], roi[0])
        and min(box[3], roi[3]) > max(box[1], roi[1])
    )


def dashed_line(draw, start, end, fill, width, dash=10, gap=7):
    x1, y1 = start
    x2, y2 = end
    if y1 == y2:
        direction, position = (1 if x2 >= x1 else -1), x1
        while (position - x2) * direction <= 0:
            stop = position + direction * dash
            if (stop - x2) * direction > 0:
                stop = x2
            draw.line((position, y1, stop, y2), fill=fill, width=width)
            position = stop + direction * gap
    else:
        direction, position = (1 if y2 >= y1 else -1), y1
        while (position - y2) * direction <= 0:
            stop = position + direction * dash
            if (stop - y2) * direction > 0:
                stop = y2
            draw.line((x1, position, x2, stop), fill=fill, width=width)
            position = stop + direction * gap


def dashed_rectangle(draw, box, fill, width=3):
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    dashed_line(draw, (x1, y1), (x2, y1), fill, width)
    dashed_line(draw, (x1, y2), (x2, y2), fill, width)
    dashed_line(draw, (x1, y1), (x1, y2), fill, width)
    dashed_line(draw, (x2, y1), (x2, y2), fill, width)


def draw_crop(
    image: Image.Image,
    roi: tuple[int, int, int, int],
    gt: list[float],
    gt_color: str,
    predictions: list[dict] | None,
    ev,
) -> Image.Image:
    # Draw at original resolution first so all panels use identical geometry.
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    dashed_rectangle(draw, gt, gt_color, 3)
    if predictions is not None:
        for prediction in predictions:
            if not intersects(prediction["bbox_xyxy"], roi):
                continue
            status = "tp" if ev.iou(prediction["bbox_xyxy"], gt) >= 0.5 else "fp"
            box = [int(round(value)) for value in prediction["bbox_xyxy"]]
            draw.rectangle(box, outline=COLORS[status], width=4)
    return canvas.crop(roi).resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ev = load_evaluator()
    image_names = {case["image_name"] for case in CASES}
    gt_all, ignore_integrals = ev.load_gt(DATASET, image_names)
    model_predictions = {}
    for model in MODELS:
        predictions = load_selected_predictions(
            PREDICTIONS / model["file"], image_names
        )
        predictions = ev.limit_detections_per_image(predictions, 500)
        predictions = ev.drop_predictions_in_ignored_regions(
            predictions, ignore_integrals
        )
        model_predictions[model["name"]] = predictions

    audit = {
        "note": (
            "Representative target-level strict wins. Scale is computed from "
            "the original GT before crop/resize. All models in a row use the "
            "same ROI."
        ),
        "output_size": list(OUTPUT_SIZE),
        "cases": {},
    }

    for case in CASES:
        scale = case["scale"]
        image_name = case["image_name"]
        category = case["category"]
        target = case["gt"]
        key = (image_name, category)
        if not any(
            all(abs(left - right) < 1e-6 for left, right in zip(item["bbox"], target))
            and not item["ignored"]
            and item["scale"] == scale
            for item in gt_all[key]
        ):
            raise RuntimeError(f"{scale}: selected GT is missing or has wrong scale")
        size = math.sqrt((target[2] - target[0]) * (target[3] - target[1]))
        image = Image.open(DATASET / "images" / image_name).convert("RGB")
        roi = crop_for_target(target, image.size)
        visible_prompt_gt = [
            candidate["bbox"]
            for candidate in gt_all.get((image_name, category), [])
            if not candidate["ignored"] and intersects(candidate["bbox"], roi)
        ]
        if len(visible_prompt_gt) != 1:
            raise RuntimeError(
                f"{scale}: zoom ROI must contain exactly one valid {category} GT; "
                f"found {visible_prompt_gt}"
            )

        gt_panel = draw_crop(
            image, roi, target, COLORS[scale], None, ev
        )
        gt_panel.save(
            OUTPUT / f"{scale}_ground_truth.png",
            dpi=(300, 300),
            compress_level=3,
        )

        model_record = {}
        for model in MODELS:
            filtered = [
                prediction
                for prediction in model_predictions[model["name"]].get(key, [])
                if prediction["score"] >= model["threshold"]
            ]
            best_iou = max(
                (ev.iou(prediction["bbox_xyxy"], target) for prediction in filtered),
                default=0.0,
            )
            model_record[model["name"]] = {
                "best_iou": best_iou,
                "correct_for_selected_gt": best_iou >= 0.5,
            }
            if model["slug"] == "without_replay":
                continue
            panel = draw_crop(
                image, roi, target, COLORS["gt"], filtered, ev
            )
            panel.save(
                OUTPUT / f"{scale}_{model['slug']}.png",
                dpi=(300, 300),
                compress_level=3,
            )

        if not model_record["Adapted SAM 3 with replay (Ours)"][
            "correct_for_selected_gt"
        ]:
            raise RuntimeError(f"{scale}: Ours does not match the target")
        if any(
            model_record[model["name"]]["correct_for_selected_gt"]
            for model in MODELS
            if model["slug"] != "ours"
        ):
            raise RuntimeError(f"{scale}: not a strict win against every baseline")

        audit["cases"][scale] = {
            "image_name": image_name,
            "category": category,
            "gt_xyxy": target,
            "s_original_pixels": size,
            "verified_scale": scale,
            "roi_xyxy": list(roi),
            "valid_prompt_gt_inside_roi": visible_prompt_gt,
            "models": model_record,
        }

    (OUTPUT / "target_zoom_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUTPUT / "README.md").write_text(
        """# Experiment 1 target-level zoom crops

Place these 15 files in the 3x5 PPT table.

Columns:
1. ground_truth
2. yolov8x
3. yolo_worldv2
4. original_sam3
5. ours

Rows:
- Tiny: car, s=28.72
- Small: truck, s=37.76
- Regular: bus, s=73.36

Every non-Ours baseline misses the selected GT at class-aware IoU 0.50 under
its frozen threshold. Ours matches it. These are selected target-level
qualitative examples; use the full-test table for aggregate claims.

Box colors:
- GT-only panel: orange/green/purple dashed box for Tiny/Small/Regular.
- Model panels: green dashed box=selected GT, blue solid=correct prediction,
  red solid=incorrect prediction intersecting the crop.
""",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
