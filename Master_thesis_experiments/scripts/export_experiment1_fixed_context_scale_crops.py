#!/usr/bin/env python3
"""Export Experiment 1 crops with a fixed source-space field of view.

All Tiny/Small/Regular panels use the same 400x300-pixel crop before being
resized for PowerPoint.  This preserves the visible scale difference instead
of enlarging every selected object to approximately the same display size.
Only official VisDrone GT and saved model predictions are drawn.
"""

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
    / "ppt_table_fixed_context_crops"
)
EVALUATOR_PATH = ROOT / "scripts/evaluate_visdrone_predictions.py"
SOURCE_CROP_SIZE = (400, 300)
OUTPUT_SIZE = (800, 600)

CASES = [
    {
        "scale": "tiny",
        "image_name": "9999973_00000_d_0000058.jpg",
        "category": "car",
        "selected_gt": [1167.0, 32.0, 1200.0, 57.0],
    },
    {
        "scale": "small",
        "image_name": "0000259_03000_d_0000007.jpg",
        "category": "truck",
        "selected_gt": [675.0, 133.0, 706.0, 179.0],
    },
    {
        "scale": "regular",
        "image_name": "9999938_00000_d_0000354.jpg",
        "category": "bus",
        "selected_gt": [779.0, 288.0, 818.0, 426.0],
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
    "tp": "#168bd2",
    "fp": "#d62728",
}


def load_evaluator():
    spec = importlib.util.spec_from_file_location("vis_eval", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_selected_predictions(path: Path, image_names: set[str]) -> dict:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "prediction" and row["image_name"] in image_names:
                result[(row["image_name"], row["category"])] = row["predictions"]
    return result


def fixed_crop(target: list[float], image_size: tuple[int, int]):
    crop_width, crop_height = SOURCE_CROP_SIZE
    image_width, image_height = image_size
    center_x = (target[0] + target[2]) / 2
    center_y = (target[1] + target[3]) / 2
    left = max(0, min(image_width - crop_width, round(center_x - crop_width / 2)))
    top = max(0, min(image_height - crop_height, round(center_y - crop_height / 2)))
    return (left, top, left + crop_width, top + crop_height)


def center_inside(box: list[float], roi: tuple[int, int, int, int]) -> bool:
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    return roi[0] <= center_x < roi[2] and roi[1] <= center_y < roi[3]


def intersects(box: list[float], roi: tuple[int, int, int, int]) -> bool:
    return (
        min(box[2], roi[2]) > max(box[0], roi[0])
        and min(box[3], roi[3]) > max(box[1], roi[1])
    )


def shift_and_clip(box: list[float], roi: tuple[int, int, int, int]):
    x1 = max(roi[0], min(roi[2], box[0])) - roi[0]
    y1 = max(roi[1], min(roi[3], box[1])) - roi[1]
    x2 = max(roi[0], min(roi[2], box[2])) - roi[0]
    y2 = max(roi[1], min(roi[3], box[3])) - roi[1]
    return [x1, y1, x2, y2]


def dashed_line(draw, start, end, fill, width, dash=9, gap=6):
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


def match_predictions(ev, predictions: list[dict], targets: list[dict]):
    matched = set()
    result = []
    for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
        best_index = None
        best_iou = 0.5
        for index, target in enumerate(targets):
            if index in matched:
                continue
            overlap = ev.iou(prediction["bbox_xyxy"], target["bbox"])
            if overlap >= best_iou:
                best_index, best_iou = index, overlap
        if best_index is None:
            result.append({**prediction, "status": "fp"})
        else:
            matched.add(best_index)
            result.append({**prediction, "status": "tp"})
    return result, matched


def render_panel(
    image: Image.Image,
    roi,
    targets: list[dict],
    selected_gt: list[float],
    selected_scale: str,
    predictions: list[dict] | None,
    ev,
):
    crop = image.crop(roi)
    draw = ImageDraw.Draw(crop)
    visible_targets = [target for target in targets if intersects(target["bbox"], roi)]
    if predictions is None:
        for target in visible_targets:
            is_selected = all(
                abs(a - b) < 1e-6 for a, b in zip(target["bbox"], selected_gt)
            )
            color = COLORS[selected_scale] if is_selected else COLORS["gt"]
            dashed_rectangle(draw, shift_and_clip(target["bbox"], roi), color, 3)
    else:
        for target in visible_targets:
            dashed_rectangle(draw, shift_and_clip(target["bbox"], roi), COLORS["gt"], 3)
        visible_predictions = [
            prediction
            for prediction in predictions
            if intersects(prediction["bbox_xyxy"], roi)
        ]
        # Match against every same-prompt target before clipping. This prevents a
        # correct prediction for a target crossing the crop boundary from being
        # mislabeled as an FP merely because its centre is outside the ROI.
        annotated, _ = match_predictions(ev, visible_predictions, targets)
        for prediction in annotated:
            draw.rectangle(
                [int(round(v)) for v in shift_and_clip(prediction["bbox_xyxy"], roi)],
                outline=COLORS[prediction["status"]],
                width=4,
            )
    return crop.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS), visible_targets


def make_grid(rows: list[list[Image.Image]], output: Path):
    gutter = 8
    panel_width, panel_height = OUTPUT_SIZE
    canvas = Image.new(
        "RGB",
        (
            panel_width * len(rows[0]) + gutter * (len(rows[0]) - 1),
            panel_height * len(rows) + gutter * (len(rows) - 1),
        ),
        "white",
    )
    for row_index, row in enumerate(rows):
        for column_index, panel in enumerate(row):
            canvas.paste(
                panel,
                (
                    column_index * (panel_width + gutter),
                    row_index * (panel_height + gutter),
                ),
            )
    canvas.save(output, dpi=(300, 300), compress_level=3)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ev = load_evaluator()
    image_names = {case["image_name"] for case in CASES}
    gt_all, ignore_integrals = ev.load_gt(DATASET, image_names)
    model_predictions = {}
    for model in MODELS:
        predictions = load_selected_predictions(PREDICTIONS / model["file"], image_names)
        predictions = ev.limit_detections_per_image(predictions, 500)
        predictions = ev.drop_predictions_in_ignored_regions(predictions, ignore_integrals)
        model_predictions[model["name"]] = predictions

    audit = {
        "source_crop_size_pixels": list(SOURCE_CROP_SIZE),
        "output_size_pixels": list(OUTPUT_SIZE),
        "note": "All rows use the same source-space field of view; no generative editing.",
        "cases": {},
    }
    rows = []
    for case in CASES:
        key = (case["image_name"], case["category"])
        image = Image.open(DATASET / "images" / case["image_name"]).convert("RGB")
        targets = [target for target in gt_all[key] if not target["ignored"]]
        selected = case["selected_gt"]
        selected_target = next(
            target
            for target in targets
            if all(abs(a - b) < 1e-6 for a, b in zip(target["bbox"], selected))
        )
        if selected_target["scale"] != case["scale"]:
            raise RuntimeError(f"Wrong scale for {case['scale']}")
        roi = fixed_crop(selected, image.size)
        gt_panel, visible_targets = render_panel(
            image, roi, targets, selected, case["scale"], None, ev
        )
        gt_panel.save(OUTPUT / f"{case['scale']}_ground_truth.png", dpi=(300, 300))
        row = [gt_panel]
        model_audit = {}
        for model in MODELS:
            filtered = [
                prediction
                for prediction in model_predictions[model["name"]].get(key, [])
                if prediction["score"] >= model["threshold"]
            ]
            panel, _ = render_panel(
                image, roi, targets, selected, case["scale"], filtered, ev
            )
            panel.save(OUTPUT / f"{case['scale']}_{model['slug']}.png", dpi=(300, 300))
            row.append(panel)
            best_iou = max(
                (ev.iou(prediction["bbox_xyxy"], selected) for prediction in filtered),
                default=0.0,
            )
            model_audit[model["name"]] = {
                "selected_target_best_iou": best_iou,
                "selected_target_correct": best_iou >= 0.5,
            }
        rows.append(row)
        width = selected[2] - selected[0]
        height = selected[3] - selected[1]
        audit["cases"][case["scale"]] = {
            "image_name": case["image_name"],
            "category": case["category"],
            "selected_gt_xyxy": selected,
            "selected_gt_width_height": [width, height],
            "s_original_pixels": math.sqrt(width * height),
            "verified_scale": selected_target["scale"],
            "roi_xyxy": list(roi),
            "visible_valid_prompt_gt_count": len(visible_targets),
            "models": model_audit,
        }

    make_grid(rows, OUTPUT / "comparison_fixed_context_3x5_no_text.png")
    (OUTPUT / "scale_and_crop_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        """# Experiment 1 fixed-context PPT crops

All 15 panels use the same 400x300-pixel source-space crop and are exported at
800x600. This preserves the real Tiny/Small/Regular visual-size difference.

Rows: Tiny car, Small truck, Regular bus.
Columns: Ground truth, YOLOv8x, YOLO-WorldV2, Original SAM 3, Ours.

The files are produced only from official VisDrone images/GT and saved model
predictions. No target, detection, or object size is synthesized or edited.
""",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
