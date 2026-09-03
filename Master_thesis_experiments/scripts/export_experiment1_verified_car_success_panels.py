#!/usr/bin/env python3
"""Export verified, prompt-specific Experiment 1 success panels.

The selected scenes use the same ``car`` prompt at Tiny, Small, and Regular
scales.  Every evaluable car GT is matched by Ours at the frozen test threshold,
with zero false positives.  The script verifies that claim before exporting.
No generative image editing is used.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path("/home/alien/sam3/Master_thesis_experiments")
DATASET = Path("/home/alien/Downloads/VisDrone2019-DET-test-dev")
OUTPUT = ROOT / "figures/experiment1/ppt_clean_panels_detection_baseline"
EVALUATOR_PATH = ROOT / "scripts/evaluate_visdrone_predictions.py"
PREDICTION_ROOT = ROOT / "predictions/experiment1/test"
PROMPT = "car"

SCENES = [
    {
        "scale": "tiny",
        "image_name": "9999938_00000_d_0000252.jpg",
        "gt_count": 14,
    },
    {
        "scale": "small",
        "image_name": "9999996_00000_d_0000018.jpg",
        "gt_count": 13,
    },
    {
        "scale": "regular",
        "image_name": "9999963_00000_d_0000009.jpg",
        "gt_count": 18,
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
        "slug": "yolo_world_v2",
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

EXPECTED = {
    "tiny": {
        "YOLOv8x": (0, 0, 14),
        "YOLOv8x-WorldV2": (11, 3, 3),
        "Original SAM 3": (14, 1, 0),
        "Adapted SAM 3 without replay": (13, 186, 1),
        "Adapted SAM 3 with replay (Ours)": (14, 0, 0),
    },
    "small": {
        "YOLOv8x": (11, 6, 2),
        "YOLOv8x-WorldV2": (11, 0, 2),
        "Original SAM 3": (13, 1, 0),
        "Adapted SAM 3 without replay": (0, 0, 13),
        "Adapted SAM 3 with replay (Ours)": (13, 0, 0),
    },
    "regular": {
        "YOLOv8x": (3, 1, 15),
        "YOLOv8x-WorldV2": (17, 2, 1),
        "Original SAM 3": (18, 1, 0),
        "Adapted SAM 3 without replay": (0, 0, 18),
        "Adapted SAM 3 with replay (Ours)": (18, 0, 0),
    },
}

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
    predictions = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "prediction" and row["image_name"] in image_names:
                predictions[(row["image_name"], row["category"])] = row["predictions"]
    return predictions


def annotate_predictions(ev, predictions: list[dict], targets: list[dict]):
    evaluable = [index for index, target in enumerate(targets) if not target["ignored"]]
    ignored = [index for index, target in enumerate(targets) if target["ignored"]]
    matched: set[int] = set()
    annotated = []
    for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
        best_index, best_iou = None, 0.5
        for index in evaluable:
            overlap = ev.iou(prediction["bbox_xyxy"], targets[index]["bbox"])
            if index not in matched and overlap >= best_iou:
                best_index, best_iou = index, overlap
        if best_index is not None:
            matched.add(best_index)
            annotated.append({**prediction, "status": "TP", "iou": best_iou})
            continue
        if any(
            ev.ioa_detection(prediction["bbox_xyxy"], targets[index]["bbox"]) >= 0.5
            for index in ignored
        ):
            continue
        annotated.append({**prediction, "status": "FP", "iou": 0.0})
    return annotated


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


def draw_gt_panel(image: Image.Image, targets: list[dict]) -> Image.Image:
    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    for target in targets:
        if not target["ignored"]:
            dashed_rectangle(draw, target["bbox"], COLORS[target["scale"]], 4)
    return panel


def draw_model_panel(
    image: Image.Image,
    targets: list[dict],
    annotated: list[dict],
) -> Image.Image:
    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    for target in targets:
        if not target["ignored"]:
            dashed_rectangle(draw, target["bbox"], COLORS["gt"], 3)
    for prediction in annotated:
        color = COLORS["tp"] if prediction["status"] == "TP" else COLORS["fp"]
        box = [int(round(value)) for value in prediction["bbox_xyxy"]]
        draw.rectangle(box, outline=color, width=4)
    return panel


def make_grid(rows: list[list[Image.Image]], output: Path) -> None:
    panel_width, panel_height = rows[0][0].size
    gutter = 12
    width = panel_width * len(rows[0]) + gutter * (len(rows[0]) - 1)
    height = panel_height * len(rows) + gutter * (len(rows) - 1)
    sheet = Image.new("RGB", (width, height), "white")
    for row_index, row in enumerate(rows):
        for column_index, panel in enumerate(row):
            sheet.paste(
                panel,
                (
                    column_index * (panel_width + gutter),
                    row_index * (panel_height + gutter),
                ),
            )
    sheet.save(output, dpi=(300, 300), compress_level=3)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ev = load_evaluator()
    image_names = {scene["image_name"] for scene in SCENES}
    gt, ignore_integrals = ev.load_gt(DATASET, image_names)

    model_predictions = {}
    for model in MODELS:
        predictions = load_selected_predictions(
            PREDICTION_ROOT / model["file"], image_names
        )
        predictions = ev.limit_detections_per_image(predictions, 500)
        predictions = ev.drop_predictions_in_ignored_regions(
            predictions, ignore_integrals
        )
        model_predictions[model["name"]] = predictions

    verification = {
        "protocol": {
            "prompt": PROMPT,
            "matching": "class-aware IoU >= 0.50",
            "thresholds": {
                model["name"]: model["threshold"] for model in MODELS
            },
            "max_detections_per_image": 500,
            "note": "Representative selected success cases; not aggregate test-set results.",
        },
        "scenes": {},
    }
    compact_rows = []
    original_rows = []
    full_rows = []

    for scene in SCENES:
        image_name = scene["image_name"]
        scale = scene["scale"]
        key = (image_name, PROMPT)
        targets = gt.get(key, [])
        evaluable = [target for target in targets if not target["ignored"]]
        scale_counts = {
            name: sum(target["scale"] == name for target in evaluable)
            for name in ev.SCALES
        }
        if len(evaluable) != scene["gt_count"] or scale_counts[scale] != len(evaluable):
            raise RuntimeError(
                f"{image_name}: expected {scene['gt_count']} pure {scale} car GT, "
                f"found {len(evaluable)} with {scale_counts}"
            )

        image = Image.open(DATASET / "images" / image_name).convert("RGB")
        gt_panel = draw_gt_panel(image, targets)
        gt_panel.save(
            OUTPUT / f"{scale}_01_ground_truth_car.png",
            dpi=(300, 300),
            compress_level=3,
        )
        panels = {}
        scene_record = {
            "image_name": image_name,
            "prompt": PROMPT,
            "scale_counts": scale_counts,
            "models": {},
        }

        for index, model in enumerate(MODELS, start=2):
            predictions = model_predictions[model["name"]].get(key, [])
            labelled, gt_count = ev.label_predictions(
                {key: predictions}, {key: targets}
            )
            metrics = ev.counts_at_threshold(
                labelled, gt_count, model["threshold"]
            )
            actual = (metrics["tp"], metrics["fp"], metrics["fn"])
            if actual != EXPECTED[scale][model["name"]]:
                raise RuntimeError(
                    f"{scale}/{model['name']}: expected "
                    f"{EXPECTED[scale][model['name']]}, found {actual}"
                )
            filtered = [
                prediction
                for prediction in predictions
                if prediction["score"] >= model["threshold"]
            ]
            annotated = annotate_predictions(ev, filtered, targets)
            panel = draw_model_panel(image, targets, annotated)
            panel.save(
                OUTPUT / f"{scale}_{index:02d}_{model['slug']}_car.png",
                dpi=(300, 300),
                compress_level=3,
            )
            panels[model["slug"]] = panel
            scene_record["models"][model["name"]] = metrics

        ours = scene_record["models"]["Adapted SAM 3 with replay (Ours)"]
        if (ours["tp"], ours["fp"], ours["fn"]) != (
            scene["gt_count"],
            0,
            0,
        ):
            raise RuntimeError(f"{scale}: Ours is not a strict success: {ours}")

        verification["scenes"][scale] = scene_record
        compact_rows.append(
            [
                gt_panel,
                panels["yolov8x"],
                panels["yolo_world_v2"],
                panels["ours"],
            ]
        )
        original_rows.append(
            [
                gt_panel,
                panels["yolov8x"],
                panels["yolo_world_v2"],
                panels["original_sam3"],
                panels["ours"],
            ]
        )
        full_rows.append(
            [
                gt_panel,
                panels["yolov8x"],
                panels["yolo_world_v2"],
                panels["original_sam3"],
                panels["without_replay"],
                panels["ours"],
            ]
        )

    make_grid(
        compact_rows,
        OUTPUT / "qualitative_car_strict_success_4col_clean_grid_no_text.png",
    )
    make_grid(
        original_rows,
        OUTPUT / "qualitative_car_strict_success_5col_clean_grid_no_text.png",
    )
    make_grid(
        full_rows,
        OUTPUT / "qualitative_car_strict_success_all_models_clean_grid_no_text.png",
    )

    (OUTPUT / "verification.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUTPUT / "README.md").write_text(
        """# Experiment 1 verified prompt-specific success panels

All files are deterministically redrawn from the original VisDrone images,
official GT annotations, and saved model predictions. No generative editing is
used. The prompt is `car` in all three scenes.

Rows:
1. Tiny: `9999938_00000_d_0000252.jpg` (14 car GT)
2. Small: `9999996_00000_d_0000018.jpg` (13 car GT)
3. Regular: `9999963_00000_d_0000009.jpg` (18 car GT)

Recommended PPT layout:
- Use `qualitative_car_strict_success_5col_clean_grid_no_text.png`.
- Columns: GT, YOLOv8x, YOLO-WorldV2, Original SAM 3, Ours.
- Add the row and column labels in PowerPoint, not inside the bitmap.
- The 4-column grid omits Original SAM 3.
- The all-model grid also includes without replay, but its Tiny panel is
  visually dense because the frozen threshold produces 186 false positives.

Box colors:
- GT-only column: orange=Tiny, green=Small, purple=Regular.
- Model columns: dashed green=valid car GT, solid blue=TP, solid red=FP.

Verified Ours results in every row: all GT matched, FP=0, FN=0.
These are representative selected success cases. Use the aggregate test table
for the full-dataset conclusion.
""",
        encoding="utf-8",
    )
    print(f"output={OUTPUT}")
    print(json.dumps(verification["scenes"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
