#!/usr/bin/env python3
"""Export native-resolution Experiment 1 panels without any embedded text."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("/home/alien/sam3")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Master_thesis_experiments.scripts import make_experiment1_full_scene_figures as source


OUTPUT = (
    ROOT
    / "Master_thesis_experiments/figures/experiment1/ppt_clean_panels_detection_baselines"
)
MODELS = [
    ("yolov8x", "YOLOv8x"),
    ("yolo_world_v2", "YOLOv8x-WorldV2"),
    ("ours", "Adapted SAM 3 with replay (Ours)"),
]
HEX_COLORS = {
    "gt": "#00a878",
    "tp": "#1f77b4",
    "fp": "#d62728",
    **source.SCALE_COLORS,
}


def dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill: str,
    width: int,
    dash: int = 10,
    gap: int = 7,
) -> None:
    x1, y1 = start
    x2, y2 = end
    if y1 == y2:
        direction = 1 if x2 >= x1 else -1
        position = x1
        while (position - x2) * direction <= 0:
            stop = position + direction * dash
            if (stop - x2) * direction > 0:
                stop = x2
            draw.line((position, y1, stop, y2), fill=fill, width=width)
            position = stop + direction * gap
    else:
        direction = 1 if y2 >= y1 else -1
        position = y1
        while (position - y2) * direction <= 0:
            stop = position + direction * dash
            if (stop - y2) * direction > 0:
                stop = y2
            draw.line((x1, position, x2, stop), fill=fill, width=width)
            position = stop + direction * gap


def dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    fill: str,
    width: int = 3,
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    dashed_line(draw, (x1, y1), (x2, y1), fill, width)
    dashed_line(draw, (x1, y2), (x2, y2), fill, width)
    dashed_line(draw, (x1, y1), (x1, y2), fill, width)
    dashed_line(draw, (x2, y1), (x2, y2), fill, width)


def ground_truth_panel(item: dict) -> Image.Image:
    image = item["image"].copy()
    draw = ImageDraw.Draw(image)
    for targets in item["gt"].values():
        for target in targets:
            if not target["ignored"]:
                dashed_rectangle(
                    draw,
                    target["bbox"],
                    HEX_COLORS[target["scale"]],
                    3,
                )
    return image


def model_panel(item: dict, model: str) -> Image.Image:
    image = item["image"].copy()
    draw = ImageDraw.Draw(image)
    for targets in item["gt"].values():
        for target in targets:
            if not target["ignored"]:
                dashed_rectangle(draw, target["bbox"], HEX_COLORS["gt"], 2)

    annotated = [
        prediction
        for predictions in item["models"][model]["annotated"].values()
        for prediction in predictions
    ]
    for prediction in annotated:
        color = HEX_COLORS["tp"] if prediction["status"] == "TP" else HEX_COLORS["fp"]
        box = [int(round(value)) for value in prediction["bbox_xyxy"]]
        draw.rectangle(box, outline=color, width=3)
    return image


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prepared, _ = source.prepare_scene_data()
    rows: list[list[Image.Image]] = []

    for item in prepared:
        scale_slug = item["scene"]["scale"].lower().replace("-", "_")
        row = [ground_truth_panel(item)]
        row.extend(model_panel(item, model) for _, model in MODELS)
        filenames = [
            f"{scale_slug}_01_ground_truth.png",
            *[
                f"{scale_slug}_{index + 2:02d}_{slug}.png"
                for index, (slug, _) in enumerate(MODELS)
            ],
        ]
        for image, filename in zip(row, filenames):
            image.save(OUTPUT / filename, dpi=(300, 300), compress_level=3)
        rows.append(row)

    panel_width, panel_height = rows[0][0].size
    gutter = 12
    sheet = Image.new(
        "RGB",
        (
            panel_width * 4 + gutter * 3,
            panel_height * 3 + gutter * 2,
        ),
        "white",
    )
    for row_index, row in enumerate(rows):
        for column_index, image in enumerate(row):
            sheet.paste(
                image,
                (
                    column_index * (panel_width + gutter),
                    row_index * (panel_height + gutter),
                ),
            )
    sheet.save(
        OUTPUT / "qualitative_detection_baselines_clean_grid_no_text.png",
        dpi=(300, 300),
        compress_level=3,
    )

    readme = """# Clean native-resolution PPT panels

No titles, labels, scores, legends, or other text are embedded in these images.
All panels are regenerated from the original VisDrone images and saved measured
predictions; no generative editing is used.

Rows/scenes:
1. tiny
2. small
3. regular_dominant

Columns/models:
1. ground_truth
2. YOLOv8x
3. YOLOv8x-WorldV2
4. Adapted SAM 3 with replay (Ours)

Box colors:
- GT-only panels: orange=Tiny, green=Small, purple=Regular.
- Model panels: dashed green=GT, solid blue=TP, solid red=FP.

Each individual panel is 1400 x 788 pixels. The clean grid is 5636 x 2388
pixels. Add titles, model names, legends, and captions in PowerPoint.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
