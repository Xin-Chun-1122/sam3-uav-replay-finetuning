#!/usr/bin/env python3
"""Export measured Experiment 3 results as equal-size PPT table panels.

The exporter reuses the frozen thresholds, saved predictions, matching policy,
and representative examples from make_experiment3_representative_figures.py.
It does not alter boxes or predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import make_experiment3_representative_figures as representative
import make_experiment3_success_figures as base


OUTPUT = base.ROOT / "figures/experiment3/ppt_table_panels"
PANEL_SIZE = (1200, 800)
COLORS = {
    "gt": "#00a878",
    "tp": "#1677c8",
    "fp": "#d62728",
}


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def aspect_crop(
    crop: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Expand a crop to the panel aspect ratio without cutting image content."""
    x1, y1, x2, y2 = crop
    desired = PANEL_SIZE[0] / PANEL_SIZE[1]
    width, height = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    if width / height < desired:
        width = height * desired
    else:
        height = width / desired
    x1, x2 = center_x - width / 2, center_x + width / 2
    y1, y2 = center_y - height / 2, center_y + height / 2
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if x2 > image_width:
        x1 -= x2 - image_width
        x2 = image_width
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if y2 > image_height:
        y1 -= y2 - image_height
        y2 = image_height
    return (
        max(0, round(x1)),
        max(0, round(y1)),
        min(image_width, round(x2)),
        min(image_height, round(y2)),
    )


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: str) -> None:
    label_font = font(19)
    left, top = xy
    box = draw.textbbox((left, top), text, font=label_font)
    padding = 5
    draw.rectangle(
        (box[0] - padding, box[1] - padding, box[2] + padding, box[3] + padding),
        fill=color,
    )
    draw.text((left, top), text, font=label_font, fill="white")


def dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    color: str,
    width: int = 6,
    dash: int = 18,
    gap: int = 10,
) -> None:
    left, top, right, bottom = box
    for start in range(left, right, dash + gap):
        draw.line((start, top, min(start + dash, right), top), fill=color, width=width)
        draw.line((start, bottom, min(start + dash, right), bottom), fill=color, width=width)
    for start in range(top, bottom, dash + gap):
        draw.line((left, start, left, min(start + dash, bottom)), fill=color, width=width)
        draw.line((right, start, right, min(start + dash, bottom)), fill=color, width=width)


def export_panel(
    image: Image.Image,
    crop: tuple[float, float, float, float],
    gt: list[list[float]],
    predictions: list[dict] | None,
    output: Path,
    *,
    letterbox: bool = False,
    draw_gt: bool = True,
) -> None:
    crop_box = aspect_crop(crop, image.width, image.height)
    x1, y1, x2, y2 = crop_box
    cropped = image.crop(crop_box)
    if letterbox:
        scale = min(PANEL_SIZE[0] / cropped.width, PANEL_SIZE[1] / cropped.height)
        rendered_size = (round(cropped.width * scale), round(cropped.height * scale))
        rendered = cropped.resize(rendered_size, Image.Resampling.LANCZOS)
        panel = Image.new("RGB", PANEL_SIZE, "white")
        offset_x = (PANEL_SIZE[0] - rendered_size[0]) // 2
        offset_y = (PANEL_SIZE[1] - rendered_size[1]) // 2
        panel.paste(rendered, (offset_x, offset_y))
        sx = sy = scale
    else:
        panel = cropped.resize(PANEL_SIZE, Image.Resampling.LANCZOS)
        offset_x = offset_y = 0
        sx = PANEL_SIZE[0] / (x2 - x1)
        sy = PANEL_SIZE[1] / (y2 - y1)
    draw = ImageDraw.Draw(panel)

    def transform(box: list[float]) -> tuple[int, int, int, int]:
        return (
            round((box[0] - x1) * sx + offset_x),
            round((box[1] - y1) * sy + offset_y),
            round((box[2] - x1) * sx + offset_x),
            round((box[3] - y1) * sy + offset_y),
        )

    if draw_gt:
        for index, box in enumerate(gt, 1):
            px = transform(box)
            dashed_rectangle(draw, px, COLORS["gt"])
            draw_label(
                draw,
                (max(4, px[0]), max(4, px[1] - 26)),
                f"GT {index}",
                COLORS["gt"],
            )

    if predictions is not None:
        for row in predictions:
            box = row["bbox_xyxy"]
            if min(box[2], x2) <= max(box[0], x1) or min(box[3], y2) <= max(box[1], y1):
                continue
            px = transform(box)
            color = COLORS["tp"] if row["is_tp"] else COLORS["fp"]
            label = f"{'TP' if row['is_tp'] else 'FP'} {row['score']:.2f}"
            draw.rectangle(px, outline=color, width=7)
            draw_label(draw, (max(4, px[0]), max(4, px[1] - 26)), label, color)

    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output, quality=96)


def synonym_panels() -> dict:
    units = json.loads(
        (base.ROOT / "annotations/experiment3/category_prompt_units.json").read_text(
            encoding="utf-8"
        )
    )
    unit = next(
        row for row in units
        if row["file_name"] == "0000314_00000_d_0000142.jpg"
        and row["category"] == "truck"
    )
    raw_predictions = {
        model: representative.load_synonym_predictions(path)
        for model, path in representative.FULL_PREDICTIONS.items()
    }
    raw = {
        model: rows[(unit["file_name"], unit["category"])]
        for model, rows in raw_predictions.items()
    }
    filtered, metrics = representative.prepare(raw, unit["gt_boxes_xyxy"])
    image = Image.open(base.IMAGE_DIR / unit["file_name"]).convert("RGB")
    crop = base.crop_bounds(unit["gt_boxes_xyxy"], filtered, image.width, image.height)
    folder = OUTPUT / "synonym_lorry"
    export_panel(image, crop, unit["gt_boxes_xyxy"], None, folder / "ground_truth.png")
    export_panel(
        image, crop, unit["gt_boxes_xyxy"], filtered["Original SAM 3"],
        folder / "original_sam3.png",
    )
    export_panel(
        image, crop, unit["gt_boxes_xyxy"], filtered["Adapted SAM 3 with replay"],
        folder / "ours.png",
    )
    return {"prompt": unit["synonym_prompt"], "metrics": metrics}


def progression_panels() -> list[dict]:
    manifest = json.loads(base.MANIFEST.read_text(encoding="utf-8"))
    example = next(row for row in manifest if row["example_id"] == "refdrone-specificity-038")
    predictions = {
        model: base.load_predictions(path)
        for model, path in base.PREDICTION_FILES.items()
    }
    image = Image.open(base.IMAGE_DIR / example["file_name"]).convert("RGB")
    # Keep exactly the same full-scene view for all three prompt levels.  The
    # green bus remains visible together with several white buses, so Layers 2
    # and 3 demonstrate attribute/spatial selectivity rather than succeeding
    # simply because only one bus is present in a tight crop.
    full_scene = (0.0, 0.0, float(image.width), float(image.height))
    records = []
    for layer in (1, 2, 3):
        layer_row = example["layers"][layer - 1]
        gt = layer_row["gt_boxes_xyxy"]
        raw = {
            model: predictions[model][(example["example_id"], layer)]
            for model in predictions
        }
        filtered, metrics = representative.prepare(raw, gt)
        folder = OUTPUT / "progressive_bus_full_scene" / f"layer{layer}"
        export_panel(
            image, full_scene, gt, None, folder / "ground_truth.png", letterbox=True
        )
        export_panel(
            image, full_scene, gt, filtered["Original SAM 3"],
            folder / "original_sam3.png", letterbox=True,
        )
        export_panel(
            image, full_scene, gt, filtered["Adapted SAM 3 with replay"],
            folder / "ours.png", letterbox=True,
        )
        records.append({
            "layer": layer,
            "prompt": layer_row["prompt"],
            "gt_count": len(gt),
            "metrics": metrics,
        })
    return records


def main() -> None:
    synonym = synonym_panels()
    progression = progression_panels()
    audit = {
        "panel_size": list(PANEL_SIZE),
        "box_legend": {
            "green": "ground truth",
            "blue": "matched prediction (IoU >= 0.50)",
            "red": "unmatched prediction",
        },
        "synonym_lorry": synonym,
        "progressive_bus": progression,
        "note": "Measured predictions only; frozen thresholds; boxes were not edited.",
    }
    (OUTPUT / "verification.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
