#!/usr/bin/env python3
"""Render auditable, prompt-specific figures for the self-collected UAV set."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "person": "#FFD400",
    "car": "#00B7FF",
    "bus": "#E83E8C",
}


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def clip_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, round(x1))),
        max(0, min(height - 1, round(y1))),
        max(0, min(width - 1, round(x2))),
        max(0, min(height - 1, round(y2))),
    )


def draw_predictions(image: Image.Image, prompt: str, predictions: list[dict]) -> Image.Image:
    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    color = COLORS[prompt]
    label_font = font(18)
    line_width = max(3, round(min(image.size) / 280))
    for prediction in predictions:
        box = clip_box(prediction["bbox_xyxy"], image.width, image.height)
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle(box, outline=color, width=line_width)
        label = f"{prompt} {prediction['score']:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=label_font)
        text_width = right - left
        text_height = bottom - top
        label_y = max(0, y1 - text_height - 6)
        draw.rectangle((x1, label_y, x1 + text_width + 8, label_y + text_height + 6), fill=color)
        draw.text((x1 + 4, label_y + 2), label, fill="black", font=label_font)
    return panel


def make_contact_sheet(items: list[tuple[str, Image.Image]], output: Path, title: str) -> None:
    thumb_width = 720
    title_height = 72
    caption_height = 42
    margin = 18
    columns = 2
    thumb_height = round(thumb_width * items[0][1].height / items[0][1].width)
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (
            margin + columns * (thumb_width + margin),
            title_height + margin + rows * (thumb_height + caption_height + margin),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 14), title, fill="black", font=font(34))
    for index, (caption, image) in enumerate(items):
        row, col = divmod(index, columns)
        x = margin + col * (thumb_width + margin)
        y = title_height + margin + row * (thumb_height + caption_height + margin)
        resized = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        canvas.paste(resized, (x, y))
        draw.text((x, y + thumb_height + 6), caption, fill="black", font=font(22))
    canvas.save(output, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()]
    metadata = records[0]
    threshold = float(metadata["operating_threshold"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = args.output_dir / "annotated_prompt_specific"
    clean_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[str, Image.Image]]] = defaultdict(list)
    count_rows = []
    for record in records[1:]:
        prompt = record["prompt"]
        predictions = [item for item in record["predictions"] if item["score"] >= threshold]
        source = Image.open(args.images_dir / record["image_name"]).convert("RGB")
        rendered = draw_predictions(source, prompt, predictions)
        stem = Path(record["image_name"]).stem.replace(" ", "_").replace("(", "").replace(")", "")
        output_name = f"{stem}_prompt_{prompt}.png"
        rendered.save(clean_dir / output_name)
        caption = f"{record['image_name']} | prompt: {prompt} | predictions: {len(predictions)}"
        grouped[prompt].append((caption, rendered))
        count_rows.append(
            {
                "image_name": record["image_name"],
                "prompt": prompt,
                "prediction_count": len(predictions),
                "operating_threshold": f"{threshold:.10f}",
            }
        )

    for prompt, items in grouped.items():
        make_contact_sheet(
            items,
            args.output_dir / f"contact_sheet_prompt_{prompt}.png",
            f"Self-collected UAV images — Ours — prompt: {prompt}",
        )

    with (args.output_dir / "prediction_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(count_rows[0]))
        writer.writeheader()
        writer.writerows(count_rows)

    audit = {
        "model": metadata["model_name"],
        "checkpoint": metadata["checkpoint"],
        "evaluation_type": metadata["evaluation_type"],
        "operating_threshold": threshold,
        "threshold_source": metadata["threshold_source"],
        "manual_box_editing": False,
        "ground_truth_available": False,
        "quantitative_metrics_permitted": False,
        "source_predictions": str(args.predictions.resolve()),
        "rendered_panel_count": len(count_rows),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
