#!/usr/bin/env python3
"""Render Original SAM 3 versus Ours on self-collected UAV images.

All boxes are read directly from saved model predictions.  The script does
not add, remove, or relocate detections.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def font(size: int):
    return ImageFont.truetype(str(FONT_PATH), size) if FONT_PATH.is_file() else ImageFont.load_default()


def load(path: Path) -> tuple[dict, dict[tuple[str, str], dict]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return records[0], {(r["image_name"], r["prompt"]): r for r in records[1:]}


def clipped(box: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, round(x1))),
        max(0, min(height - 1, round(y1))),
        max(0, min(width - 1, round(x2))),
        max(0, min(height - 1, round(y2))),
    )


def render(source: Image.Image, predictions: list[dict], threshold: float, color: str) -> tuple[Image.Image, int]:
    output = source.copy()
    draw = ImageDraw.Draw(output)
    width = max(4, round(min(source.size) / 260))
    kept = [p for p in predictions if p["score"] >= threshold]
    for prediction in kept:
        box = clipped(prediction["bbox_xyxy"], source.size)
        if box[2] > box[0] and box[3] > box[1]:
            draw.rectangle(box, outline=color, width=width)
    return output, len(kept)


def comparison_canvas(
    source: Image.Image,
    prompt: str,
    original_predictions: list[dict],
    ours_predictions: list[dict],
    original_threshold: float,
    ours_threshold: float,
) -> Image.Image:
    panel_width = 900
    panel_height = round(panel_width * source.height / source.width)
    original, original_count = render(source, original_predictions, original_threshold, "#E63946")
    ours, ours_count = render(source, ours_predictions, ours_threshold, "#0077B6")
    original = original.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    ours = ours.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    top = 126
    margin = 24
    canvas = Image.new("RGB", (panel_width * 2 + margin * 3, panel_height + top + margin), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 14), f"Self-collected UAV data | prompt: {prompt}", fill="black", font=font(34))
    draw.text((margin, 66), f"Original SAM 3 ({original_count} predictions)", fill="#E63946", font=font(28))
    draw.text((panel_width + margin * 2, 66), f"Ours ({ours_count} predictions)", fill="#0077B6", font=font(28))
    canvas.paste(original, (margin, top))
    canvas.paste(ours, (panel_width + margin * 2, top))
    return canvas


def paper_person_figure(rows: list[dict], output: Path) -> None:
    """Create a compact paper figure from already rendered, unedited panels."""
    panel_width = 820
    row_height = round(panel_width * rows[0]["source"].height / rows[0]["source"].width)
    left = 150
    top = 90
    gap = 18
    canvas = Image.new(
        "RGB",
        (left + panel_width * 2 + gap * 3, top + (row_height + gap) * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((left + gap, 18), "Original SAM 3", fill="#E63946", font=font(30))
    draw.text((left + panel_width + gap * 2, 18), "Ours", fill="#0077B6", font=font(30))
    for index, row in enumerate(rows, start=1):
        y = top + (index - 1) * (row_height + gap)
        draw.text((18, y + row_height // 2 - 18), f"Image {index}", fill="black", font=font(27))
        original = row["original"].resize((panel_width, row_height), Image.Resampling.LANCZOS)
        ours = row["ours"].resize((panel_width, row_height), Image.Resampling.LANCZOS)
        canvas.paste(original, (left + gap, y))
        canvas.paste(ours, (left + panel_width + gap * 2, y))
    canvas.save(output, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    original_meta, original = load(args.original)
    ours_meta, ours = load(args.ours)
    original_threshold = float(original_meta["operating_threshold"])
    ours_threshold = float(ours_meta["operating_threshold"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paper_panels = args.output_dir / "paper_panels"
    paper_panels.mkdir(parents=True, exist_ok=True)

    manifest = []
    selected_person_rows = []
    for key in sorted(ours):
        if key not in original:
            raise ValueError(f"Missing Original SAM 3 record: {key}")
        image_name, prompt = key
        source = Image.open(args.images_dir / image_name).convert("RGB")
        canvas = comparison_canvas(
            source,
            prompt,
            original[key]["predictions"],
            ours[key]["predictions"],
            original_threshold,
            ours_threshold,
        )
        stem = Path(image_name).stem.replace(" ", "_").replace("(", "").replace(")", "")
        output = args.output_dir / f"{stem}_prompt_{prompt}_original_vs_ours.png"
        canvas.save(output, quality=95)
        original_count = sum(p["score"] >= original_threshold for p in original[key]["predictions"])
        ours_count = sum(p["score"] >= ours_threshold for p in ours[key]["predictions"])
        manifest.append(
            {
                "image_name": image_name,
                "prompt": prompt,
                "original_prediction_count": original_count,
                "ours_prediction_count": ours_count,
                "output": str(output.resolve()),
            }
        )

        if prompt == "person" and stem in {"image_98", "image_99", "image_100"}:
            original_panel, _ = render(
                source, original[key]["predictions"], original_threshold, "#E63946"
            )
            ours_panel, _ = render(source, ours[key]["predictions"], ours_threshold, "#0077B6")
            original_panel.save(paper_panels / f"{stem}_person_original_sam3.png", quality=95)
            ours_panel.save(paper_panels / f"{stem}_person_ours.png", quality=95)
            selected_person_rows.append(
                {"stem": stem, "source": source, "original": original_panel, "ours": ours_panel}
            )

    with (args.output_dir / "comparison_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_name", "prompt", "original_prediction_count", "ours_prediction_count"],
        )
        writer.writeheader()
        for row in manifest:
            writer.writerow({name: row[name] for name in writer.fieldnames})

    selected_person_rows.sort(key=lambda row: int(row["stem"].split("_")[-1]))
    if len(selected_person_rows) == 3:
        paper_person_figure(
            selected_person_rows,
            args.output_dir / "paper_self_collected_person_original_vs_ours.png",
        )

    audit = {
        "evaluation_type": "qualitative_external_generalization_without_ground_truth",
        "ground_truth_available": False,
        "manual_box_editing": False,
        "original_checkpoint": original_meta["checkpoint"],
        "ours_checkpoint": ours_meta["checkpoint"],
        "original_threshold": original_threshold,
        "ours_threshold": ours_threshold,
        "comparisons": manifest,
    }
    (args.output_dir / "comparison_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
