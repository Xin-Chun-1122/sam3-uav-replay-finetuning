#!/usr/bin/env python3
"""Export a verified single-target RefDrone synonym-prompt example."""

from pathlib import Path
from PIL import Image, ImageDraw
import json


ROOT = Path("/home/alien/sam3/Master_thesis_experiments")
IMAGE_NAME = "9999952_00000_d_0000207.jpg"
TASK_ID = f"synonym:{IMAGE_NAME}:truck"
ORIGINAL_THRESHOLD = 0.48943960666656494
OURS_THRESHOLD = 0.4773983955383301
IOU_THRESHOLD = 0.50
OUT = ROOT / "figures/experiment3/ppt_single_target_lorry_case"


def load_unit():
    with (ROOT / "annotations/experiment3/category_prompt_units.json").open() as f:
        units = json.load(f)
    return next(u for u in units if u["unit_id"] == f"{IMAGE_NAME}:truck")


def load_prediction(path):
    with path.open() as f:
        for line in f:
            item = json.loads(line)
            if item.get("task_id") == TASK_ID:
                return item
    raise RuntimeError(f"Missing task: {TASK_ID}")


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter else 0.0


def draw_box(image, box, color, width=5, dashed=False):
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = box
    if not dashed:
        for k in range(width):
            draw.rectangle((x1-k, y1-k, x2+k, y2+k), outline=color)
        return
    dash, gap = 14, 8
    for x in range(int(x1), int(x2), dash + gap):
        draw.line((x, y1, min(x + dash, x2), y1), fill=color, width=width)
        draw.line((x, y2, min(x + dash, x2), y2), fill=color, width=width)
    for y in range(int(y1), int(y2), dash + gap):
        draw.line((x1, y, x1, min(y + dash, y2)), fill=color, width=width)
        draw.line((x2, y, x2, min(y + dash, y2)), fill=color, width=width)


def crop_and_resize(image, boxes, crop):
    x0, y0, x1, y1 = crop
    panel = image.crop(crop)
    sx, sy = 1200 / (x1 - x0), 800 / (y1 - y0)
    panel = panel.resize((1200, 800), Image.Resampling.LANCZOS)
    converted = []
    for box in boxes:
        converted.append([
            (box[0] - x0) * sx,
            (box[1] - y0) * sy,
            (box[2] - x0) * sx,
            (box[3] - y0) * sy,
        ])
    return panel, converted


def main():
    unit = load_unit()
    original = load_prediction(ROOT / "predictions/experiment3/sam3_original.jsonl")
    ours = load_prediction(ROOT / "predictions/experiment3/sam3_adapted_with_replay.jsonl")
    gt = unit["gt_boxes_xyxy"]
    op = [p for p in original["predictions"] if p["score"] >= ORIGINAL_THRESHOLD]
    ap = [p for p in ours["predictions"] if p["score"] >= OURS_THRESHOLD]

    assert len(gt) == 1, gt
    assert len(op) == 0, op
    assert len(ap) == 1, ap
    score_iou = iou(gt[0], ap[0]["bbox_xyxy"])
    assert score_iou >= IOU_THRESHOLD, score_iou

    OUT.mkdir(parents=True, exist_ok=True)
    image = Image.open(ROOT / "datasets/raw/RefDrone/all_image" / IMAGE_NAME).convert("RGB")
    # Moderate landscape crop: the target remains visible while nearby vehicles
    # provide enough scene context to show that only the lorry is selected.
    crop = (250, 105, 950, 572)

    gt_panel, gt_boxes = crop_and_resize(image, gt, crop)
    draw_box(gt_panel, gt_boxes[0], "#00B894", width=6, dashed=True)
    gt_panel.save(OUT / "ground_truth.png", quality=95)

    original_panel, _ = crop_and_resize(image, [], crop)
    original_panel.save(OUT / "original_sam3.png", quality=95)

    ours_panel, ours_boxes = crop_and_resize(image, [ap[0]["bbox_xyxy"]], crop)
    draw_box(ours_panel, ours_boxes[0], "#00A8FF", width=6, dashed=False)
    ours_panel.save(OUT / "ours.png", quality=95)

    manifest = {
        "image": IMAGE_NAME,
        "evaluation": "synonym prompt",
        "category": "truck",
        "prompt": "lorry",
        "gt_count": len(gt),
        "original_filtered_prediction_count": len(op),
        "ours_filtered_prediction_count": len(ap),
        "ours_score": ap[0]["score"],
        "ours_iou": score_iou,
        "thresholds": {"original": ORIGINAL_THRESHOLD, "ours": OURS_THRESHOLD},
        "crop_xyxy": crop,
        "note": "Boxes are exported from saved predictions; no prediction was manually added or removed.",
    }
    with (OUT / "verification.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
