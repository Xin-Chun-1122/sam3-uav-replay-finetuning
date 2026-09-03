#!/usr/bin/env python3
"""Create thesis-ready Experiment 3 adapted-only comparative success figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path("/home/alien/sam3/Master_thesis_experiments")
IMAGE_DIR = ROOT / "datasets/raw/RefDrone/all_image"
MANIFEST = ROOT / "annotations/experiment3/specificity_50_manifest.json"
OUTPUT = ROOT / "figures/experiment3/paper_success_cases"
PREDICTION_FILES = {
    "Original SAM 3": ROOT / "predictions/experiment3/sam3_original_specificity_short.jsonl",
    "Adapted SAM 3 with replay": (
        ROOT / "predictions/experiment3/sam3_adapted_with_replay_specificity_short.jsonl"
    ),
}
THRESHOLDS = {
    "Original SAM 3": 0.48943960666656494,
    "Adapted SAM 3 with replay": 0.4773983955383301,
}
CASES = [
    ("person", "refdrone-specificity-005", 3),
    ("car", "refdrone-specificity-018", 3),
    ("truck", "refdrone-specificity-022", 2),
    ("bus", "refdrone-specificity-038", 3),
    ("motorcycle", "refdrone-specificity-049", 2),
]


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0


def match_predictions(predictions: list[dict], gt: list[list[float]]) -> tuple[list[dict], set[int], int, int, int]:
    matched: set[int] = set()
    annotated = []
    tp = fp = 0
    for prediction in sorted(predictions, key=lambda row: row["score"], reverse=True):
        choices = [
            (box_iou(prediction["bbox_xyxy"], box), index)
            for index, box in enumerate(gt)
            if index not in matched
        ]
        best_iou, best_index = max(choices, default=(0.0, -1))
        is_tp = best_iou >= 0.5
        annotated.append({**prediction, "is_tp": is_tp, "iou": best_iou})
        if is_tp:
            matched.add(best_index)
            tp += 1
        else:
            fp += 1
    return annotated, matched, tp, fp, len(gt) - tp


def load_predictions(path: Path) -> dict[tuple[str, int], list[dict]]:
    output = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "prediction" and row.get("evaluation") == "specificity":
                output[(row["example_id"], int(row["layer"]))] = row["predictions"]
    return output


def crop_bounds(
    gt: list[list[float]],
    predictions_by_model: dict[str, list[dict]],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    # Focus on GT and predictions that are spatially related to any GT. Distant FPs
    # remain represented numerically in the panel title.
    boxes = list(gt)
    for predictions in predictions_by_model.values():
        boxes.extend(
            row["bbox_xyxy"]
            for row in predictions
            if max((box_iou(row["bbox_xyxy"], box) for box in gt), default=0.0) > 0.05
        )
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    box_width, box_height = x2 - x1, y2 - y1
    target_width = max(280.0, box_width * 2.0)
    target_height = max(200.0, box_height * 2.0)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    x1 = max(0.0, center_x - target_width / 2)
    y1 = max(0.0, center_y - target_height / 2)
    x2 = min(float(width), center_x + target_width / 2)
    y2 = min(float(height), center_y + target_height / 2)
    if x2 - x1 < target_width:
        x1 = max(0.0, x2 - target_width)
        x2 = min(float(width), x1 + target_width)
    if y2 - y1 < target_height:
        y1 = max(0.0, y2 - target_height)
        y2 = min(float(height), y1 + target_height)
    return x1, y1, x2, y2


def draw_box(ax, box: list[float], color: str, label: str, linewidth: float = 2.2) -> None:
    x1, y1, x2, y2 = box
    ax.add_patch(Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        fill=False, edgecolor=color, linewidth=linewidth,
    ))
    ax.text(
        x1, y1, label, color="white", fontsize=7, va="bottom",
        bbox={"facecolor": color, "edgecolor": "none", "pad": 1.2, "alpha": 0.9},
    )


def draw_panel(
    ax,
    image,
    gt: list[list[float]],
    predictions: list[dict] | None,
    crop: tuple[float, float, float, float],
    title: str,
) -> None:
    ax.imshow(image)
    for index, box in enumerate(gt, 1):
        draw_box(ax, box, "#00a878", f"GT {index}")
    if predictions is not None:
        for prediction in predictions:
            box = prediction["bbox_xyxy"]
            intersects_crop = (
                min(box[2], crop[2]) > max(box[0], crop[0])
                and min(box[3], crop[3]) > max(box[1], crop[1])
            )
            if not intersects_crop:
                continue
            color = "#d62728" if not prediction["is_tp"] else "#1f77b4"
            label = f"{'TP' if prediction['is_tp'] else 'FP'} {prediction['score']:.2f}"
            draw_box(ax, box, color, label)
    ax.set_xlim(crop[0], crop[2])
    ax.set_ylim(crop[3], crop[1])
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        row["example_id"]: row
        for row in json.loads(MANIFEST.read_text(encoding="utf-8"))
    }
    raw_predictions = {
        model: load_predictions(path)
        for model, path in PREDICTION_FILES.items()
    }
    prepared = []
    readme_rows = []

    for category, example_id, layer in CASES:
        example = manifest[example_id]
        layer_row = example["layers"][layer - 1]
        gt = layer_row["gt_boxes_xyxy"]
        image = Image.open(IMAGE_DIR / example["file_name"]).convert("RGB")
        filtered = {}
        matched = {}
        for model in PREDICTION_FILES:
            rows = [
                row for row in raw_predictions[model][(example_id, layer)]
                if row["score"] >= THRESHOLDS[model]
            ]
            annotated, _, tp, fp, fn = match_predictions(rows, gt)
            filtered[model] = annotated
            matched[model] = {"tp": tp, "fp": fp, "fn": fn}
        if not (
            matched["Adapted SAM 3 with replay"]["tp"] > matched["Original SAM 3"]["tp"]
        ):
            raise RuntimeError(f"{example_id} is no longer an adapted comparative win")
        crop = crop_bounds(gt, filtered, image.width, image.height)
        prepared.append((category, example, layer, layer_row, image, gt, filtered, matched, crop))

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.suptitle(
            f"Adapted-only success ({category}, Layer {layer}) — prompt: “{layer_row['prompt']}”",
            fontsize=14,
        )
        draw_panel(axes[0], image, gt, None, crop, f"Ground truth ({len(gt)} target(s))")
        for axis, model in zip(axes[1:], PREDICTION_FILES):
            metrics = matched[model]
            draw_panel(
                axis, image, gt, filtered[model], crop,
                f"{model}\nTP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}",
            )
        stem = f"qualitative_adapted_win_{category}_layer{layer}"
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        readme_rows.append(
            f"| {category} | {layer} | `{example_id}` | {layer_row['prompt']} | "
            f"{matched['Original SAM 3']['tp']}/{matched['Original SAM 3']['fp']}/"
            f"{matched['Original SAM 3']['fn']} | "
            f"{matched['Adapted SAM 3 with replay']['tp']}/"
            f"{matched['Adapted SAM 3 with replay']['fp']}/"
            f"{matched['Adapted SAM 3 with replay']['fn']} | `{stem}.png` |"
        )

    fig, axes = plt.subplots(len(prepared), 3, figsize=(14, 3.6 * len(prepared)))
    fig.suptitle(
        "Experiment 3: Adapted SAM 3 with replay comparative success cases\n"
        "Green: GT; Blue: matched prediction; Red: unmatched prediction",
        fontsize=16,
    )
    for row_index, (category, _, layer, layer_row, image, gt, filtered, matched, crop) in enumerate(prepared):
        draw_panel(
            axes[row_index, 0], image, gt, None, crop,
            f"{category} · Layer {layer}\n“{layer_row['prompt']}”",
        )
        for column, model in enumerate(PREDICTION_FILES, 1):
            metrics = matched[model]
            draw_panel(
                axes[row_index, column], image, gt, filtered[model], crop,
                f"{model}\nTP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}",
            )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.0, w_pad=1.0)
    fig.savefig(OUTPUT / "qualitative_adapted_comparative_wins_all_categories.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "qualitative_adapted_comparative_wins_all_categories.pdf", bbox_inches="tight")
    plt.close(fig)

    readme = [
        "# Experiment 3 comparative success figures",
        "",
        "Every case was selected from measured predictions using the frozen model-specific "
        "operating thresholds and IoU >= 0.50. Selection requires Adapted SAM 3 with replay "
        "to have more true positives than Original SAM 3. Boxes were not edited.",
        "",
        "TP/FP/FN below are written as `TP/FP/FN`.",
        "",
        "| Category | Layer | Example | Prompt | Original | Adapted | Figure |",
        "|---|---:|---|---|---:|---:|---|",
        *readme_rows,
        "",
        "Recommended main-paper figure: "
        "`qualitative_adapted_comparative_wins_all_categories.png`.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
