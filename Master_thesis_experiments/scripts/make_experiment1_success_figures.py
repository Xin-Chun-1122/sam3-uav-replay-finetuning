#!/usr/bin/env python3
"""Create thesis-ready Experiment 1 comparative success figures."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path("/home/alien/sam3/Master_thesis_experiments")
IMAGE_DIR = Path("/home/alien/Downloads/VisDrone2019-DET-test-dev/images")
OUTPUT = ROOT / "figures/experiment1/paper_success_cases"
PREDICTION_FILES = {
    "YOLOv8x": ROOT / "predictions/experiment1/test/yolo_test1610.jsonl",
    "YOLOv8x-WorldV2": ROOT / "predictions/experiment1/test/yolo_world_test1610.jsonl",
    "Original SAM 3": ROOT / "predictions/experiment1/test/sam3_original_test1610.jsonl",
    "Adapted SAM 3 without replay": (
        ROOT / "predictions/experiment1/test/sam3_adapted_no_replay_test1610.jsonl"
    ),
    "Adapted SAM 3 with replay": (
        ROOT / "predictions/experiment1/test/sam3_adapted_with_replay_test1610.jsonl"
    ),
}
THRESHOLDS = {
    "YOLOv8x": 0.06027992069721222,
    "YOLOv8x-WorldV2": 0.052561935037374496,
    "Original SAM 3": 0.48943960666656494,
    "Adapted SAM 3 without replay": 2.187017789090362e-10,
    "Adapted SAM 3 with replay": 0.4773983955383301,
}
CASES = [
    {
        "scale": "Tiny",
        "image_name": "9999938_00000_d_0000210.jpg",
        "category": "motorcycle",
        "gt": [1105.0, 267.0, 1125.0, 279.0],
    },
    {
        "scale": "Small",
        "image_name": "0000259_03000_d_0000007.jpg",
        "category": "truck",
        "gt": [675.0, 133.0, 706.0, 179.0],
    },
    {
        "scale": "Regular",
        "image_name": "9999938_00000_d_0000354.jpg",
        "category": "bus",
        "gt": [779.0, 288.0, 818.0, 426.0],
    },
]


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0


def load_case_predictions(path: Path) -> dict[tuple[str, str], list[dict]]:
    required = {(case["image_name"], case["category"]) for case in CASES}
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = (record.get("image_name"), record.get("category"))
            if record.get("type") == "prediction" and key in required:
                rows[key] = record["predictions"]
    if set(rows) != required:
        raise RuntimeError(f"Missing case predictions in {path}")
    return rows


def crop_bounds(
    gt: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = gt
    width, height = x2 - x1, y2 - y1
    crop_width = max(260.0, width * 4.0)
    crop_height = max(190.0, height * 4.0)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    left = max(0.0, center_x - crop_width / 2)
    top = max(0.0, center_y - crop_height / 2)
    right = min(float(image_width), center_x + crop_width / 2)
    bottom = min(float(image_height), center_y + crop_height / 2)
    if right - left < crop_width:
        left = max(0.0, right - crop_width)
        right = min(float(image_width), left + crop_width)
    if bottom - top < crop_height:
        top = max(0.0, bottom - crop_height)
        bottom = min(float(image_height), top + crop_height)
    return left, top, right, bottom


def intersects(box: list[float], crop: tuple[float, float, float, float]) -> bool:
    return (
        min(box[2], crop[2]) > max(box[0], crop[0])
        and min(box[3], crop[3]) > max(box[1], crop[1])
    )


def draw_box(ax, box: list[float], color: str, label: str, linewidth: float = 2.4) -> None:
    x1, y1, x2, y2 = box
    ax.add_patch(Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        fill=False, edgecolor=color, linewidth=linewidth,
    ))
    ax.text(
        x1, y1, label, fontsize=7.5, color="white", va="bottom",
        bbox={"facecolor": color, "edgecolor": "none", "pad": 1.3, "alpha": 0.92},
    )


def draw_panel(
    ax,
    image: Image.Image,
    gt: list[float],
    predictions: list[dict] | None,
    crop: tuple[float, float, float, float],
    title: str,
    prediction_color: str,
) -> None:
    ax.imshow(image)
    draw_box(ax, gt, "#00a878", "GT")
    if predictions is not None:
        for prediction in predictions:
            if not intersects(prediction["bbox_xyxy"], crop):
                continue
            overlap = iou(prediction["bbox_xyxy"], gt)
            draw_box(
                ax,
                prediction["bbox_xyxy"],
                prediction_color,
                f"Pred {prediction['score']:.2f} · IoU {overlap:.2f}",
            )
    ax.set_xlim(crop[0], crop[2])
    ax.set_ylim(crop[3], crop[1])
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def best_prediction(predictions: list[dict], gt: list[float]) -> list[dict]:
    """Return only the prediction most relevant to the selected GT."""
    if not predictions:
        return []
    best = max(predictions, key=lambda row: (iou(row["bbox_xyxy"], gt), row["score"]))
    return [best] if iou(best["bbox_xyxy"], gt) > 0 else []


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = {
        model: load_case_predictions(path)
        for model, path in PREDICTION_FILES.items()
    }
    prepared = []
    readme_rows = []
    for case in CASES:
        key = (case["image_name"], case["category"])
        image = Image.open(IMAGE_DIR / case["image_name"]).convert("RGB")
        x1, y1, x2, y2 = case["gt"]
        size = math.sqrt((x2 - x1) * (y2 - y1))
        crop = crop_bounds(case["gt"], image.width, image.height)
        selected = {}
        summary = {}
        for model in PREDICTION_FILES:
            predictions = [
                row for row in raw[model][key]
                if row["score"] >= THRESHOLDS[model]
            ]
            selected[model] = predictions
            best = max(
                (
                    (iou(row["bbox_xyxy"], case["gt"]), row["score"], row["bbox_xyxy"])
                    for row in predictions
                ),
                default=(0.0, 0.0, None),
            )
            summary[model] = {"best_iou": best[0], "best_score": best[1]}
        if summary["Adapted SAM 3 with replay"]["best_iou"] < 0.5:
            raise RuntimeError(f"{case['scale']} case is no longer an Adapted success")
        failed_baselines = [
            model for model in PREDICTION_FILES
            if model != "Adapted SAM 3 with replay" and summary[model]["best_iou"] >= 0.5
        ]
        if failed_baselines:
            raise RuntimeError(
                f"{case['scale']} is not a strict all-baseline win: {failed_baselines}"
            )
        prepared.append((case, image, size, crop, selected, summary))

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.suptitle(
            f"{case['scale']} success case — {case['category']}, "
            f"s = {size:.1f} pixels",
            fontsize=14,
        )
        draw_panel(axes[0], image, case["gt"], None, crop, "Ground truth", "#00a878")
        draw_panel(
            axes[1], image, case["gt"], selected["Original SAM 3"], crop,
            f"Original SAM 3\nMissed (best IoU={summary['Original SAM 3']['best_iou']:.2f})",
            "#d62728",
        )
        draw_panel(
            axes[2], image, case["gt"], selected["Adapted SAM 3 with replay"], crop,
            "Adapted SAM 3 with replay\n"
            f"Correct (best IoU={summary['Adapted SAM 3 with replay']['best_iou']:.2f})",
            "#1f77b4",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        stem = f"qualitative_adapted_win_{case['scale'].lower()}_{case['category']}"
        fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        readme_rows.append(
            f"| {case['scale']} | {case['category']} | {size:.2f} | "
            f"{summary['Original SAM 3']['best_iou']:.3f} | "
            f"{summary['Adapted SAM 3 with replay']['best_iou']:.3f} | "
            f"`{stem}.png` |"
        )

    fig, axes = plt.subplots(len(prepared), 3, figsize=(14, 10.8))
    fig.suptitle(
        "Experiment 1: Adapted SAM 3 with replay comparative success cases\n"
        "Green: ground truth; Red: Original prediction; Blue: Adapted prediction",
        fontsize=16,
    )
    for row_index, (case, image, size, crop, selected, summary) in enumerate(prepared):
        draw_panel(
            axes[row_index, 0], image, case["gt"], None, crop,
            f"{case['scale']} · {case['category']} · s={size:.1f}", "#00a878",
        )
        draw_panel(
            axes[row_index, 1], image, case["gt"], selected["Original SAM 3"], crop,
            f"Original SAM 3\nMissed · best IoU={summary['Original SAM 3']['best_iou']:.2f}",
            "#d62728",
        )
        draw_panel(
            axes[row_index, 2], image, case["gt"], selected["Adapted SAM 3 with replay"], crop,
            "Adapted SAM 3 with replay\n"
            f"Correct · best IoU={summary['Adapted SAM 3 with replay']['best_iou']:.2f}",
            "#1f77b4",
        )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.0, w_pad=1.0)
    fig.savefig(OUTPUT / "qualitative_adapted_comparative_wins_all_scales.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "qualitative_adapted_comparative_wins_all_scales.pdf", bbox_inches="tight")
    plt.close(fig)

    model_order = list(PREDICTION_FILES)
    fig, axes = plt.subplots(len(prepared), len(model_order), figsize=(18, 10.2))
    fig.suptitle(
        "Experiment 1: Strict comparative wins across object scales\n"
        "Green: ground truth; Red: missed localization; Blue: correct localization (IoU ≥ 0.50)",
        fontsize=16,
    )
    for row_index, (case, image, size, crop, selected, summary) in enumerate(prepared):
        for column_index, model in enumerate(model_order):
            is_adapted = model == "Adapted SAM 3 with replay"
            status = "Correct" if summary[model]["best_iou"] >= 0.5 else "Missed"
            title = (
                f"{model}\n{status} · best IoU={summary[model]['best_iou']:.2f}"
                if row_index == 0
                else f"{status} · best IoU={summary[model]['best_iou']:.2f}"
            )
            if column_index == 0:
                title = (
                    f"{case['scale']} · {case['category']} · s={size:.1f}\n"
                    f"{title}"
                )
            draw_panel(
                axes[row_index, column_index],
                image,
                case["gt"],
                best_prediction(selected[model], case["gt"]),
                crop,
                title,
                "#1f77b4" if is_adapted else "#d62728",
            )
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.0, w_pad=0.7)
    strict_stem = "qualitative_adapted_strict_wins_all_baselines_all_scales"
    fig.savefig(OUTPUT / f"{strict_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{strict_stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    readme = [
        "# Experiment 1 comparative success figures",
        "",
        "Cases are selected from the saved measured predictions. A case is retained only "
        "when Adapted SAM 3 with replay matches the GT at IoU >= 0.50 and every other "
        "baseline does not, using each model's frozen validation threshold. Boxes are not edited.",
        "",
        "| Scale | Category | s (pixels) | Original best IoU | Adapted best IoU | Figure |",
        "|---|---|---:|---:|---:|---|",
        *readme_rows,
        "",
        "Recommended presentation figure: "
        "`qualitative_adapted_strict_wins_all_baselines_all_scales.png`. Each model panel "
        "shows the highest-IoU prediction for the selected GT to avoid clutter.",
        "",
        "These are representative qualitative successes. Aggregate claims must still be "
        "based on the complete quantitative test table.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
