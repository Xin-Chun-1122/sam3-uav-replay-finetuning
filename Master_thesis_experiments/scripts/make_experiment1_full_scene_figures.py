#!/usr/bin/env python3
"""Create full-scene Experiment 1 qualitative comparisons with all targets."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path("/home/alien/sam3/Master_thesis_experiments")
DATASET = Path("/home/alien/Downloads/VisDrone2019-DET-test-dev")
OUTPUT = ROOT / "figures/experiment1/paper_full_scene_cases"
EVALUATOR_PATH = ROOT / "scripts/evaluate_visdrone_predictions.py"
MODEL_FILES = {
    "YOLOv8x": ("yolo_test1610.jsonl", 0.06027992069721222),
    "YOLOv8x-WorldV2": ("yolo_world_test1610.jsonl", 0.052561935037374496),
    "Original SAM 3": ("sam3_original_test1610.jsonl", 0.48943960666656494),
    "Adapted SAM 3 without replay": (
        "sam3_adapted_no_replay_test1610.jsonl",
        2.187017789090362e-10,
    ),
    "Adapted SAM 3 with replay (Ours)": (
        "sam3_adapted_with_replay_test1610.jsonl",
        0.4773983955383301,
    ),
}
SCENES = [
    {
        "scale": "Tiny",
        "image_name": "9999976_00000_d_0000002.jpg",
        "expected_scale_counts": {"tiny": 11, "small": 0, "regular": 0},
    },
    {
        "scale": "Small",
        "image_name": "9999938_00000_d_0000461.jpg",
        "expected_scale_counts": {"tiny": 0, "small": 18, "regular": 0},
    },
    {
        "scale": "Regular-dominant",
        "image_name": "9999952_00000_d_0000211.jpg",
        "expected_scale_counts": {"tiny": 0, "small": 8, "regular": 10},
    },
]
SCALE_COLORS = {
    "tiny": "#f0a202",
    "small": "#2ca02c",
    "regular": "#9467bd",
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


def annotate_predictions(ev, predictions: list[dict], targets: list[dict]) -> tuple[list[dict], set[int]]:
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
    return annotated, matched


def prepare_scene_data():
    ev = load_evaluator()
    image_names = {scene["image_name"] for scene in SCENES}
    gt, ignore_integrals = ev.load_gt(DATASET, image_names)
    model_predictions = {}
    for model, (filename, threshold) in MODEL_FILES.items():
        predictions = load_selected_predictions(
            ROOT / "predictions/experiment1/test" / filename,
            image_names,
        )
        predictions = ev.limit_detections_per_image(predictions, 500)
        predictions = ev.drop_predictions_in_ignored_regions(
            predictions, ignore_integrals
        )
        predictions = {
            key: [row for row in rows if row["score"] >= threshold]
            for key, rows in predictions.items()
        }
        model_predictions[model] = predictions

    prepared = []
    verification = {}
    for scene in SCENES:
        image_name = scene["image_name"]
        image = Image.open(DATASET / "images" / image_name).convert("RGB")
        gt_by_category = {
            category: gt.get((image_name, category), [])
            for category in ev.CATEGORIES
        }
        scale_counts = {
            scale: sum(
                1
                for targets in gt_by_category.values()
                for target in targets
                if not target["ignored"] and target["scale"] == scale
            )
            for scale in ev.SCALES
        }
        if scale_counts != scene["expected_scale_counts"]:
            raise RuntimeError(
                f"{image_name}: scale counts changed: {scale_counts}"
            )

        per_model = {}
        for model, (_, threshold) in MODEL_FILES.items():
            predictions_by_category = {
                category: model_predictions[model].get((image_name, category), [])
                for category in ev.CATEGORIES
            }
            annotated_by_category = {}
            matched_by_category = {}
            for category in ev.CATEGORIES:
                annotated, matched = annotate_predictions(
                    ev,
                    predictions_by_category[category],
                    gt_by_category[category],
                )
                annotated_by_category[category] = annotated
                matched_by_category[category] = matched
            prediction_subset = {
                (image_name, category): predictions_by_category[category]
                for category in ev.CATEGORIES
            }
            gt_subset = {
                (image_name, category): gt_by_category[category]
                for category in ev.CATEGORIES
            }
            labelled, gt_count = ev.label_predictions(prediction_subset, gt_subset)
            metrics = ev.counts_at_threshold(labelled, gt_count, threshold)
            per_model[model] = {
                "annotated": annotated_by_category,
                "matched": matched_by_category,
                "metrics": metrics,
            }
        prepared.append({
            "scene": scene,
            "image": image,
            "gt": gt_by_category,
            "scale_counts": scale_counts,
            "models": per_model,
        })
        verification[image_name] = {
            "scale_counts": scale_counts,
            "models": {
                model: values["metrics"]
                for model, values in per_model.items()
            },
        }
    return prepared, verification


def draw_box(ax, box: list[float], color: str, label: str, linewidth: float, linestyle: str = "-"):
    x1, y1, x2, y2 = box
    ax.add_patch(Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
    ))
    if label:
        ax.text(
            x1,
            y1,
            label,
            fontsize=5.5,
            color="white",
            va="bottom",
            bbox={
                "facecolor": color,
                "edgecolor": "none",
                "pad": 0.8,
                "alpha": 0.88,
            },
        )


def draw_ground_truth(ax, item: dict, title: str):
    ax.imshow(item["image"])
    for category, targets in item["gt"].items():
        for target in targets:
            if target["ignored"]:
                continue
            scale = target["scale"]
            draw_box(
                ax,
                target["bbox"],
                SCALE_COLORS[scale],
                f"{category}·{scale[0].upper()}",
                1.5,
            )
    counts = item["scale_counts"]
    ax.set_title(
        f"{title}\nT/S/R={counts['tiny']}/{counts['small']}/{counts['regular']}",
        fontsize=9,
    )
    ax.axis("off")


def draw_model(ax, item: dict, model: str, show_model_name: bool):
    ax.imshow(item["image"])
    # All evaluable targets remain visible in every panel.
    for targets in item["gt"].values():
        for target in targets:
            if not target["ignored"]:
                draw_box(ax, target["bbox"], "#00a878", "", 0.8, "--")

    annotated = [
        (category, prediction)
        for category, predictions in item["models"][model]["annotated"].items()
        for prediction in predictions
    ]
    true_positives = [
        (category, prediction)
        for category, prediction in annotated
        if prediction["status"] == "TP"
    ]
    false_positives = [
        (category, prediction)
        for category, prediction in annotated
        if prediction["status"] == "FP"
    ]
    fp_display_limit = 30 if model == "Adapted SAM 3 without replay" else len(false_positives)
    displayed_false_positives = false_positives[:fp_display_limit]
    for category, prediction in displayed_false_positives:
        draw_box(
            ax,
            prediction["bbox_xyxy"],
            "#d62728",
            f"FP {category}",
            1.0,
        )
    for category, prediction in true_positives:
        draw_box(
            ax,
            prediction["bbox_xyxy"],
            "#1f77b4",
            f"{category} {prediction['score']:.2f}",
            1.5,
        )

    metric = item["models"][model]["metrics"]
    model_label = model.replace("Adapted SAM 3 with replay (Ours)", "Ours")
    title = (
        f"{model_label}\n" if show_model_name else ""
    ) + f"TP/FP/FN={metric['tp']}/{metric['fp']}/{metric['fn']}"
    if len(false_positives) > fp_display_limit:
        title += f"\nFP shown {fp_display_limit}/{len(false_positives)}"
    ax.set_title(
        title,
        fontsize=8.5,
        fontweight="bold" if model.endswith("(Ours)") else "normal",
        bbox=(
            {"facecolor": "#e5d6ef", "edgecolor": "none", "pad": 2.0}
            if model.endswith("(Ours)")
            else None
        ),
    )
    ax.axis("off")


def render_grid(prepared: list[dict], models: list[str], stem: str, title: str):
    columns = 1 + len(models)
    fig, axes = plt.subplots(
        len(prepared),
        columns,
        figsize=(4.0 * columns, 3.25 * len(prepared)),
    )
    fig.suptitle(
        title + "\n"
        "All six target classes; green dashed=GT, blue=TP, red=FP",
        fontsize=16,
    )
    for row_index, item in enumerate(prepared):
        draw_ground_truth(
            axes[row_index, 0],
            item,
            f"{item['scene']['scale']} scene",
        )
        for column_index, model in enumerate(models, 1):
            draw_model(
                axes[row_index, column_index],
                item,
                model,
                show_model_name=row_index == 0,
            )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.6, w_pad=0.8)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def render_standalone_rows(prepared: list[dict]):
    models = list(MODEL_FILES)
    for item in prepared:
        fig, axes = plt.subplots(1, 1 + len(models), figsize=(24, 4.2))
        fig.suptitle(
            f"Full-scene qualitative comparison — {item['scene']['scale']} objects\n"
            "All six target classes; green dashed=GT, blue=TP, red=FP",
            fontsize=15,
        )
        draw_ground_truth(axes[0], item, "Ground truth")
        for column, model in enumerate(models, 1):
            draw_model(axes[column], item, model, True)
        fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.8)
        scale_slug = item["scene"]["scale"].lower().replace("-", "_")
        stem = OUTPUT / f"qualitative_full_scene_{scale_slug}"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def render_gt_scale_examples(prepared: list[dict]):
    fig, axes = plt.subplots(1, len(prepared), figsize=(14, 4.2))
    fig.suptitle(
        "Experiment 1: Full-scene examples for object-scale evaluation\n"
        "All evaluable targets are shown; T=Tiny, S=Small, R=Regular",
        fontsize=15,
    )
    for axis, item in zip(axes, prepared):
        draw_ground_truth(
            axis,
            item,
            f"{item['scene']['scale']} scene",
        )
    fig.text(
        0.5,
        0.02,
        "Orange: Tiny (s<32)   Green: Small (32≤s<64)   Purple: Regular (s≥64)",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.88), w_pad=1.0)
    stem = OUTPUT / "experiment1_scale_examples_all_gt"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prepared, verification = prepare_scene_data()
    render_grid(
        prepared,
        [
            "YOLOv8x",
            "YOLOv8x-WorldV2",
            "Original SAM 3",
            "Adapted SAM 3 without replay",
            "Adapted SAM 3 with replay (Ours)",
        ],
        "qualitative_full_scene_all_models_all_scales",
        "Experiment 1: Full-scene qualitative comparison across object scales",
    )
    render_grid(
        prepared,
        [
            "YOLOv8x",
            "YOLOv8x-WorldV2",
            "Adapted SAM 3 with replay (Ours)",
        ],
        "qualitative_full_scene_detection_baselines",
        "Experiment 1: Detection baseline comparison",
    )
    render_grid(
        prepared,
        [
            "Original SAM 3",
            "Adapted SAM 3 without replay",
            "Adapted SAM 3 with replay (Ours)",
        ],
        "qualitative_full_scene_sam_ablation",
        "Experiment 1: SAM 3 adaptation and replay ablation",
    )
    render_standalone_rows(prepared)
    render_gt_scale_examples(prepared)
    (OUTPUT / "verification.json").write_text(
        json.dumps(
            {
                "protocol": (
                    "All evaluable GT targets for the six requested categories are shown. "
                    "Predictions use frozen validation thresholds, class-aware IoU >= 0.50, "
                    "official ignored-region filtering, and maxDets=500/image."
                ),
                "fp_display_note": (
                    "Only the without-replay panel caps drawn false-positive boxes at 30 "
                    "for legibility; its title reports the full FP count."
                ),
                "scenes": verification,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    readme = [
        "# Experiment 1 full-scene qualitative figures",
        "",
        "These figures answer the full-image visualization requirement: every evaluable "
        "GT target from car, person, bus, van, truck, and motorcycle is drawn in every "
        "panel. Green dashed boxes are GT, blue boxes are true-positive predictions, and "
        "red boxes are false-positive predictions.",
        "",
        "- Paper/main complete figure: `qualitative_full_scene_all_models_all_scales.png`",
        "- PPT detection-baseline slide: `qualitative_full_scene_detection_baselines.png`",
        "- PPT SAM/replay ablation slide: `qualitative_full_scene_sam_ablation.png`",
        "- One-row scale figures: `qualitative_full_scene_tiny.png`, "
        "`qualitative_full_scene_small.png`, and "
        "`qualitative_full_scene_regular_dominant.png`.",
        "- Experimental-design scale examples: `experiment1_scale_examples_all_gt.png`.",
        "",
        "Exact per-image metrics and protocol are in `verification.json`.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
