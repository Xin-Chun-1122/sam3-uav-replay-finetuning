#!/usr/bin/env python3
"""Create the two most representative measured figures for Experiment 3."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

import make_experiment3_success_figures as base


OUTPUT = base.ROOT / "figures/experiment3/paper_representative_cases"
FULL_PREDICTIONS = {
    "Original SAM 3": base.ROOT / "predictions/experiment3/sam3_original.jsonl",
    "Adapted SAM 3 with replay": (
        base.ROOT / "predictions/experiment3/sam3_adapted_with_replay.jsonl"
    ),
}


def load_synonym_predictions(path: Path) -> dict[tuple[str, str], list[dict]]:
    output = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "prediction" and row.get("evaluation") == "synonym":
                output[(row["image_name"], row["category"])] = row["predictions"]
    return output


def prepare(
    raw: dict[str, list[dict]],
    gt: list[list[float]],
) -> tuple[dict[str, list[dict]], dict[str, dict[str, int]]]:
    filtered = {}
    metrics = {}
    for model in base.PREDICTION_FILES:
        rows = [
            row for row in raw[model]
            if row["score"] >= base.THRESHOLDS[model]
        ]
        annotated, _, tp, fp, fn = base.match_predictions(rows, gt)
        filtered[model] = annotated
        metrics[model] = {"tp": tp, "fp": fp, "fn": fn}
    return filtered, metrics


def render_synonym_lorry() -> dict:
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
    predictions = {
        model: load_synonym_predictions(path)
        for model, path in FULL_PREDICTIONS.items()
    }
    raw = {
        model: predictions[model][(unit["file_name"], unit["category"])]
        for model in predictions
    }
    filtered, metrics = prepare(raw, unit["gt_boxes_xyxy"])
    adapted = metrics["Adapted SAM 3 with replay"]
    original = metrics["Original SAM 3"]
    if not (
        adapted == {"tp": 2, "fp": 0, "fn": 0}
        and original["tp"] < adapted["tp"]
    ):
        raise RuntimeError("Locked synonym representative is no longer a strict win")

    image = Image.open(base.IMAGE_DIR / unit["file_name"]).convert("RGB")
    crop = base.crop_bounds(
        unit["gt_boxes_xyxy"], filtered, image.width, image.height
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Synonym prompt success — “lorry” (truck)\n"
        "Frozen thresholds; correct localization requires IoU ≥ 0.50",
        fontsize=15,
    )
    base.draw_panel(
        axes[0], image, unit["gt_boxes_xyxy"], None, crop,
        "Ground truth (2 targets)",
    )
    for axis, model in zip(axes[1:], base.PREDICTION_FILES):
        metric = metrics[model]
        base.draw_panel(
            axis, image, unit["gt_boxes_xyxy"], filtered[model], crop,
            f"{model}\nTP={metric['tp']}, FP={metric['fp']}, FN={metric['fn']}",
        )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    stem = OUTPUT / "qualitative_representative_synonym_lorry"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {"unit": unit, "metrics": metrics}


def render_bus_progression() -> dict:
    manifest = json.loads(base.MANIFEST.read_text(encoding="utf-8"))
    example = next(
        row for row in manifest
        if row["example_id"] == "refdrone-specificity-038"
    )
    predictions = {
        model: base.load_predictions(path)
        for model, path in base.PREDICTION_FILES.items()
    }
    image = Image.open(base.IMAGE_DIR / example["file_name"]).convert("RGB")
    # Use one unchanged full-scene view at every specificity level.  This keeps
    # the differently colored buses visible when evaluating "green bus" and
    # prevents the attribute task from being trivialized by a target-only crop.
    full_scene = (0.0, 0.0, float(image.width), float(image.height))
    prepared = []
    for layer in (1, 2, 3):
        layer_row = example["layers"][layer - 1]
        raw = {
            model: predictions[model][(example["example_id"], layer)]
            for model in predictions
        }
        filtered, metrics = prepare(raw, layer_row["gt_boxes_xyxy"])
        prepared.append((layer, layer_row, filtered, metrics, full_scene))

    expected = [
        ({"tp": 7, "fp": 1, "fn": 0}, {"tp": 7, "fp": 0, "fn": 0}),
        ({"tp": 1, "fp": 0, "fn": 0}, {"tp": 1, "fp": 0, "fn": 0}),
        ({"tp": 0, "fp": 0, "fn": 1}, {"tp": 1, "fp": 0, "fn": 0}),
    ]
    for (_, _, _, metrics, _), (original, adapted) in zip(prepared, expected):
        if (
            metrics["Original SAM 3"] != original
            or metrics["Adapted SAM 3 with replay"] != adapted
        ):
            raise RuntimeError("Locked bus progression metrics changed")

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    fig.suptitle(
        "Progressive prompt specificity on the same UAV image\n"
        "Layer 1: category → Layer 2: attribute → Layer 3: spatial description",
        fontsize=16,
    )
    for row_index, (layer, layer_row, filtered, metrics, crop) in enumerate(prepared):
        gt = layer_row["gt_boxes_xyxy"]
        base.draw_panel(
            axes[row_index, 0], image, gt, None, crop,
            f"Layer {layer} · “{layer_row['prompt']}”\nGT targets={len(gt)}",
        )
        for column, model in enumerate(base.PREDICTION_FILES, 1):
            metric = metrics[model]
            base.draw_panel(
                axes[row_index, column], image, gt, filtered[model], crop,
                f"{model}\nTP={metric['tp']}, FP={metric['fp']}, FN={metric['fn']}",
            )
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.0, w_pad=1.0)
    stem = OUTPUT / "qualitative_representative_bus_layer1_to_layer3"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "example_id": example["example_id"],
        "file_name": example["file_name"],
        "layers": [
            {
                "layer": layer,
                "prompt": layer_row["prompt"],
                "gt_count": len(layer_row["gt_boxes_xyxy"]),
                "metrics": metrics,
            }
            for layer, layer_row, _, metrics, _ in prepared
        ],
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    synonym = render_synonym_lorry()
    progression = render_bus_progression()
    audit = {
        "selection_policy": (
            "Representative measured cases only; frozen model thresholds and IoU >= 0.50. "
            "Predictions and boxes are not edited."
        ),
        "synonym_lorry": {
            "file_name": synonym["unit"]["file_name"],
            "category": synonym["unit"]["category"],
            "prompt": synonym["unit"]["synonym_prompt"],
            "metrics": synonym["metrics"],
        },
        "bus_progression": progression,
    }
    (OUTPUT / "verification.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = [
        "# Experiment 3 recommended representative figures",
        "",
        "Use these two figures after the Experiment 3 quantitative tables:",
        "",
        "1. `qualitative_representative_synonym_lorry.png` — synonym robustness.",
        "2. `qualitative_representative_bus_layer1_to_layer3.png` — fixed three-level "
        "prompt specificity on the same image.",
        "",
        "Both figures use saved measured predictions, frozen validation thresholds, and "
        "IoU >= 0.50. No prediction or ground-truth box was edited. Exact metrics are "
        "recorded in `verification.json`.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
