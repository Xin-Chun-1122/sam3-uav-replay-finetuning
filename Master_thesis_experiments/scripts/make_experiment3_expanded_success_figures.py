#!/usr/bin/env python3
"""Mine and render additional measured Experiment 3 comparative successes."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

import make_experiment3_success_figures as base


OUTPUT = base.ROOT / "figures/experiment3/expanded_success_cases"
EXCLUDED = {(example_id, layer) for _, example_id, layer in base.CASES}


def f1(metrics: dict[str, int]) -> float:
    denominator = 2 * metrics["tp"] + metrics["fp"] + metrics["fn"]
    return 2 * metrics["tp"] / denominator if denominator else 0.0


def evaluate(
    predictions: dict[str, dict[tuple[str, int], list[dict]]],
    example: dict,
    layer: int,
) -> dict:
    gt = example["layers"][layer - 1]["gt_boxes_xyxy"]
    filtered = {}
    metrics = {}
    for model in base.PREDICTION_FILES:
        rows = [
            row
            for row in predictions[model][(example["example_id"], layer)]
            if row["score"] >= base.THRESHOLDS[model]
        ]
        annotated, _, tp, fp, fn = base.match_predictions(rows, gt)
        filtered[model] = annotated
        metrics[model] = {"tp": tp, "fp": fp, "fn": fn}
    return {"gt": gt, "filtered": filtered, "metrics": metrics}


def render_case(case: dict, destination: Path) -> None:
    example = case["example"]
    layer = case["layer"]
    layer_row = example["layers"][layer - 1]
    image = Image.open(base.IMAGE_DIR / example["file_name"]).convert("RGB")
    crop = base.crop_bounds(
        case["gt"], case["filtered"], image.width, image.height
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"Measured comparative improvement ({example['category']}, Layer {layer}) — "
        f"prompt: “{layer_row['prompt']}”",
        fontsize=14,
    )
    base.draw_panel(
        axes[0], image, case["gt"], None, crop,
        f"Ground truth ({len(case['gt'])} target(s))",
    )
    for axis, model in zip(axes[1:], base.PREDICTION_FILES):
        metric = case["metrics"][model]
        base.draw_panel(
            axis,
            image,
            case["gt"],
            case["filtered"][model],
            crop,
            f"{model}\nTP={metric['tp']}, FP={metric['fp']}, FN={metric['fn']}",
        )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_category_sheet(category: str, cases: list[dict]) -> None:
    fig, axes = plt.subplots(len(cases), 3, figsize=(14, 4.0 * len(cases)))
    if len(cases) == 1:
        axes = [axes]
    fig.suptitle(
        f"Experiment 3: Additional measured {category} comparative successes\n"
        "Green: GT; Blue: matched prediction; Red: unmatched prediction",
        fontsize=16,
    )
    for row_index, case in enumerate(cases):
        example = case["example"]
        layer = case["layer"]
        layer_row = example["layers"][layer - 1]
        image = Image.open(base.IMAGE_DIR / example["file_name"]).convert("RGB")
        crop = base.crop_bounds(
            case["gt"], case["filtered"], image.width, image.height
        )
        base.draw_panel(
            axes[row_index][0],
            image,
            case["gt"],
            None,
            crop,
            f"Layer {layer} · “{layer_row['prompt']}”",
        )
        for column, model in enumerate(base.PREDICTION_FILES, 1):
            metric = case["metrics"][model]
            base.draw_panel(
                axes[row_index][column],
                image,
                case["gt"],
                case["filtered"][model],
                crop,
                f"{model}\nTP={metric['tp']}, FP={metric['fp']}, FN={metric['fn']}",
            )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.0, w_pad=1.0)
    stem = OUTPUT / f"qualitative_additional_wins_{category}"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(base.MANIFEST.read_text(encoding="utf-8"))
    predictions = {
        model: base.load_predictions(path)
        for model, path in base.PREDICTION_FILES.items()
    }
    cases = []
    for example in manifest:
        for layer in (1, 2, 3):
            if (example["example_id"], layer) in EXCLUDED:
                continue
            result = evaluate(predictions, example, layer)
            original = result["metrics"]["Original SAM 3"]
            adapted = result["metrics"]["Adapted SAM 3 with replay"]
            if adapted["tp"] <= original["tp"] or f1(adapted) <= f1(original):
                continue
            cases.append({
                "example": example,
                "layer": layer,
                **result,
                "clean": adapted["fp"] == 0 and adapted["fn"] == 0,
                "f1_gain": f1(adapted) - f1(original),
            })

    cases.sort(
        key=lambda case: (
            case["example"]["category"],
            not case["clean"],
            -case["f1_gain"],
            case["layer"],
            case["example"]["example_id"],
        )
    )
    rows = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        example = case["example"]
        layer = case["layer"]
        original = case["metrics"]["Original SAM 3"]
        adapted = case["metrics"]["Adapted SAM 3 with replay"]
        stem = (
            f"qualitative_additional_{example['category']}_"
            f"{example['example_id']}_layer{layer}"
        )
        render_case(case, OUTPUT / stem)
        grouped[example["category"]].append(case)
        rows.append(
            f"| {example['category']} | {layer} | `{example['example_id']}` | "
            f"{example['layers'][layer - 1]['prompt']} | "
            f"{original['tp']}/{original['fp']}/{original['fn']} | "
            f"{adapted['tp']}/{adapted['fp']}/{adapted['fn']} | "
            f"{f1(original):.3f} | {f1(adapted):.3f} | "
            f"{'yes' if case['clean'] else 'no'} | `{stem}.png` |"
        )

    for category, category_cases in grouped.items():
        render_category_sheet(category, category_cases[:2])

    readme = [
        "# Experiment 3 expanded measured comparative successes",
        "",
        "These cases were mined from the already completed RefDrone test predictions. "
        "No boxes or scores were edited. Every retained case satisfies both: "
        "(1) Adapted SAM 3 with replay has more true positives than Original SAM 3, and "
        "(2) Adapted has higher instance-level F1, using frozen model-specific thresholds "
        "and IoU >= 0.50.",
        "",
        "The five cases already published in `paper_success_cases` are excluded, so every "
        "figure here is an additional test image/layer. `Clean=yes` means Adapted FP=0 and FN=0.",
        "",
        "TP/FP/FN are written as `TP/FP/FN`.",
        "",
        "| Category | Layer | Example | Prompt | Original | Adapted | Original F1 | Adapted F1 | Clean | Figure |",
        "|---|---:|---|---|---:|---:|---:|---:|---|---|",
        *rows,
        "",
        "For slides, use one of the `qualitative_additional_wins_<category>.png` sheets. "
        "Each sheet contains the two highest-ranked additional wins for that category.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    (OUTPUT / "selection_manifest.json").write_text(
        json.dumps(
            [
                {
                    "category": case["example"]["category"],
                    "example_id": case["example"]["example_id"],
                    "file_name": case["example"]["file_name"],
                    "layer": case["layer"],
                    "prompt": case["example"]["layers"][case["layer"] - 1]["prompt"],
                    "metrics": case["metrics"],
                    "clean": case["clean"],
                    "f1_gain": case["f1_gain"],
                }
                for case in cases
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"output={OUTPUT}")
    print(f"additional_cases={len(cases)}")
    print(f"category_sheets={len(grouped)}")


if __name__ == "__main__":
    main()
