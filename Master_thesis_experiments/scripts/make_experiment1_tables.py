#!/usr/bin/env python3
"""Create final Experiment 1 paper tables from frozen test result JSON files."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "experiment1" / "test"
TABLE_DIR = ROOT / "tables" / "experiment1"
MODELS = (
    ("YOLO", "yolo_test1610.json"),
    ("YOLO-World", "yolo_world_test1610.json"),
    ("Original SAM 3", "sam3_original_test1610.json"),
    ("Adapted SAM 3 without replay", "sam3_adapted_no_replay_test1610.json"),
    ("Adapted SAM 3 with replay", "sam3_adapted_with_replay_test1610.json"),
)
FIELDS = (
    "Tiny F1", "Tiny Recall", "Small F1", "Small Recall",
    "Regular F1", "Regular Recall", "AP50", "FP/image",
)


def load_rows() -> list[dict]:
    rows = []
    for model, filename in MODELS:
        result = json.loads((RESULT_DIR / filename).read_text(encoding="utf-8"))
        if result["evaluated_image_count"] != 1610:
            raise ValueError(f"{model}: expected 1610 images")
        if result["threshold_source"] != "frozen_external":
            raise ValueError(f"{model}: test threshold was not frozen externally")
        scales = result["scale_metrics"]
        rows.append({
            "Model": model,
            "Tiny F1": scales["tiny"]["f1"],
            "Tiny Recall": scales["tiny"]["recall"],
            "Small F1": scales["small"]["f1"],
            "Small Recall": scales["small"]["recall"],
            "Regular F1": scales["regular"]["f1"],
            "Regular Recall": scales["regular"]["recall"],
            # Standard multi-class detection value: mean AP50 across the six classes.
            "AP50": result["macro_ap50"],
            "Micro AP50": result["micro_ap50"],
            "FP/image": result["fp_per_image"],
            "Threshold": result["operating_threshold"],
        })
    return rows


def bold(value: float, field: str, best: dict[str, float]) -> str:
    rendered = f"{value:.4f}"
    return f"**{rendered}**" if abs(value - best[field]) < 1e-12 else rendered


def main() -> None:
    rows = load_rows()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    best = {
        field: (min(row[field] for row in rows) if field == "FP/image" else max(row[field] for row in rows))
        for field in FIELDS
    }

    with (TABLE_DIR / "experiment1_main_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Model", *FIELDS, "Micro AP50", "Threshold"))
        writer.writeheader()
        writer.writerows(rows)

    with (TABLE_DIR / "dataset_scale_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("scale", "definition", "images_containing_scale", "evaluated_gt_instances"))
        writer.writeheader()
        writer.writerows([
            {"scale": "tiny", "definition": "s < 32", "images_containing_scale": 1496, "evaluated_gt_instances": 48859},
            {"scale": "small", "definition": "32 <= s < 64", "images_containing_scale": 1523, "evaluated_gt_instances": 16361},
            {"scale": "regular", "definition": "s >= 64", "images_containing_scale": 1164, "evaluated_gt_instances": 7092},
        ])

    header = "| Model | " + " | ".join(FIELDS) + " |\n"
    separator = "|---|" + "---:|" * len(FIELDS) + "\n"
    body = "".join(
        "| " + row["Model"] + " | " + " | ".join(bold(row[field], field, best) for field in FIELDS) + " |\n"
        for row in rows
    )
    markdown = f"""# Experiment 1 — Object Scale on VisDrone-DET test-dev

## Dataset scale statistics

| Scale | Definition | Images containing scale | Evaluated GT instances |
|---|---|---:|---:|
| Tiny | $s < 32$ | 1,496 | 48,859 |
| Small | $32 \\le s < 64$ | 1,523 | 16,361 |
| Regular | $s \\ge 64$ | 1,164 | 7,092 |

Image sets overlap because one image can contain objects from multiple scale bins.

## Final test results

{header}{separator}{body}
AP50 is the 101-point macro AP at IoU 0.50 over car, person, bus, van, truck, and motorcycle. FP/image is total false positives divided by all 1,610 test images. F1, Recall, and FP/image use per-model confidence thresholds frozen on the complete 548-image validation split. All methods use class-aware matching at IoU 0.50, official VisDrone ignored-region filtering, and maxDets=500/image.

Supplementary all-instance micro AP50: YOLO {rows[0]['Micro AP50']:.4f}, YOLO-World {rows[1]['Micro AP50']:.4f}, Original SAM 3 {rows[2]['Micro AP50']:.4f}, without replay {rows[3]['Micro AP50']:.4f}, with replay {rows[4]['Micro AP50']:.4f}.
"""
    (TABLE_DIR / "experiment1_main_results.md").write_text(markdown, encoding="utf-8")

    latex_names = {
        "YOLO": "YOLO",
        "YOLO-World": "YOLO-World",
        "Original SAM 3": "Original SAM 3",
        "Adapted SAM 3 without replay": "Adapted SAM 3 w/o replay",
        "Adapted SAM 3 with replay": "Adapted SAM 3 w/ replay",
    }
    latex_rows = []
    for row in rows:
        values = []
        for field in FIELDS:
            value = f"{row[field]:.4f}"
            if abs(row[field] - best[field]) < 1e-12:
                value = f"\\textbf{{{value}}}"
            values.append(value)
        latex_rows.append(latex_names[row["Model"]] + " & " + " & ".join(values) + " \\\\")
    latex = """\\begin{table*}[t]
\\centering
\\caption{Object-scale detection results on the VisDrone-DET test-dev set.}
\\label{tab:experiment1_scale}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lcccccccc}
\\toprule
Model & Tiny F1 & Tiny R & Small F1 & Small R & Regular F1 & Regular R & AP50 & FP/image \\\\
\\midrule
""" + "\n".join(latex_rows) + """
\\bottomrule
\\end{tabular}}
\\end{table*}
"""
    (TABLE_DIR / "experiment1_main_results.tex").write_text(latex, encoding="utf-8")

    original, adapted = rows[2], rows[4]
    analysis = {
        "test_images": 1610,
        "with_replay_minus_original": {field: adapted[field] - original[field] for field in FIELDS},
        "main_ap50_definition": "macro 101-point AP at IoU 0.50 over six target categories",
        "supplementary_micro_ap50": {row["Model"]: row["Micro AP50"] for row in rows},
    }
    (TABLE_DIR / "experiment1_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    artifacts = []
    for path in sorted((ROOT / "predictions" / "experiment1" / "test").glob("*.jsonl")) + sorted(RESULT_DIR.glob("*.json")):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        artifacts.append({
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    (TABLE_DIR / "artifact_manifest.json").write_text(
        json.dumps({"hash_algorithm": "sha256", "artifacts": artifacts}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(markdown)


if __name__ == "__main__":
    main()
