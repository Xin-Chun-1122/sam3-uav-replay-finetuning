#!/usr/bin/env python3
"""Create thesis-ready Experiment 3 tables from measured result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODEL_LABELS = {
    "sam3_original": "Original SAM 3",
    "sam3_adapted_with_replay": "Adapted SAM 3 with replay",
}


def f(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    results = [
        json.loads(args.original.read_text(encoding="utf-8")),
        json.loads(args.adapted.read_text(encoding="utf-8")),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    standard_rows, synonym_rows, specificity_rows = [], [], []
    for result in results:
        model = MODEL_LABELS[result["model"]]
        for row in result["standard"]["per_category"]:
            standard_rows.append({
                "Model": model,
                "Category": row["category"],
                "Prompt": row["prompt"],
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1": row["f1"],
                "AP50": row["ap50"],
            })
        macro = result["standard"]["macro"]
        standard_rows.append({
            "Model": model, "Category": "Macro average", "Prompt": "—",
            "Precision": macro["precision"], "Recall": macro["recall"],
            "F1": macro["f1"], "AP50": macro["ap50"],
        })
        for row in result["synonym"]["per_category"]:
            synonym_rows.append({
                "Model": model,
                "Category": row["category"],
                "Synonym prompt": row["prompt"],
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1": row["f1"],
                "AP50": row["ap50"],
                "Retention": row["retention"],
            })
        macro = result["synonym"]["macro"]
        synonym_rows.append({
            "Model": model, "Category": "Macro average", "Synonym prompt": "—",
            "Precision": macro["precision"], "Recall": macro["recall"],
            "F1": macro["f1"], "AP50": macro["ap50"],
            "Retention": macro["retention"],
        })
        for row in result["specificity"]["overall_by_layer"]:
            specificity_rows.append({
                "Model": model,
                "Layer": row["layer"],
                "Prompt scope": {
                    1: "Category",
                    2: "Category + attribute",
                    3: "Attribute + spatial/relational description",
                }[row["layer"]],
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1": row["f1"],
                "AP50": row["ap50"],
                "Mean GT count": row["mean_gt_count"],
                "Mean predicted count": row["mean_predicted_count"],
            })

    write_csv(
        args.output_dir / "table1_standard_prompts.csv",
        ["Model", "Category", "Prompt", "Precision", "Recall", "F1", "AP50"],
        standard_rows,
    )
    write_csv(
        args.output_dir / "table2_synonym_prompts.csv",
        ["Model", "Category", "Synonym prompt", "Precision", "Recall", "F1", "AP50", "Retention"],
        synonym_rows,
    )
    write_csv(
        args.output_dir / "table3_specificity_layers.csv",
        [
            "Model", "Layer", "Prompt scope", "Precision", "Recall", "F1", "AP50",
            "Mean GT count", "Mean predicted count",
        ],
        specificity_rows,
    )

    lines = [
        "# Experiment 3 — Prompt Specificity Evaluation",
        "",
        "All values below are measured on the frozen RefDrone test protocol. "
        "IoU threshold is 0.50; operating thresholds were frozen on the Experiment 1 validation split.",
        "",
        "## Table 1. Standard category prompts",
        "",
        "| Model | Category | Prompt | Precision | Recall | F1 | AP50 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in standard_rows:
        lines.append(
            f"| {row['Model']} | {row['Category']} | {row['Prompt']} | "
            f"{f(row['Precision'])} | {f(row['Recall'])} | {f(row['F1'])} | {f(row['AP50'])} |"
        )
    lines += [
        "",
        "## Table 2. Synonym prompts",
        "",
        "| Model | Category | Synonym | Precision | Recall | F1 | AP50 | Retention |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in synonym_rows:
        lines.append(
            f"| {row['Model']} | {row['Category']} | {row['Synonym prompt']} | "
            f"{f(row['Precision'])} | {f(row['Recall'])} | {f(row['F1'])} | "
            f"{f(row['AP50'])} | {f(row['Retention'])} |"
        )
    lines += [
        "",
        "Bus is excluded from synonym retention because no bus synonym was specified in the experiment.",
        "",
        "## Table 3. Fixed three-layer specificity subset (50 images)",
        "",
        "| Model | Layer | Prompt scope | Precision | Recall | F1 | AP50 | Mean GT | Mean predicted |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in specificity_rows:
        lines.append(
            f"| {row['Model']} | {row['Layer']} | {row['Prompt scope']} | "
            f"{f(row['Precision'])} | {f(row['Recall'])} | {f(row['F1'])} | "
            f"{f(row['AP50'])} | {f(row['Mean GT count'])} | {f(row['Mean predicted count'])} |"
        )
    (args.output_dir / "EXPERIMENT3_TABLES.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
