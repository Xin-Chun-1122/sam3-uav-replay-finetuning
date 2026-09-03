#!/usr/bin/env python3
"""Export a measured Layer-3 white-bus case with a visible green-bus distractor."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import export_experiment3_ppt_panels as panels
import make_experiment3_representative_figures as representative
import make_experiment3_success_figures as base


ROOT = Path("/home/alien/sam3/Master_thesis_experiments")
MANIFEST = ROOT / "annotations/experiment3/white_bus_green_distractor_0145_probe_manifest.json"
OUTPUT = ROOT / "figures/experiment3/ppt_table_panels/white_bus_with_green_distractor_layer3_0145"
PREDICTIONS = {
    "Original SAM 3": ROOT / "predictions/experiment3/sam3_original_white_bus_green_distractor_0145.jsonl",
    "Adapted SAM 3 with replay": ROOT / "predictions/experiment3/sam3_adapted_white_bus_green_distractor_0145.jsonl",
}
EXAMPLE_ID = "white-bus-green-distractor-0145-3"


def read_prediction(path: Path, example_id: str) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "prediction" and row.get("example_id") == example_id:
                return row["predictions"]
    raise RuntimeError(f"No prediction for {example_id} in {path}")


def main() -> None:
    examples = json.loads(MANIFEST.read_text(encoding="utf-8"))
    example = next(row for row in examples if row["example_id"] == EXAMPLE_ID)
    layer = example["layers"][0]
    gt = layer["gt_boxes_xyxy"]
    raw = {
        model: read_prediction(path, EXAMPLE_ID)
        for model, path in PREDICTIONS.items()
    }
    filtered, metrics = representative.prepare(raw, gt)
    expected = {
        "Original SAM 3": {"tp": 0, "fp": 0, "fn": 1},
        "Adapted SAM 3 with replay": {"tp": 1, "fp": 0, "fn": 0},
    }
    if metrics != expected:
        raise RuntimeError(f"Measured metrics changed: {metrics}")

    image = Image.open(base.IMAGE_DIR / example["file_name"]).convert("RGB")
    # A moderate, unchanged crop: it keeps the green-bus distractor above the
    # requested white bus, plus enough road context for the spatial phrase.
    full_scene = (700.0, 0.0, 1400.0, 700.0)
    panels.export_panel(
        image, full_scene, gt, None, OUTPUT / "layer3_ground_truth.png", letterbox=True
    )
    panels.export_panel(
        image,
        full_scene,
        gt,
        filtered["Original SAM 3"],
        OUTPUT / "layer3_original_sam3.png",
        letterbox=True,
        draw_gt=False,
    )
    panels.export_panel(
        image,
        full_scene,
        gt,
        filtered["Adapted SAM 3 with replay"],
        OUTPUT / "layer3_ours.png",
        letterbox=True,
        draw_gt=False,
    )

    verification = {
        "example_id": EXAMPLE_ID,
        "file_name": example["file_name"],
        "prompt": layer["prompt"],
        "gt_boxes_xyxy": gt,
        "metrics": metrics,
        "raw_top_scores": {
            model: rows[0]["score"] if rows else None for model, rows in raw.items()
        },
        "display_crop_xyxy": list(full_scene),
        "scene_check": (
            "The unchanged crop visibly contains a green-bus distractor above the single "
            "large white target bus. Ground truth and Ours mark only the white bus."
        ),
        "note": "Saved measured predictions only; no box or prediction was edited.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
