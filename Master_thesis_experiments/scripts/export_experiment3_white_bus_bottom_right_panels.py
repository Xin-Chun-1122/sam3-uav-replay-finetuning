#!/usr/bin/env python3
"""Export measured Layer-3 white-bus panels for the Experiment 3 PPT."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import export_experiment3_ppt_panels as panels
import make_experiment3_representative_figures as representative
import make_experiment3_success_figures as base


ROOT = Path("/home/alien/sam3/Master_thesis_experiments")
MANIFEST = ROOT / "annotations/experiment3/white_bus_bottom_right_single_manifest.json"
OUTPUT = ROOT / "figures/experiment3/ppt_table_panels/white_bus_bottom_right_layer3"
PREDICTIONS = {
    "Original SAM 3": ROOT / "predictions/experiment3/sam3_original_white_bus_far_right.jsonl",
    "Adapted SAM 3 with replay": ROOT / "predictions/experiment3/sam3_adapted_white_bus_far_right.jsonl",
}


def read_prediction(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "prediction":
                return row["predictions"]
    raise RuntimeError(f"No prediction row in {path}")


def main() -> None:
    example = json.loads(MANIFEST.read_text(encoding="utf-8"))[0]
    layer = example["layers"][0]
    gt = layer["gt_boxes_xyxy"]
    raw = {model: read_prediction(path) for model, path in PREDICTIONS.items()}
    filtered, metrics = representative.prepare(raw, gt)
    expected = {
        "Original SAM 3": {"tp": 0, "fp": 0, "fn": 1},
        "Adapted SAM 3 with replay": {"tp": 1, "fp": 0, "fn": 0},
    }
    if metrics != expected:
        raise RuntimeError(f"Measured metrics changed: {metrics}")

    image = Image.open(base.IMAGE_DIR / example["file_name"]).convert("RGB")
    full_scene = (0.0, 0.0, float(image.width), float(image.height))
    panels.export_panel(
        image, full_scene, gt, None, OUTPUT / "layer3_ground_truth.png", letterbox=True
    )
    panels.export_panel(
        image, full_scene, gt, filtered["Original SAM 3"],
        OUTPUT / "layer3_original_sam3.png", letterbox=True,
    )
    panels.export_panel(
        image, full_scene, gt, filtered["Adapted SAM 3 with replay"],
        OUTPUT / "layer3_ours.png", letterbox=True,
    )
    verification = {
        "prompt": layer["prompt"],
        "file_name": example["file_name"],
        "gt_boxes_xyxy": gt,
        "metrics": metrics,
        "panel_size": list(panels.PANEL_SIZE),
        "note": "Fresh inference with the exact white-bus Layer-3 prompt; boxes were not edited.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
