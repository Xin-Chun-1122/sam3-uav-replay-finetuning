#!/usr/bin/env python3
"""Create equal, verified ROI crops for the Experiment 1 PPT table."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

from PIL import Image


ROOT = Path("/home/alien/sam3/Master_thesis_experiments")
DATASET = Path("/home/alien/Downloads/VisDrone2019-DET-test-dev")
SOURCE = ROOT / "figures/experiment1/ppt_clean_panels_detection_baseline"
OUTPUT = SOURCE / "ppt_table_crops"
EVALUATOR_PATH = ROOT / "scripts/evaluate_visdrone_predictions.py"
OUTPUT_SIZE = (960, 720)
PROMPT = "car"

SCENES = {
    "tiny": "9999938_00000_d_0000252.jpg",
    "small": "9999996_00000_d_0000018.jpg",
    "regular": "9999963_00000_d_0000009.jpg",
}

PANELS = {
    "ground_truth": "01_ground_truth_car.png",
    "yolov8x": "02_yolov8x_car.png",
    "yolo_worldv2": "03_yolo_world_v2_car.png",
    "original_sam3": "04_original_sam3_car.png",
    "ours": "06_ours_car.png",
}


def load_evaluator():
    specification = importlib.util.spec_from_file_location("vis_eval", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def fit_roi(
    union: tuple[float, float, float, float],
    image_size: tuple[int, int],
    padding: int,
    aspect: float = 4 / 3,
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x1, y1, x2, y2 = union
    x1, y1 = max(0.0, x1 - padding), max(0.0, y1 - padding)
    x2, y2 = min(float(image_width), x2 + padding), min(float(image_height), y2 + padding)
    width, height = x2 - x1, y2 - y1
    if width / height < aspect:
        width = height * aspect
    else:
        height = width / aspect
    width, height = min(width, image_width), min(height, image_height)
    if width / height < aspect:
        width = min(float(image_width), height * aspect)
    elif width / height > aspect:
        height = min(float(image_height), width / aspect)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    left = max(0.0, min(image_width - width, center_x - width / 2))
    top = max(0.0, min(image_height - height, center_y - height / 2))
    return (
        int(round(left)),
        int(round(top)),
        int(round(left + width)),
        int(round(top + height)),
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ev = load_evaluator()
    gt, _ = ev.load_gt(DATASET, set(SCENES.values()))
    audit = {
        "note": (
            "Scale is computed from the original VisDrone annotation before "
            "cropping or resizing: s=sqrt(width*height)."
        ),
        "prompt": PROMPT,
        "output_size": list(OUTPUT_SIZE),
        "scenes": {},
    }

    for scale, image_name in SCENES.items():
        targets = [
            target
            for target in gt[(image_name, PROMPT)]
            if not target["ignored"]
        ]
        if not targets or any(target["scale"] != scale for target in targets):
            raise RuntimeError(f"{scale}: GT scale verification failed")
        sizes = []
        for target in targets:
            x1, y1, x2, y2 = target["bbox"]
            sizes.append(math.sqrt((x2 - x1) * (y2 - y1)))

        with Image.open(DATASET / "images" / image_name) as source_image:
            image_size = source_image.size
        union = (
            min(target["bbox"][0] for target in targets),
            min(target["bbox"][1] for target in targets),
            max(target["bbox"][2] for target in targets),
            max(target["bbox"][3] for target in targets),
        )
        padding = 55 if scale != "regular" else 0
        roi = fit_roi(union, image_size, padding)
        if not all(
            roi[0] <= target["bbox"][0]
            and roi[1] <= target["bbox"][1]
            and roi[2] >= target["bbox"][2]
            and roi[3] >= target["bbox"][3]
            for target in targets
        ):
            raise RuntimeError(f"{scale}: ROI clips a valid GT: {roi}")

        for panel, suffix in PANELS.items():
            source_path = SOURCE / f"{scale}_{suffix}"
            with Image.open(source_path) as image:
                cropped = image.convert("RGB").crop(roi)
                cropped = cropped.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
                cropped.save(
                    OUTPUT / f"{scale}_{panel}.png",
                    dpi=(300, 300),
                    compress_level=3,
                )

        audit["scenes"][scale] = {
            "image_name": image_name,
            "original_image_size": list(image_size),
            "roi_xyxy": list(roi),
            "valid_car_gt": len(targets),
            "s_min": min(sizes),
            "s_max": max(sizes),
            "all_scale_labels": sorted({target["scale"] for target in targets}),
            "all_gt_inside_roi": True,
        }

    (OUTPUT / "scale_and_crop_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUTPUT / "README.md").write_text(
        """# PPT table crops

Insert the 15 PNG files into the blank 3x5 image table.

Columns:
1. ground_truth
2. yolov8x
3. yolo_worldv2
4. original_sam3
5. ours

Rows:
1. tiny
2. small
3. regular

Every file is 960x720 (4:3). A row always uses the exact same ROI for every
model. All valid `car` GT boxes are inside the ROI. Scale is computed from the
original annotation before cropping/resizing, never from the displayed panel.
""",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
