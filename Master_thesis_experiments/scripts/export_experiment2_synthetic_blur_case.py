#!/usr/bin/env python3
"""Export the verified full-clip synthetic motion-blur comparison case."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from export_experiment2_ppt_assets import (
    MODELS,
    fixed_case_crop,
    image_path,
    make_contact_sheet,
    read_all_gt,
    render_cell,
)
from infer_sam3_mot_events import motion_blur


ROOT = Path("/home/alien/sam3/Master_thesis_experiments")
EVENTS = ROOT / "annotations/experiment2_camera_blur_fullclip/events.json"
PREDICTIONS = ROOT / "predictions/experiment2_camera_blur_fullclip"
RESULTS = ROOT / "results/experiment2_camera_blur_fullclip"
OUTPUT = ROOT / "figures/experiment2/ppt_local_tracking_cases/camera_motion_or_blur"
EVENT_ID = "camera_motion_blur_fullclip_k31-001"
FRAMES = {"before": 428, "during": 438, "after": 442}


def load_predictions(model: str) -> dict[int, list[dict]]:
    path = PREDICTIONS / model / f"{EVENT_ID}.jsonl"
    rows: dict[int, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        rows[int(item["frame_index"])] = item["predictions"]
    return rows


def selected_id(model: str) -> int:
    data = json.loads((RESULTS / f"{model}.json").read_text(encoding="utf-8"))
    row = next(item for item in data["event_results"] if item["event_id"] == EVENT_ID)
    return int(row["selected_sam3_obj_id"])


def main() -> None:
    event = next(
        item for item in json.loads(EVENTS.read_text(encoding="utf-8"))
        if item["event_id"] == EVENT_ID
    )
    gt = read_all_gt(event["sequence_id"])
    crop = fixed_case_crop(event["sequence_id"], gt, int(event["target_id"]), FRAMES)
    models = list(MODELS)[1:]
    predictions = {model: load_predictions(model) for model in models}
    selected = {model: selected_id(model) for model in models}
    cells: dict[tuple[str, str], object] = {}
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for phase, frame in FRAMES.items():
        source = cv2.imread(str(image_path(event["sequence_id"], frame)))
        source = motion_blur(
            source,
            int(event["synthetic_motion_blur_kernel"]),
            float(event["synthetic_motion_blur_angle_degrees"]),
        )
        gt_cell = render_cell(
            source, gt.get(frame, []), event["category"], int(event["target_id"]), crop
        )
        cells[("ground_truth", phase)] = gt_cell
        cv2.imwrite(str(OUTPUT / f"{phase}_ground_truth.png"), gt_cell)
        for model in models:
            cell = render_cell(
                source,
                gt.get(frame, []),
                event["category"],
                int(event["target_id"]),
                crop,
                predictions[model].get(frame, []),
                selected[model],
            )
            cells[(model, phase)] = cell
            cv2.imwrite(str(OUTPUT / f"{phase}_{model}.png"), cell)

    case = {
        "frames": FRAMES,
        "prompt": event["category"],
        "subtitle": "Ours tracks through controlled severe blur while Native-ID fails",
        "gt_legend": (
            "Synthetic full-clip motion blur: kernel=31 px, angle=12 deg; "
            "yellow dashed: GT; blue: correct native ID; red: error/miss"
        ),
        "phase_labels": {
            "before": "Before | Frame 428 | synthetic blur k=31",
            "during": "During | Frame 438 | synthetic blur k=31",
            "after": "After | Frame 442 | synthetic blur k=31",
        },
    }
    make_contact_sheet(cells, case, event, OUTPUT / "comparison_labeled.png")

    # Borderless 4-row x 3-column asset for the user's existing PPT table.
    clean = cv2.vconcat([
        cv2.hconcat([cells[(model, phase)] for phase in ("before", "during", "after")])
        for model in ("ground_truth", "sam3_original", "sam3_native_id_tracker", "sam3_adapted_with_replay")
    ])
    cv2.imwrite(str(OUTPUT / "comparison_clean_4x3.png"), clean)


if __name__ == "__main__":
    main()
