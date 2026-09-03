#!/usr/bin/env python3
"""Export event 037 with a genuinely smaller final target frame."""

from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Master_thesis_experiments.scripts import export_experiment2_ppt_assets as assets
from Master_thesis_experiments.scripts import make_experiment2_figures as paper_figures


EVENT_ID = "scale_decrease_and_recovery-037"
# Same event as the previously approved figure.  Frame 220 is the measured
# minimum-scale frame; frame 227 was the larger recovered target.
FRAMES = {"before": 202, "during": 210, "after": 220}
PROMPT = "car"
TARGET_ID = 13
CROP_SIZE = (480, 300)
OUTPUT = (
    assets.ROOT
    / "figures/experiment2/ppt_local_tracking_cases"
    / "scale_decrease_and_recovery_037_smaller_final"
)


def centered_crop(box, image_shape):
    height, width = image_shape[:2]
    crop_w, crop_h = CROP_SIZE
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    x0 = int(np.clip(cx - crop_w / 2, 0, width - crop_w))
    y0 = int(np.clip(cy - crop_h / 2, 0, height - crop_h))
    return (x0, y0, x0 + crop_w, y0 + crop_h)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    event = next(
        row for row in assets.load_json(assets.EVENTS_PATH)
        if row["event_id"] == EVENT_ID
    )
    sequence = event["sequence_id"]
    gt = assets.read_all_gt(sequence)
    _, selected = assets.result_index()
    predictions = {
        model: assets.read_predictions(model, EVENT_ID)
        for model in list(assets.MODELS)[1:]
    }

    cells, scales = {}, {}
    for phase, frame in FRAMES.items():
        image = cv2.imread(str(assets.image_path(sequence, frame)))
        target = next(row for row in gt[frame] if row["target_id"] == TARGET_ID)
        box = target["bbox_xyxy"]
        scales[phase] = ((box[2] - box[0]) * (box[3] - box[1])) ** 0.5
        crop = centered_crop(box, image.shape)

        cell = assets.render_cell(
            image, gt[frame], PROMPT, TARGET_ID, crop,
            selected_only=False, gt_categories={PROMPT},
        )
        cells[("ground_truth", phase)] = cell
        cv2.imwrite(str(OUTPUT / f"{phase}_ground_truth.png"), cell)

        for model in list(assets.MODELS)[1:]:
            cell = assets.render_cell(
                image, gt[frame], PROMPT, TARGET_ID, crop,
                predictions[model].get(frame, []), selected[model][EVENT_ID],
                label_all_predictions=True,
                selected_only=False,
                gt_categories={PROMPT},
            )
            cells[(model, phase)] = cell
            cv2.imwrite(str(OUTPUT / f"{phase}_{model}.png"), cell)

    sheet = np.vstack([
        np.hstack([cells[(model, phase)] for phase in assets.PHASE_NAMES])
        for model in assets.MODELS
    ])
    cv2.imwrite(str(OUTPUT / "comparison_clean_4x3.png"), sheet)

    # PPT version requested by the user: three model rows only.  Each cell
    # retains all visible car GT boxes and every returned SAM 3 object ID.
    all_ids_3x3 = np.vstack([
        np.hstack([cells[(model, phase)] for phase in assets.PHASE_NAMES])
        for model in list(assets.MODELS)[1:]
    ])
    cv2.imwrite(str(OUTPUT / "comparison_all_cars_all_ids_3x3.png"), all_ids_3x3)

    # Recreate the previously approved 3x3 paper layout, changing only the
    # final phase from recovered frame 227 to the smaller forward frame 220.
    paper_figures.OUTPUT = OUTPUT
    paper_figures.make_qualitative(
        event,
        paper_figures.selected_ids_and_scores()[0],
        "Scale decrease",
        filename_prefix="comparison_original_layout_3x3_",
        heading_prefix="Scale decrease — ",
        focus_model="sam3_adapted_with_replay",
        forced_frames=[202, 210, 220],
    )

    readme = [
        "# Event 037 — smaller final frame",
        "",
        f"- Event: `{EVENT_ID}`",
        f"- Frames: `{FRAMES}`",
        f"- Scales: before={scales['before']:.2f}, during={scales['during']:.2f}, final={scales['after']:.2f}",
        "- Every visible car GT intersecting each crop is drawn.",
        "- All prediction boxes come from saved model outputs; none are invented.",
        "- Equal-size crops are used for all phases.",
        "- `comparison_original_layout_3x3_scale_decrease_and_recovery.png` keeps the approved 3x3 layout and changes only the final frame.",
        "- `comparison_all_cars_all_ids_3x3.png` shows all visible car GT and all returned IDs for the three models.",
        "",
        "PPT title:",
        "Experiment 2: Tracking under Progressive Scale Decrease, Prompt: car",
    ]
    (OUTPUT / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
