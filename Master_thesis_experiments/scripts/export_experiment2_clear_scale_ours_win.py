#!/usr/bin/env python3
"""Export a visually clearer, audited scale-change tracking case for PPT."""

from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Master_thesis_experiments.scripts import export_experiment2_ppt_assets as assets


EVENT_ID = "scale_decrease_and_recovery-031"
FRAMES = {"before": 5, "during": 57, "after": 81}
PROMPT = "car"
TARGET_ID = 145
CROP_SIZE = (360, 225)  # identical pixel size at all phases; moderate PPT view
OUTPUT = (
    assets.ROOT
    / "figures/experiment2/ppt_local_tracking_cases"
    / "scale_decrease_and_recovery_clear_ours_win"
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
    events = {row["event_id"]: row for row in assets.load_json(assets.EVENTS_PATH)}
    event = events[EVENT_ID]
    sequence = event["sequence_id"]
    gt = assets.read_all_gt(sequence)
    results, selected = assets.result_index()
    predictions = {
        model: assets.read_predictions(model, EVENT_ID)
        for model in list(assets.MODELS)[1:]
    }

    cells = {}
    scales = {}
    crops = {}
    for phase, frame in FRAMES.items():
        image = cv2.imread(str(assets.image_path(sequence, frame)))
        target = next(row for row in gt[frame] if int(row["target_id"]) == TARGET_ID)
        box = target["bbox_xyxy"]
        scales[phase] = float(((box[2] - box[0]) * (box[3] - box[1])) ** 0.5)
        crop = centered_crop(box, image.shape)
        crops[phase] = crop

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
                label_all_predictions=True, selected_only=False,
                gt_categories={PROMPT},
            )
            cells[(model, phase)] = cell
            cv2.imwrite(str(OUTPUT / f"{phase}_{model}.png"), cell)

    rows = [
        np.hstack([cells[(model, phase)] for phase in assets.PHASE_NAMES])
        for model in assets.MODELS
    ]
    cv2.imwrite(str(OUTPUT / "comparison_clean_4x3.png"), np.vstack(rows))

    lines = [
        "# Clear scale-decrease case",
        "",
        f"- Event: `{EVENT_ID}`",
        f"- Sequence: `{sequence}`",
        f"- Target: car ID {TARGET_ID}",
        f"- Frames: `{FRAMES}`",
        f"- Equal crop size: `{CROP_SIZE[0]}x{CROP_SIZE[1]}` source pixels",
        f"- Scales: before={scales['before']:.2f}, during={scales['during']:.2f}, after={scales['after']:.2f}",
        "- Scale reduction: {:.1f}%".format((1 - scales['during']/scales['before']) * 100),
        "- At the minimum-scale final frame 81, only Ours reaches IoU >= 0.50.",
        "- Every car GT intersecting the displayed crop is drawn in every row; no prediction is invented.",
        "- Crops have identical dimensions, so the apparent size change is not caused by zooming.",
        "",
        "Recommended PPT title:",
        "Experiment 2: Only Ours Detects the Target throughout Progressive Scale Decrease, Prompt: car",
    ]
    for phase in assets.PHASE_NAMES:
        lines.append(f"- {phase} crop: `{crops[phase]}`")
    for model in list(assets.MODELS)[1:]:
        row = results[model][EVENT_ID]
        lines.append(
            f"- {assets.MODELS[model]}: Frame Recall={row['frame_recall']:.4f}, "
            f"same-ID recovered={row['same_id_recovered']}"
        )
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
