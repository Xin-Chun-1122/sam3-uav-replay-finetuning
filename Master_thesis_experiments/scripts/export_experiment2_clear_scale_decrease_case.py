#!/usr/bin/env python3
"""Export a clear, audit-friendly large-to-small tracking example."""

from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Master_thesis_experiments.scripts import export_experiment2_ppt_assets as assets


EVENT_ID = "scale_decrease_and_recovery-049"
FRAMES = {"before": 66, "during": 81, "after": 97}
PROMPT = "car"
OUTPUT = (
    assets.ROOT
    / "figures/experiment2/ppt_local_tracking_cases"
    / "scale_decrease_clear_alternative"
)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    event = next(
        row for row in assets.load_json(assets.EVENTS_PATH)
        if row["event_id"] == EVENT_ID
    )
    sequence = event["sequence_id"]
    target_id = int(event["target_id"])
    gt = assets.read_all_gt(sequence)
    _, selected = assets.result_index()
    predictions = {
        model: assets.read_predictions(model, EVENT_ID)
        for model in list(assets.MODELS)[1:]
    }

    cells = {}
    scales = {}
    counts = {}
    for phase, frame in FRAMES.items():
        image = cv2.imread(str(assets.image_path(sequence, frame)))
        height, width = image.shape[:2]
        crop = (0, 0, width, height)
        target = next(row for row in gt[frame] if row["target_id"] == target_id)
        box = target["bbox_xyxy"]
        scales[phase] = ((box[2] - box[0]) * (box[3] - box[1])) ** 0.5
        counts[phase] = sum(row["category"] == PROMPT for row in gt[frame])

        ground_truth = assets.render_cell(
            image,
            gt[frame],
            PROMPT,
            target_id,
            crop,
            selected_only=False,
            gt_categories={PROMPT},
        )
        cells[("ground_truth", phase)] = ground_truth
        cv2.imwrite(str(OUTPUT / f"{phase}_ground_truth.png"), ground_truth)

        for model in list(assets.MODELS)[1:]:
            cell = assets.render_cell(
                image,
                gt[frame],
                PROMPT,
                target_id,
                crop,
                predictions[model].get(frame, []),
                selected[model][EVENT_ID],
                label_all_predictions=True,
                selected_only=False,
                gt_categories={PROMPT},
            )
            cells[(model, phase)] = cell
            cv2.imwrite(str(OUTPUT / f"{phase}_{model}.png"), cell)

    four_by_three = np.vstack([
        np.hstack([cells[(model, phase)] for phase in assets.PHASE_NAMES])
        for model in assets.MODELS
    ])
    cv2.imwrite(str(OUTPUT / "comparison_all_cars_all_ids_4x3.png"), four_by_three)

    three_by_three = np.vstack([
        np.hstack([cells[(model, phase)] for phase in assets.PHASE_NAMES])
        for model in list(assets.MODELS)[1:]
    ])
    cv2.imwrite(str(OUTPUT / "comparison_all_cars_all_ids_3x3.png"), three_by_three)

    lines = [
        "# Clear scale-decrease alternative",
        "",
        f"- Event: `{EVENT_ID}`",
        f"- Frames: `{FRAMES}`",
        (
            "- Target scales: "
            f"{scales['before']:.2f} -> {scales['during']:.2f} -> {scales['after']:.2f} px"
        ),
        f"- Car GT counts: `{counts}`",
        "- Every car GT is drawn; every displayed ID comes from saved model output.",
        "- Missed detections are intentionally left as misses and are not fabricated.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
