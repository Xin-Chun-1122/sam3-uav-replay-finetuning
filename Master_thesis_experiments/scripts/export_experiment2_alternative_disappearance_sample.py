#!/usr/bin/env python3
"""Export an alternative audited PPT sample for temporary disappearance."""

from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Master_thesis_experiments.scripts import export_experiment2_ppt_assets as assets


EVENT_ID = "temporary_complete_disappearance-049"
FRAMES = {"before": 353, "during": 393, "after": 434}
PROMPT = "person"
# Audited 4:3 landscape crops.  Every official person GT intersecting these
# crops is fully contained and has a real Ours prediction at IoU >= 0.50.
# The three crops contain 3, 2, and 2 people respectively.
CROPS = {
    "before": [704, 0, 1344, 480],
    "during": [560, 0, 1360, 600],
    "after": [494, 0, 1294, 600],
}
OUTPUT = (
    assets.ROOT
    / "figures/experiment2/ppt_local_tracking_cases"
    / "temporary_complete_disappearance_alternative"
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    events = {row["event_id"]: row for row in assets.load_json(assets.EVENTS_PATH)}
    event = events[EVENT_ID]
    sequence = event["sequence_id"]
    target_id = int(event["target_id"])
    gt = assets.read_all_gt(sequence)
    results, selected = assets.result_index()
    predictions = {
        model: assets.read_predictions(model, EVENT_ID)
        for model in list(assets.MODELS)[1:]
    }
    crops = CROPS
    cells = {}
    for phase, frame in FRAMES.items():
        image = cv2.imread(str(assets.image_path(sequence, frame)))
        crop = crops[phase]
        gt_cell = assets.render_cell(
            image, gt.get(frame, []), PROMPT, target_id, crop,
            selected_only=False,
        )
        cells[("ground_truth", phase)] = gt_cell
        cv2.imwrite(str(OUTPUT / f"{phase}_ground_truth.png"), gt_cell)
        for model in list(assets.MODELS)[1:]:
            phase_predictions = predictions[model].get(frame, [])
            cell = assets.render_cell(
                image,
                gt.get(frame, []),
                PROMPT,
                target_id,
                crop,
                phase_predictions,
                selected[model][EVENT_ID],
                label_all_predictions=True,
                selected_only=False,
            )
            cells[(model, phase)] = cell
            cv2.imwrite(str(OUTPUT / f"{phase}_{model}.png"), cell)

    rows = []
    for model in assets.MODELS:
        rows.append(np.hstack([cells[(model, phase)] for phase in assets.PHASE_NAMES]))
    cv2.imwrite(str(OUTPUT / "comparison_clean_4x3.png"), np.vstack(rows))

    # PPT-focused comparison.  This event demonstrates a real improvement over
    # Original SAM 3; the Native-ID baseline also recovers and is therefore not
    # used to support an "Ours-only" qualitative claim.
    ppt_rows = []
    for model in ("ground_truth", "sam3_original", "sam3_adapted_with_replay"):
        ppt_rows.append(np.hstack([cells[(model, phase)] for phase in assets.PHASE_NAMES]))
    cv2.imwrite(str(OUTPUT / "comparison_ours_vs_original_3x3.png"), np.vstack(ppt_rows))

    summary = [
        "Alternative temporary complete disappearance sample",
        f"Event: {EVENT_ID}",
        f"Sequence: {sequence}",
        f"Target ID: {target_id}",
        f"Prompt: {PROMPT}",
        f"Frames: {FRAMES}",
        f"Natural disappearance duration: {event['duration_frames']} frames",
        "",
    ]
    for model in list(assets.MODELS)[1:]:
        row = results[model][EVENT_ID]
        summary.append(
            f"{assets.MODELS[model]}: frame recall={row['frame_recall']:.4f}, "
            f"same-ID recovered={row['same_id_recovered']}, "
            f"selected native ID={selected[model][EVENT_ID]}"
        )
    summary.extend([
        "",
        "All boxes are rendered from official VisDrone-MOT GT and saved model predictions.",
        "All person GT boxes and all model predictions intersecting the local crop are shown.",
        "Ours matches every displayed official person GT at IoU >= 0.50 in all three phases.",
        "The 4:3 crops contain 3, 2, and 2 fully visible person instances.",
        "Use comparison_ours_vs_original_3x3.png to claim improvement over Original SAM 3 only.",
        "Do not claim that Ours is the only successful method: Native-ID also recovers this event.",
        "No prediction box is manually invented.",
    ])
    (OUTPUT / "README.md").write_text("\n".join(summary), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
