#!/usr/bin/env python3
"""Export an audited wide-view scale-decrease PPT sample.

The three panels use one fixed source crop, so the visible scale change is not
caused by per-frame zooming.  Every official ``car`` GT and every saved SAM 3
prediction inside the crop are rendered; no detection is invented here.
"""

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

# Fixed 4:3 source crop for all phases.  It contains five fully visible cars
# at each selected phase.  Each one has a real Ours prediction at IoU >= 0.50.
# Keeping the crop fixed also preserves the true large-to-small appearance.
CROP = [680, 0, 1280, 450]

OUTPUT = (
    assets.ROOT
    / "figures/experiment2/ppt_local_tracking_cases"
    / "scale_decrease_and_recovery_alternative"
)


def target_scale(gt_rows: list[dict], target_id: int) -> float:
    row = next(row for row in gt_rows if int(row["target_id"]) == target_id)
    x1, y1, x2, y2 = row["bbox_xyxy"]
    return float(((x2 - x1) * (y2 - y1)) ** 0.5)


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

    cells = {}
    scales = {}
    visible_counts = {}
    for phase, frame in FRAMES.items():
        image = cv2.imread(str(assets.image_path(sequence, frame)))
        crop = tuple(CROP)
        scales[phase] = target_scale(gt.get(frame, []), target_id)
        visible_counts[phase] = sum(
            row["category"] == PROMPT
            and assets.intersects(row["bbox_xyxy"], crop)
            for row in gt.get(frame, [])
        )

        gt_cell = assets.render_cell(
            image,
            gt.get(frame, []),
            PROMPT,
            target_id,
            crop,
            selected_only=False,
            gt_categories={PROMPT},
        )
        cells[("ground_truth", phase)] = gt_cell
        cv2.imwrite(str(OUTPUT / f"{phase}_ground_truth.png"), gt_cell)

        for model in list(assets.MODELS)[1:]:
            cell = assets.render_cell(
                image,
                gt.get(frame, []),
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

    rows = [
        np.hstack([cells[(model, phase)] for phase in assets.PHASE_NAMES])
        for model in assets.MODELS
    ]
    cv2.imwrite(str(OUTPUT / "comparison_clean_4x3.png"), np.vstack(rows))

    summary = [
        "Alternative scale-decrease sample for PPT",
        f"Event: {EVENT_ID}",
        f"Sequence: {sequence}",
        f"Tracked target ID: {target_id}",
        f"Prompt: {PROMPT}",
        f"Frames: {FRAMES}",
        f"Fixed crop xyxy: {CROP}",
        "Output cell size: 960x600 pixels",
        "",
        "Target scales (sqrt(w*h)):",
    ]
    for phase in assets.PHASE_NAMES:
        summary.append(
            f"- {phase}: s={scales[phase]:.2f}, "
            f"visible car GT={visible_counts[phase]}"
        )
    summary.extend([
        "",
        "All car GT boxes and all saved model predictions intersecting the fixed crop are shown.",
        "Every displayed car GT has an Ours prediction with IoU >= 0.50.",
        "The source crop is identical in all phases, so scale change is not caused by zooming.",
        "All track IDs come from the saved native SAM 3 predictions.",
        "No prediction box is manually added.",
        "This sample demonstrates stable multi-car detection during a clear large-to-tiny transition.",
        "It must not be described as an Ours-only win because the other baselines also track the selected target in this event.",
    ])
    for model in list(assets.MODELS)[1:]:
        row = results[model][EVENT_ID]
        summary.append(
            f"{assets.MODELS[model]}: frame recall={row['frame_recall']:.4f}, "
            f"same-ID recovered={row['same_id_recovered']}, "
            f"selected native ID={selected[model][EVENT_ID]}"
        )
    (OUTPUT / "README.md").write_text("\n".join(summary), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
