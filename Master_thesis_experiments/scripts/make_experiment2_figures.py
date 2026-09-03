#!/usr/bin/env python3
"""Create thesis-ready Experiment 2 figures from measured predictions only."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path("/home/alien/sam3/Master_thesis_experiments")
DATA = ROOT / "datasets/raw/VisDrone2019-MOT-test-dev"
EVENTS_PATH = ROOT / "annotations/experiment2/events.json"
RESULTS = ROOT / "results/experiment2"
PREDICTIONS = ROOT / "predictions/experiment2"
OUTPUT = ROOT / "figures/experiment2"

MODELS = [
    ("sam3_original", "Original SAM 3"),
    ("sam3_native_id_tracker", "SAM 3 + Native-ID Refinement"),
    ("sam3_adapted_with_replay", "Adapted SAM 3 with replay"),
]
CONDITIONS = [
    ("partial_occlusion", "Partial occlusion"),
    ("temporary_complete_disappearance", "Temporary disappearance"),
    ("scale_decrease_and_recovery", "Scale decrease and recovery"),
    ("camera_motion_or_blur", "Camera motion / blur"),
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_gt(sequence: str, target_id: int) -> dict[int, list[float]]:
    result = {}
    path = DATA / "annotations" / f"{sequence}.txt"
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split(",")
        if len(fields) < 6 or int(fields[1]) != target_id:
            continue
        frame = int(fields[0])
        x, y, w, h = map(float, fields[2:6])
        result[frame] = [x, y, x + w, y + h]
    return result


def read_predictions(model: str, event_id: str) -> dict[int, list[dict]]:
    result = {}
    path = PREDICTIONS / model / f"{event_id}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        result[int(row["frame_index"])] = row["predictions"]
    return result


def selected_ids_and_scores() -> tuple[dict, dict]:
    selected, scores = {}, {}
    for model, _ in MODELS:
        data = load_json(RESULTS / f"{model}.json")
        selected[model] = {
            row["event_id"]: row["selected_sam3_obj_id"] for row in data["event_results"]
        }
        scores[model] = {
            row["event_id"]: row["frame_recall"] for row in data["event_results"]
        }
    return selected, scores


def representative_events(events: list[dict], scores: dict) -> dict[str, dict]:
    """Select the event closest to the median native-ID Frame Recall per condition."""
    output = {}
    native_scores = scores["sam3_native_id_tracker"]
    for condition, _ in CONDITIONS:
        candidates = [event for event in events if event["condition"] == condition]
        values = np.asarray([native_scores[event["event_id"]] for event in candidates])
        median = float(np.median(values))
        output[condition] = min(
            candidates,
            key=lambda event: (
                abs(native_scores[event["event_id"]] - median),
                event["event_id"],
            ),
        )
    return output


def successful_events(events: list[dict], scores: dict) -> dict[str, dict]:
    """Select a measured native-ID recovery success with the largest baseline gain."""
    output = {}
    native = load_json(RESULTS / "sam3_native_id_tracker.json")["event_results"]
    recovered = {row["event_id"]: row["same_id_recovered"] for row in native}
    for condition, _ in CONDITIONS:
        candidates = [
            event for event in events
            if event["condition"] == condition and recovered[event["event_id"]]
        ]
        output[condition] = min(
            candidates,
            key=lambda event: (
                -(
                    scores["sam3_native_id_tracker"][event["event_id"]]
                    - max(
                        scores["sam3_original"][event["event_id"]],
                        scores["sam3_adapted_with_replay"][event["event_id"]],
                    )
                ),
                -scores["sam3_native_id_tracker"][event["event_id"]],
                event["event_id"],
            ),
        )
    return output


def adapted_successful_events(events: list[dict], scores: dict) -> dict[str, dict]:
    """Select measured adapted-model successes with the largest gain over Original."""
    adapted = load_json(RESULTS / "sam3_adapted_with_replay.json")["event_results"]
    recovered = {row["event_id"]: row["same_id_recovered"] for row in adapted}
    output = {}
    for condition, _ in CONDITIONS:
        candidates = [
            event for event in events
            if event["condition"] == condition and recovered[event["event_id"]]
        ]
        output[condition] = min(
            candidates,
            key=lambda event: (
                -(
                    scores["sam3_adapted_with_replay"][event["event_id"]]
                    - scores["sam3_original"][event["event_id"]]
                ),
                -scores["sam3_adapted_with_replay"][event["event_id"]],
                event["event_id"],
            ),
        )
    return output


def image_path(sequence: str, frame: int) -> Path:
    direct = DATA / "sequences" / sequence / f"{frame:07d}.jpg"
    if direct.exists():
        return direct
    matches = sorted((DATA / "sequences" / sequence).glob(f"*{frame}.jpg"))
    if not matches:
        raise FileNotFoundError(direct)
    return matches[0]


def interpolated_box(gt: dict[int, list[float]], frame: int) -> list[float]:
    if frame in gt:
        return gt[frame]
    before = [value for value in gt if value < frame]
    after = [value for value in gt if value > frame]
    if before and after:
        left, right = max(before), min(after)
        alpha = (frame - left) / (right - left)
        return (
            np.asarray(gt[left]) * (1 - alpha) + np.asarray(gt[right]) * alpha
        ).tolist()
    nearest = min(gt, key=lambda value: abs(value - frame))
    return gt[nearest]


def crop_bounds(image: np.ndarray, reference: list[float]) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = reference
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    object_side = max(x2 - x1, y2 - y1)
    crop_w = int(np.clip(object_side * 12, 220, min(600, width)))
    crop_h = int(min(height, crop_w * 0.75))
    x0 = int(np.clip(cx - crop_w / 2, 0, max(0, width - crop_w)))
    y0 = int(np.clip(cy - crop_h / 2, 0, max(0, height - crop_h)))
    return x0, y0, x0 + crop_w, y0 + crop_h


def add_box(axis, box, crop, color, label, linewidth=2.2):
    x0, y0, _, _ = crop
    x1, y1, x2, y2 = box
    axis.add_patch(
        Rectangle(
            (x1 - x0, y1 - y0),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
        )
    )
    axis.text(
        x1 - x0,
        max(3, y1 - y0 - 4),
        label,
        color="white",
        fontsize=7,
        bbox={"facecolor": color, "alpha": 0.85, "pad": 1.5, "edgecolor": "none"},
    )


def intersects_crop(box, crop) -> bool:
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = crop
    return min(x2, cx2) > max(x1, cx1) and min(y2, cy2) > max(y1, cy1)


def iou(box_a, box_b) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def make_qualitative(
    event: dict,
    selected_ids: dict,
    condition_title: str,
    filename_prefix: str = "qualitative_",
    heading_prefix: str = "",
    focus_model: str | None = None,
    strict_comparative: bool = False,
    forced_frames: list[int] | None = None,
) -> tuple[Path, Path]:
    event_id = event["event_id"]
    gt = read_gt(event["sequence_id"], int(event["target_id"]))
    model_predictions = {
        model: read_predictions(model, event_id) for model, _ in MODELS
    }
    frames = [
        max(event["clip_start_frame"], event["start_frame"] - 1),
        (event["start_frame"] + event["end_frame"]) // 2,
        min(event["clip_end_frame"], event["end_frame"] + 1),
    ]
    if focus_model is not None:
        focus_id = selected_ids[focus_model][event_id]
        focus_predictions = model_predictions[focus_model]

        def model_matches(model, candidate):
            if candidate not in gt:
                return False
            model_id = selected_ids[model][event_id]
            rows = [
                pred for pred in model_predictions[model].get(candidate, [])
                if pred["sam3_obj_id"] == model_id
            ]
            return bool(rows and iou(rows[0]["bbox_xyxy"], gt[candidate]) >= 0.5)

        def matching(candidate_frames):
            matches = []
            for candidate in candidate_frames:
                if candidate not in gt:
                    continue
                rows = [
                    pred for pred in focus_predictions.get(candidate, [])
                    if pred["sam3_obj_id"] == focus_id
                ]
                if rows and iou(rows[0]["bbox_xyxy"], gt[candidate]) >= 0.5:
                    matches.append(candidate)
            return matches

        def strict_matching(candidate_frames):
            return [
                candidate for candidate in candidate_frames
                if model_matches(focus_model, candidate)
                and not model_matches("sam3_original", candidate)
                and not model_matches("sam3_native_id_tracker", candidate)
            ]

        before = matching(
            range(event["clip_start_frame"], event["start_frame"])
        )
        after = matching(
            range(event["end_frame"] + 1, event["clip_end_frame"] + 1)
        )
        if before:
            frames[0] = before[-1]
        if event["condition"] != "temporary_complete_disappearance":
            during_range = range(event["start_frame"], event["end_frame"] + 1)
            during = (
                strict_matching(during_range)
                if strict_comparative else matching(during_range)
            )
            if strict_comparative and not during:
                during = matching(during_range)
            if during:
                midpoint = (event["start_frame"] + event["end_frame"]) / 2
                frames[1] = min(during, key=lambda value: abs(value - midpoint))
        strict_after = strict_matching(
            range(event["end_frame"] + 1, event["clip_end_frame"] + 1)
        )
        if strict_comparative and strict_after:
            frames[2] = strict_after[0]
        elif after:
            frames[2] = after[0]
    if forced_frames is not None:
        if len(forced_frames) != 3:
            raise ValueError("forced_frames must contain before/during/after frames")
        frames = list(forced_frames)
    phase_names = ["Before event", "During event", "After event"]
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.0), constrained_layout=True)
    for column, (frame, phase) in enumerate(zip(frames, phase_names)):
        source = cv2.cvtColor(
            cv2.imread(str(image_path(event["sequence_id"], frame))),
            cv2.COLOR_BGR2RGB,
        )
        reference = interpolated_box(gt, frame)
        crop = crop_bounds(source, reference)
        x0, y0, x1, y1 = crop
        for row, (model, model_title) in enumerate(MODELS):
            axis = axes[row, column]
            axis.imshow(source[y0:y1, x0:x1])
            if frame in gt:
                add_box(axis, gt[frame], crop, "#22a884", "GT")
            native_id = selected_ids[model][event_id]
            matches = [
                pred
                for pred in model_predictions[model].get(frame, [])
                if pred["sam3_obj_id"] == native_id
            ]
            if matches:
                pred = matches[0]
                if intersects_crop(pred["bbox_xyxy"], crop):
                    add_box(
                        axis,
                        pred["bbox_xyxy"],
                        crop,
                        "#d62728",
                        f"ID {native_id}",
                    )
                else:
                    axis.text(
                        0.03,
                        0.94,
                        f"ID {native_id}: drifted outside crop",
                        transform=axis.transAxes,
                        va="top",
                        color="white",
                        fontsize=8,
                        bbox={"facecolor": "#d62728", "alpha": 0.85, "pad": 2},
                    )
            else:
                axis.text(
                    0.03,
                    0.94,
                    f"ID {native_id}: not detected",
                    transform=axis.transAxes,
                    va="top",
                    color="white",
                    fontsize=8,
                    bbox={"facecolor": "#d62728", "alpha": 0.85, "pad": 2},
                )
            if row == 0:
                axis.set_title(f"{phase}\nFrame {frame}", fontsize=10)
            if column == 0:
                axis.set_ylabel(model_title, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        f"{heading_prefix}{condition_title}: {event_id} ({event['category']})\n"
        "Green: ground truth; Red: fixed native SAM 3 track ID",
        fontsize=13,
    )
    png = OUTPUT / f"{filename_prefix}{event['condition']}.png"
    pdf = OUTPUT / f"{filename_prefix}{event['condition']}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def make_overall_chart() -> tuple[Path, Path]:
    metrics = ["Frame Recall", "Recovery Rate", "AP50"]
    values = []
    for model, _ in MODELS:
        overall = load_json(RESULTS / f"{model}.json")["overall"]
        values.append(
            [overall["frame_recall"], overall["recovery_rate"], overall["track_ap50"]]
        )
    x = np.arange(len(metrics))
    width = 0.24
    colors = ["#4c78a8", "#22a884", "#e07a5f"]
    fig, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    for index, ((_, label), row, color) in enumerate(zip(MODELS, values, colors)):
        bars = axis.bar(x + (index - 1) * width, row, width, label=label, color=color)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axis.set_xticks(x, metrics)
    axis.set_ylim(0, 0.62)
    axis.set_ylabel("Score")
    axis.set_title("Overall Tracking Performance on VisDrone-MOT (Measured)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    png = OUTPUT / "overall_tracking_metrics.png"
    pdf = OUTPUT / "overall_tracking_metrics.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def make_scale_chart() -> tuple[Path, Path]:
    phases = ["Before tiny", "Tiny", "After recovery"]
    fig, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    colors = ["#4c78a8", "#22a884", "#e07a5f"]
    label_offsets = [-14, 8, 8]
    for series_index, ((model, label), color) in enumerate(zip(MODELS, colors)):
        condition = load_json(RESULTS / f"{model}.json")["by_condition"][
            "scale_decrease_and_recovery"
        ]
        phase = condition["phase_frame_recall"]
        values = [
            phase["pre"]["frame_recall"],
            phase["low_reliability"]["frame_recall"],
            phase["post"]["frame_recall"],
        ]
        axis.plot(phases, values, marker="o", linewidth=2.2, label=label, color=color)
        for index, value in enumerate(values):
            axis.annotate(f"{value:.3f}", (index, value),
                          xytext=(0, label_offsets[series_index]),
                          textcoords="offset points", ha="center", fontsize=8)
    axis.set_ylim(0, 0.72)
    axis.set_ylabel("Frame Recall")
    axis.set_title("Tracking Across Reliable → Tiny → Reliable Scale Transition")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    png = OUTPUT / "scale_recovery_analysis.png"
    pdf = OUTPUT / "scale_recovery_analysis.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    events = load_json(EVENTS_PATH)
    selected_ids, scores = selected_ids_and_scores()
    representatives = representative_events(events, scores)
    successes = successful_events(events, scores)
    adapted_successes = adapted_successful_events(events, scores)
    comparative_wins = {
        "partial_occlusion": next(
            event for event in events if event["event_id"] == "partial_occlusion-007"
        ),
        "temporary_complete_disappearance": next(
            event for event in events
            if event["event_id"] == "temporary_complete_disappearance-025"
        ),
        "scale_decrease_and_recovery": next(
            event for event in events
            if event["event_id"] == "scale_decrease_and_recovery-037"
        ),
        "camera_motion_or_blur": next(
            event for event in events
            if event["event_id"] == "camera_motion_or_blur-003"
        ),
    }
    created = [*make_overall_chart(), *make_scale_chart()]
    manifest = {
        "selection_rule": (
            "For each condition, select the event whose measured native-ID-refinement "
            "Frame Recall is closest to that condition's median; event_id breaks ties."
        ),
        "success_selection_rule": (
            "Among events with measured same-ID recovery, select the largest native-ID "
            "Frame Recall gain over both other baselines, then the highest native-ID "
            "Frame Recall; event_id breaks ties."
        ),
        "figures": [],
        "success_figures": [],
        "adapted_success_figures": [],
        "adapted_comparative_win_figures": [],
    }
    for condition, title in CONDITIONS:
        event = representatives[condition]
        files = make_qualitative(event, selected_ids, title)
        created.extend(files)
        manifest["figures"].append(
            {"condition": condition, "event_id": event["event_id"],
             "files": [str(path) for path in files]}
        )
        success_event = successes[condition]
        success_files = make_qualitative(
            success_event,
            selected_ids,
            title,
            filename_prefix="qualitative_success_",
            heading_prefix="Successful recovery example — ",
        )
        created.extend(success_files)
        manifest["success_figures"].append(
            {
                "condition": condition,
                "event_id": success_event["event_id"],
                "files": [str(path) for path in success_files],
            }
        )
        adapted_event = adapted_successes[condition]
        adapted_files = make_qualitative(
            adapted_event,
            selected_ids,
            title,
            filename_prefix="qualitative_adapted_success_",
            heading_prefix="Adapted SAM 3 with replay success — ",
            focus_model="sam3_adapted_with_replay",
        )
        created.extend(adapted_files)
        manifest["adapted_success_figures"].append(
            {
                "condition": condition,
                "event_id": adapted_event["event_id"],
                "files": [str(path) for path in adapted_files],
            }
        )
        if condition in comparative_wins:
            comparative_event = comparative_wins[condition]
            camera_relaxed = condition == "camera_motion_or_blur"
            comparative_files = make_qualitative(
                comparative_event,
                selected_ids,
                title,
                filename_prefix="qualitative_adapted_comparative_win_",
                heading_prefix=(
                    "Adapted succeeds while Native-ID fails — "
                    if camera_relaxed
                    else "Adapted-only correct tracking example — "
                ),
                focus_model="sam3_adapted_with_replay",
                strict_comparative=not camera_relaxed,
            )
            created.extend(comparative_files)
            manifest["adapted_comparative_win_figures"].append(
                {
                    "condition": condition,
                    "event_id": comparative_event["event_id"],
                    "files": [str(path) for path in comparative_files],
                }
            )
    (OUTPUT / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
