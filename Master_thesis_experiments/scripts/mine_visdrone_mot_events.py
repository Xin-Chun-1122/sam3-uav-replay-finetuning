#!/usr/bin/env python3
"""Mine reproducible natural tracking events from VisDrone2019-MOT test-dev."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


TARGET_CATEGORIES = {1: "person", 2: "person", 4: "car", 5: "van", 6: "truck", 9: "bus", 10: "motorcycle"}
CONDITIONS = (
    "partial_occlusion",
    "temporary_complete_disappearance",
    "scale_decrease_and_recovery",
    "camera_motion_or_blur",
)


@dataclass(frozen=True)
class Ann:
    frame: int
    target_id: int
    left: float
    top: float
    width: float
    height: float
    category_id: int
    truncation: int
    occlusion: int

    @property
    def scale(self) -> float:
        return math.sqrt(self.width * self.height)


def stable_key(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def runs(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    result = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            result.append((start, previous))
            start = value
        previous = value
    result.append((start, previous))
    return result


def read_annotations(path: Path) -> list[Ann]:
    result = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 10:
            continue
        frame, target_id = int(parts[0]), int(parts[1])
        left, top, width, height = map(float, parts[2:6])
        category_id = int(parts[7])
        truncation, occlusion = int(parts[8]), int(parts[9])
        if category_id in TARGET_CATEGORIES and width > 0 and height > 0 and truncation <= 1:
            result.append(Ann(frame, target_id, left, top, width, height, category_id, truncation, occlusion))
    return result


def event(condition: str, sequence: str, target_id: int | str, start: int, end: int,
          total_frames: int, category: str, notes: str, **extra) -> dict:
    return {
        "event_id": "",
        "condition": condition,
        "sequence_id": sequence,
        "target_id": target_id,
        "category": category,
        "clip_start_frame": max(1, start - 15),
        "clip_end_frame": min(total_frames, end + 15),
        "start_frame": start,
        "end_frame": end,
        "duration_frames": end - start + 1,
        "source_type": "natural",
        "notes": notes,
        **extra,
    }


def mine_target_events(sequence: str, anns: list[Ann], total_frames: int) -> dict[str, list[dict]]:
    by_target: dict[int, list[Ann]] = defaultdict(list)
    for ann in anns:
        by_target[ann.target_id].append(ann)
    output = {condition: [] for condition in CONDITIONS}
    for target_id, target_anns in by_target.items():
        target_anns.sort(key=lambda item: item.frame)
        by_frame = {item.frame: item for item in target_anns}
        category = TARGET_CATEGORIES[target_anns[0].category_id]

        partial_frames = [item.frame for item in target_anns if item.occlusion == 1]
        for start, end in runs(partial_frames):
            if end - start + 1 < 2:
                continue
            before = next((by_frame[f] for f in range(start - 1, max(0, start - 16), -1) if f in by_frame), None)
            after = next((by_frame[f] for f in range(end + 1, min(total_frames, end + 15) + 1) if f in by_frame), None)
            if before is None or after is None:
                continue
            output["partial_occlusion"].append(event(
                "partial_occlusion", sequence, target_id, start, end, total_frames, category,
                "Official VisDrone occlusion=1 contiguous run, bracketed by annotated target frames.",
            ))

        observed = sorted(by_frame)
        for left_frame, right_frame in zip(observed, observed[1:]):
            gap = right_frame - left_frame - 1
            if not 2 <= gap <= 90:
                continue
            left_ann, right_ann = by_frame[left_frame], by_frame[right_frame]
            # Large center jumps are more likely track exit/re-entry than temporary disappearance.
            lc = (left_ann.left + left_ann.width / 2, left_ann.top + left_ann.height / 2)
            rc = (right_ann.left + right_ann.width / 2, right_ann.top + right_ann.height / 2)
            normalized_jump = math.dist(lc, rc) / max(1.0, math.sqrt(left_ann.width * left_ann.height))
            if normalized_jump > max(6.0, gap * 1.5):
                continue
            output["temporary_complete_disappearance"].append(event(
                "temporary_complete_disappearance", sequence, target_id, left_frame + 1,
                right_frame - 1, total_frames, category,
                "Internal GT-ID annotation gap bracketed by the same target ID.",
                pre_visible_frame=left_frame, post_visible_frame=right_frame,
            ))

        tiny_frames = [item.frame for item in target_anns if item.scale < 32]
        for start, end in runs(tiny_frames):
            if end - start + 1 < 2:
                continue
            before = next((by_frame[f] for f in range(start - 1, max(0, start - 31), -1) if f in by_frame), None)
            after = next((by_frame[f] for f in range(end + 1, min(total_frames, end + 30) + 1) if f in by_frame), None)
            if before is None or after is None or before.scale < 32 or after.scale < 32:
                continue
            output["scale_decrease_and_recovery"].append(event(
                "scale_decrease_and_recovery", sequence, target_id, start, end, total_frames, category,
                "Target transitions from small/regular to tiny and later returns to small/regular.",
                pre_scale=before.scale, minimum_scale=min(by_frame[f].scale for f in range(start, end + 1)),
                post_scale=after.scale,
            ))
    return output


def camera_candidates(sequence: str, frames_dir: Path, anns: list[Ann], total_frames: int) -> list[dict]:
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if len(frame_paths) < 3:
        return []
    blur, motion = [], []
    previous = None
    for frame_index, path in enumerate(frame_paths, 1):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            blur.append(float("nan")); motion.append(float("nan")); continue
        resized = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
        blur.append(float(cv2.Laplacian(resized, cv2.CV_64F).var()))
        if previous is None:
            motion.append(0.0)
        else:
            shift, _ = cv2.phaseCorrelate(np.float32(previous), np.float32(resized))
            motion.append(float(math.hypot(*shift)))
        previous = resized
    finite_blur = np.asarray([value for value in blur if math.isfinite(value)])
    finite_motion = np.asarray([value for value in motion if math.isfinite(value)])
    if not finite_blur.size or not finite_motion.size:
        return []
    blur_cut = float(np.quantile(finite_blur, 0.10))
    motion_cut = float(np.quantile(finite_motion, 0.90))
    difficult = [i + 1 for i, (b, m) in enumerate(zip(blur, motion)) if b <= blur_cut or m >= motion_cut]
    by_frame: dict[int, list[Ann]] = defaultdict(list)
    for ann in anns:
        by_frame[ann.frame].append(ann)
    result = []
    for start, end in runs(difficult):
        if end - start + 1 < 2:
            continue
        mid = (start + end) // 2
        nearby = next((by_frame[f] for distance in range(0, 11)
                       for f in (mid - distance, mid + distance) if by_frame.get(f)), [])
        if not nearby:
            continue
        representative = max(nearby, key=lambda item: item.width * item.height)
        affected = sorted({ann.target_id for f in range(start, end + 1) for ann in by_frame.get(f, [])})
        result.append(event(
            "camera_motion_or_blur", sequence, representative.target_id, start, end, total_frames,
            TARGET_CATEGORIES[representative.category_id],
            "Sequence-relative bottom-10% sharpness or top-10% global phase-correlation motion run.",
            affected_target_ids=affected,
            median_blur=float(statistics.median(blur[start - 1:end])),
            median_global_motion=float(statistics.median(motion[start - 1:end])),
            blur_threshold=blur_cut,
            motion_threshold=motion_cut,
        ))
    return result


def diverse_select(candidates: list[dict], count: int, seed: int, condition: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        grouped[item["sequence_id"]].append(item)
    for sequence, items in grouped.items():
        items.sort(key=lambda item: stable_key(seed, f"{condition}:{sequence}:{item['target_id']}:{item['start_frame']}"))
    sequences = sorted(grouped, key=lambda value: stable_key(seed, f"{condition}:{value}"))
    selected, used_intervals = [], defaultdict(list)
    while len(selected) < count:
        progressed = False
        for sequence in sequences:
            while grouped[sequence]:
                candidate = grouped[sequence].pop(0)
                key = (sequence, candidate["target_id"])
                if any(not (candidate["end_frame"] < start or candidate["start_frame"] > end)
                       for start, end in used_intervals[key]):
                    continue
                selected.append(candidate)
                used_intervals[key].append((candidate["clip_start_frame"], candidate["clip_end_frame"]))
                progressed = True
                break
            if len(selected) >= count:
                break
        if not progressed:
            break
    for index, item in enumerate(selected, 1):
        item["event_id"] = f"{condition}-{index:03d}"
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partial-count", type=int, default=50)
    parser.add_argument("--disappearance-count", type=int, default=50)
    parser.add_argument("--scale-count", type=int, default=50)
    parser.add_argument("--camera-count", type=int, default=10)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    annotation_dir, sequence_dir = root / "annotations", root / "sequences"
    if not annotation_dir.is_dir() or not sequence_dir.is_dir():
        raise FileNotFoundError(f"Expected annotations/ and sequences/ under {root}")
    all_candidates = {condition: [] for condition in CONDITIONS}
    sequence_sizes = {}
    for ann_path in sorted(annotation_dir.glob("*.txt")):
        sequence = ann_path.stem
        frames_dir = sequence_dir / sequence
        total_frames = len(list(frames_dir.glob("*.jpg")))
        if not total_frames:
            continue
        anns = read_annotations(ann_path)
        sequence_sizes[sequence] = {"frames": total_frames, "target_annotations": len(anns)}
        mined = mine_target_events(sequence, anns, total_frames)
        for condition in CONDITIONS:
            all_candidates[condition].extend(mined[condition])
        all_candidates["camera_motion_or_blur"].extend(camera_candidates(sequence, frames_dir, anns, total_frames))

    requested = {
        "partial_occlusion": args.partial_count,
        "temporary_complete_disappearance": args.disappearance_count,
        "scale_decrease_and_recovery": args.scale_count,
        "camera_motion_or_blur": args.camera_count,
    }
    selected = []
    summary = {"dataset_root": str(root), "seed": args.seed, "sequence_sizes": sequence_sizes, "conditions": {}}
    for condition in CONDITIONS:
        condition_selected = diverse_select(all_candidates[condition], requested[condition], args.seed, condition)
        selected.extend(condition_selected)
        targets_per_sequence: dict[str, set[int | str]] = defaultdict(set)
        for item in condition_selected:
            if condition == "camera_motion_or_blur":
                targets_per_sequence[item["sequence_id"]].update(item.get("affected_target_ids", [item["target_id"]]))
            else:
                targets_per_sequence[item["sequence_id"]].add(item["target_id"])
        summary["conditions"][condition] = {
            "requested": requested[condition],
            "natural_candidates": len(all_candidates[condition]),
            "selected_natural": len(condition_selected),
            "events": len(condition_selected),
            "shortfall_for_synthetic": requested[condition] - len(condition_selected),
            "sequences": len({item["sequence_id"] for item in condition_selected}),
            "unique_targets": len({(item["sequence_id"], item["target_id"]) for item in condition_selected}),
            "median_unique_targets_per_sequence": statistics.median(
                [len(values) for values in targets_per_sequence.values()]
            ) if targets_per_sequence else None,
            "median_duration_frames": statistics.median([item["duration_frames"] for item in condition_selected]) if condition_selected else None,
        }
    selected.sort(key=lambda item: (CONDITIONS.index(item["condition"]), item["event_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["conditions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
