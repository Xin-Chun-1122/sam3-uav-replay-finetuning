#!/usr/bin/env python3
"""
Temporal Reasoning Engine for SAM3-I UAV Inference.

Handles complex time-series prompts:
1. Detect new fire/smoke that wasn't in previous frames
2. Track objects across frames to avoid duplicate reports
3. Spatial proximity analysis (e.g., car near smoke)
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch


@dataclass
class Detection:
    """Single detection result."""
    box: List[float]        # [x1, y1, x2, y2]
    label: str
    score: float
    prompt: str
    frame_idx: int
    timestamp: float = field(default_factory=time.time)
    track_id: Optional[int] = None


@dataclass
class TrackedObject:
    """An object being tracked across frames."""
    track_id: int
    label: str
    last_box: List[float]
    last_frame: int
    first_frame: int
    confidence: float
    reported: bool = False  # Whether this has already been reported
    history: List[Detection] = field(default_factory=list)


def iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection over Union for two boxes [x1,y1,x2,y2]."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union_area = area_a + area_b - inter_area

    return inter_area / (union_area + 1e-6)


def box_center(box: List[float]) -> Tuple[float, float]:
    """Return center (cx, cy) of a box."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def distance_between_boxes(box_a: List[float], box_b: List[float]) -> float:
    """Euclidean distance between centers of two boxes."""
    cx_a, cy_a = box_center(box_a)
    cx_b, cy_b = box_center(box_b)
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


class TemporalReasoner:
    """
    Tracks detections across video frames and supports complex temporal prompts.

    Usage:
        reasoner = TemporalReasoner()
        for frame_idx, detections in enumerate(all_frame_detections):
            reasoner.update(frame_idx, detections)
            new_events = reasoner.get_new_events(frame_idx)
            print(new_events)
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,         # IoU to match detection to existing track
        max_frames_lost: int = 10,           # How many frames to keep a "lost" track
        new_event_min_frames: int = 1,       # Frames before reporting as "new"
        proximity_threshold: float = 200.0,  # Pixel distance for proximity queries
    ):
        self.iou_threshold = iou_threshold
        self.max_frames_lost = max_frames_lost
        self.new_event_min_frames = new_event_min_frames
        self.proximity_threshold = proximity_threshold

        self.tracks: Dict[int, TrackedObject] = {}
        self.next_track_id = 0
        self.frame_detections: Dict[int, List[Detection]] = defaultdict(list)
        self.reported_ids: Set[int] = set()

    def update(self, frame_idx: int, detections: List[Detection]) -> List[Detection]:
        """
        Update tracker with new frame detections.
        Returns detections with assigned track IDs.
        """
        # Try to match each detection to existing tracks
        unmatched_detections = []
        matched_track_ids = set()

        for det in detections:
            best_track_id = None
            best_iou = self.iou_threshold

            for track_id, track in self.tracks.items():
                if track.label != det.label:
                    continue
                if track_id in matched_track_ids:
                    continue
                overlap = iou(det.box, track.last_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                track = self.tracks[best_track_id]
                track.last_box = det.box
                track.last_frame = frame_idx
                track.confidence = det.score
                track.history.append(det)
                det.track_id = best_track_id
                matched_track_ids.add(best_track_id)
            else:
                unmatched_detections.append(det)

        # Create new tracks for unmatched detections
        for det in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = TrackedObject(
                track_id=track_id,
                label=det.label,
                last_box=det.box,
                last_frame=frame_idx,
                first_frame=frame_idx,
                confidence=det.score,
                history=[det],
            )
            det.track_id = track_id

        # Remove stale tracks
        stale_ids = [
            tid for tid, t in self.tracks.items()
            if frame_idx - t.last_frame > self.max_frames_lost
        ]
        for tid in stale_ids:
            del self.tracks[tid]

        self.frame_detections[frame_idx] = detections
        return detections

    def get_new_events(self, frame_idx: int, labels: Optional[List[str]] = None) -> List[Dict]:
        """
        Return list of NEW events (objects appearing for the first time).
        Pass labels=["fire","smoke"] to filter by category.
        Avoids returning the same event twice.
        """
        new_events = []
        for track_id, track in self.tracks.items():
            if track.reported:
                continue
            if labels and track.label not in labels:
                continue
            if track.first_frame == frame_idx:
                new_events.append({
                    "event": "new_detection",
                    "track_id": track_id,
                    "label": track.label,
                    "box": track.last_box,
                    "frame": frame_idx,
                    "confidence": track.confidence,
                })
                track.reported = True
                self.reported_ids.add(track_id)

        return new_events

    def get_objects_near(
        self,
        target_label: str,
        query_label: str,
        frame_idx: int,
        proximity: Optional[float] = None,
    ) -> List[Dict]:
        """
        Find all `query_label` objects that are near a `target_label` object.

        Example: get_objects_near("car", "smoke", frame_idx=5)
        -> Returns smoke objects near each car.
        """
        if proximity is None:
            proximity = self.proximity_threshold

        frame_dets = self.frame_detections.get(frame_idx, [])
        targets = [d for d in frame_dets if d.label == target_label]
        queries = [d for d in frame_dets if d.label == query_label]

        results = []
        for target in targets:
            nearby = []
            for q in queries:
                dist = distance_between_boxes(target.box, q.box)
                if dist <= proximity:
                    nearby.append({
                        "box": q.box,
                        "label": q.label,
                        "distance_px": dist,
                        "track_id": q.track_id,
                    })
            if nearby:
                results.append({
                    "target": {"box": target.box, "label": target_label, "track_id": target.track_id},
                    "nearby": nearby,
                    "frame": frame_idx,
                })

        return results

    def compare_with_previous(
        self,
        frame_idx: int,
        lookback_frames: int = 5,
        labels: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Compare current frame to frames from `lookback_frames` ago.
        Returns detections that are NEW (not present in previous frames).
        """
        if frame_idx < lookback_frames:
            return []

        prev_frame_idx = frame_idx - lookback_frames
        prev_dets = self.frame_detections.get(prev_frame_idx, [])
        curr_dets = self.frame_detections.get(frame_idx, [])

        if labels:
            prev_dets = [d for d in prev_dets if d.label in labels]
            curr_dets = [d for d in curr_dets if d.label in labels]

        new_detections = []
        for curr in curr_dets:
            is_new = True
            for prev in prev_dets:
                if prev.label == curr.label and iou(curr.box, prev.box) > self.iou_threshold:
                    is_new = False
                    break
            if is_new:
                new_detections.append({
                    "event": "appeared_since_frame",
                    "compared_frame": prev_frame_idx,
                    "current_frame": frame_idx,
                    "label": curr.label,
                    "box": curr.box,
                    "track_id": curr.track_id,
                })

        return new_detections

    def get_track_summary(self) -> List[Dict]:
        """Return summary of all active tracks."""
        return [
            {
                "track_id": t.track_id,
                "label": t.label,
                "first_frame": t.first_frame,
                "last_frame": t.last_frame,
                "duration_frames": t.last_frame - t.first_frame + 1,
                "reported": t.reported,
                "last_box": t.last_box,
            }
            for t in self.tracks.values()
        ]
