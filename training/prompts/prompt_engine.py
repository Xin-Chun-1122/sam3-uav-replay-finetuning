#!/usr/bin/env python3
"""
Complex Prompt Engine for SAM3-I UAV Inference.

Supports multi-step reasoning prompts:
1. "Find cars, then check for smoke near each car"
2. "Compare with previous frame and detect new fire"
3. "Report all fires without duplicates across frames"
4. Natural language -> detection pipeline
"""

import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add sam3 to path
sys.path.insert(0, str(Path(__file__).parents[2] / "sam3"))

from .temporal_reasoner import Detection, TemporalReasoner


# ──────────────────────────────────────────────────────────────
# Prompt Intent Classification
# ──────────────────────────────────────────────────────────────

INTENT_PATTERNS = {
    "find_near": [
        r"find\s+(\w+).+near\s+(\w+)",
        r"(\w+)\s+near\s+(\w+)",
        r"(\w+)\s+close\s+to\s+(\w+)",
        r"(\w+)\s+beside\s+(\w+)",
    ],
    "detect_new": [
        r"new\s+(\w+)",
        r"newly\s+appeared\s+(\w+)",
        r"(\w+)\s+that\s+wasn.t\s+there",
        r"compare.+previous.+(\w+)",
    ],
    "no_duplicate": [
        r"avoid\s+duplicate",
        r"without\s+repeat",
        r"unique\s+(\w+)",
        r"track.+avoid",
    ],
    "simple_detect": [
        r"find\s+(\w+)",
        r"detect\s+(\w+)",
        r"locate\s+(\w+)",
        r"show\s+me\s+(\w+)",
        r"where\s+is\s+the\s+(\w+)",
    ],
}

LABEL_SYNONYMS = {
    "fire": ["fire", "flame", "burning", "blaze"],
    "smoke": ["smoke", "haze", "smog", "fumes", "plume"],
    "car": ["car", "vehicle", "automobile", "truck", "van", "bus"],
    "person": ["person", "human", "people", "pedestrian", "man", "woman"],
}


def classify_prompt(prompt: str) -> Tuple[str, List[str]]:
    """
    Classify a natural language prompt into an intent and extract targets.
    Returns: (intent, [target_labels])
    """
    prompt_lower = prompt.lower()

    # Check each intent pattern
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                # Extract matched words and resolve to unified labels
                targets = []
                for group in match.groups():
                    label = resolve_label(group)
                    if label:
                        targets.append(label)
                if targets or intent in ("no_duplicate", "detect_new"):
                    return intent, targets

    # Default: treat whole prompt as a detection query
    return "simple_detect", []


def resolve_label(word: str) -> Optional[str]:
    """Map a natural language word to a unified category label."""
    word = word.lower().strip()
    for label, synonyms in LABEL_SYNONYMS.items():
        if word in synonyms or word == label:
            return label
    return None


# ──────────────────────────────────────────────────────────────
# Prompt Engine
# ──────────────────────────────────────────────────────────────

class PromptEngine:
    """
    Executes complex multi-step prompts against SAM3-I model output.

    The model handles perception (what objects are where).
    This engine handles reasoning (what to do with the detections).
    """

    def __init__(
        self,
        model_infer_fn: Callable,        # fn(image, prompt) -> List[Detection]
        proximity_threshold: float = 200.0,
        lookback_frames: int = 5,
    ):
        self.model_infer_fn = model_infer_fn
        self.reasoner = TemporalReasoner(proximity_threshold=proximity_threshold)
        self.lookback_frames = lookback_frames
        self.frame_idx = 0

    def process_frame(self, image, prompt: str) -> Dict[str, Any]:
        """
        Process one frame with a complex prompt.
        Returns structured result with detections and reasoning output.
        """
        intent, targets = classify_prompt(prompt)

        result = {
            "frame_idx": self.frame_idx,
            "prompt": prompt,
            "intent": intent,
            "targets": targets,
            "raw_detections": [],
            "reasoning_output": [],
        }

        # ── Step 1: Get raw detections from model ──────────────────
        if intent == "find_near" and len(targets) >= 2:
            # Two-stage detection: first target, then second target
            target_label = targets[0]
            query_label = targets[1]

            dets_a = self._detect(image, target_label)
            dets_b = self._detect(image, query_label)
            all_dets = dets_a + dets_b

        elif intent in ("detect_new", "no_duplicate"):
            # Detect relevant labels
            all_dets = []
            for label in (targets if targets else ["fire", "smoke"]):
                all_dets.extend(self._detect(image, label))

        else:
            # Simple detection
            all_dets = self.model_infer_fn(image, prompt)

        result["raw_detections"] = all_dets

        # ── Step 2: Update tracker ─────────────────────────────────
        tracked_dets = self.reasoner.update(self.frame_idx, all_dets)

        # ── Step 3: Apply reasoning based on intent ────────────────
        if intent == "find_near" and len(targets) >= 2:
            nearby_results = self.reasoner.get_objects_near(
                target_label=targets[0],
                query_label=targets[1],
                frame_idx=self.frame_idx,
            )
            result["reasoning_output"] = nearby_results
            result["summary"] = (
                f"Found {len(nearby_results)} {targets[0]}(s) "
                f"with {targets[1]} nearby."
            )

        elif intent == "detect_new":
            new_events = self.reasoner.compare_with_previous(
                frame_idx=self.frame_idx,
                lookback_frames=self.lookback_frames,
                labels=targets if targets else None,
            )
            result["reasoning_output"] = new_events
            result["summary"] = f"Detected {len(new_events)} NEW objects vs {self.lookback_frames} frames ago."

        elif intent == "no_duplicate":
            new_events = self.reasoner.get_new_events(
                frame_idx=self.frame_idx,
                labels=targets if targets else None,
            )
            result["reasoning_output"] = new_events
            result["summary"] = f"{len(new_events)} new unique events (no duplicates)."

        else:
            result["reasoning_output"] = [
                {"box": d.box, "label": d.label, "score": d.score}
                for d in tracked_dets
            ]
            result["summary"] = f"Detected {len(tracked_dets)} objects."

        self.frame_idx += 1
        return result

    def _detect(self, image, label: str) -> List[Detection]:
        """Run model inference for a specific label."""
        raw = self.model_infer_fn(image, label)
        # Ensure we return Detection objects
        if raw and isinstance(raw[0], Detection):
            return raw
        detections = []
        for r in raw:
            detections.append(Detection(
                box=r.get("box", [0, 0, 1, 1]),
                label=label,
                score=r.get("score", 1.0),
                prompt=label,
                frame_idx=self.frame_idx,
            ))
        return detections

    def reset(self):
        """Reset the engine state (start new video)."""
        self.reasoner = TemporalReasoner()
        self.frame_idx = 0
