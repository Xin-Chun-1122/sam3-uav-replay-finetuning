#!/usr/bin/env python3
"""Normalize frozen RefDrone Layer-3 expressions into short noun phrases."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CANONICAL = (
    (r"\bpedestrians?\b", "person"),
    (r"\bpeople\b", "person"),
    (r"\bcars\b", "car"),
    (r"\bvehicles\b", "vehicle"),
    (r"\btrucks\b", "truck"),
    (r"\bbuses\b", "bus"),
    (r"\btrams\b", "bus"),
    (r"\bmotorcycles?\b", "motorcycle"),
    (r"\bmotorbikes?\b", "motorcycle"),
    (r"\bmotors\b", "motorcycle"),
)


def normalize(text: str) -> str:
    phrase = text.strip().rstrip(".").lower()
    phrase = re.sub(r"^the\s+", "", phrase)
    for pattern, replacement in CANONICAL:
        phrase = re.sub(pattern, replacement, phrase)
    phrase = re.sub(
        r"\b(?:is|are)\s+(?:located|positioned|present|visible|stationed)\b",
        "",
        phrase,
    )
    phrase = re.sub(r"\b(?:is|are)\b", "", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--locked-config", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    for row in rows:
        layer3 = row["layers"][2]
        if int(layer3["layer"]) != 3:
            raise ValueError(f"{row['example_id']} has invalid layer order")
        layer3["source_expression"] = row["source_expression"]
        layer3["prompt"] = normalize(row["source_expression"])
        layer3["prompt_form"] = "normalized short noun phrase"
    args.manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    locked = json.loads(args.locked_config.read_text(encoding="utf-8"))
    locked["layer3_prompt_form"] = (
        "lower-case short noun phrase normalized from the original RefDrone expression; "
        "articles, copulas, and sentence-final punctuation removed; class nouns canonicalized"
    )
    args.locked_config.write_text(json.dumps(locked, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(row["example_id"], "=>", row["layers"][2]["prompt"])


if __name__ == "__main__":
    main()
