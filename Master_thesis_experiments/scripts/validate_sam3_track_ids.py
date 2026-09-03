#!/usr/bin/env python3
"""Reject tracking outputs whose identity did not originate from SAM 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = {"sequence_id", "frame_index", "sam3_obj_id", "track_id_source", "bbox_xyxy", "score"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_jsonl", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    rows = 0
    with args.prediction_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc})")
                continue
            missing = REQUIRED - row.keys()
            if missing:
                errors.append(f"line {line_number}: missing {sorted(missing)}")
            if row.get("track_id_source") != "sam3_native":
                errors.append(f"line {line_number}: track_id_source must equal sam3_native")
            box = row.get("bbox_xyxy")
            if not isinstance(box, list) or len(box) != 4:
                errors.append(f"line {line_number}: bbox_xyxy must have four values")
    if errors:
        print("\n".join(errors[:100]))
        raise SystemExit(f"FAILED: {len(errors)} validation error(s)")
    print(f"PASS: {rows} rows, all track IDs declare SAM 3 native provenance")


if __name__ == "__main__":
    main()

