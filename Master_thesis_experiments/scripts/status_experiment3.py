#!/usr/bin/env python3
"""Print resumable Experiment 3 inference progress."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/alien/sam3/Master_thesis_experiments")
TOTAL = 1595

for model in ("sam3_original", "sam3_adapted_with_replay"):
    path = ROOT / "predictions" / "experiment3" / f"{model}.jsonl"
    complete = 0
    predictions = 0
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                complete += row.get("type") == "image_complete"
                predictions += row.get("type") == "prediction"
    print(
        f"{model}: images={complete}/{TOTAL} ({complete / TOTAL:.1%}), "
        f"prompt_records={predictions}, output={path}"
    )
