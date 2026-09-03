#!/usr/bin/env python3
"""Check datasets, checkpoints, runtime, and experiment invariants before inference."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def directory_status(path_value: str | None, children: tuple[str, ...]) -> dict:
    if not path_value:
        return {"path": None, "exists": False, "ready": False, "reason": "path_not_configured"}
    path = Path(path_value).expanduser()
    exists = path.is_dir()
    missing = [name for name in children if not (path / name).exists()] if exists else list(children)
    return {"path": str(path), "exists": exists, "ready": exists and not missing, "missing_children": missing}


def file_status(path_value: str | None) -> dict:
    if not path_value:
        return {"path": None, "exists": False, "ready": False, "reason": "path_not_configured"}
    path = Path(path_value).expanduser()
    return {"path": str(path), "exists": path.is_file(), "ready": path.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--output", type=Path, default=Path("results/preflight.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    models = config["models"]

    report = {
        "datasets": {
            "visdrone_det_test": directory_status(paths.get("visdrone_det_test"), ("images", "annotations")),
            "visdrone_mot_test": directory_status(paths.get("visdrone_mot_test"), ("sequences", "annotations")),
            "refdrone_test": directory_status(paths.get("refdrone_test"), ("RefDrone_test_mdetr.json", "all_image")),
        },
        "models": {name: file_status(path) for name, path in models.items()},
        "python_packages": {
            name: importlib.util.find_spec(name) is not None
            for name in ("torch", "cv2", "ultralytics", "numpy", "scipy", "pycocotools", "decord", "clip")
        },
        "invariants": {
            "track_id_source_required": config["experiment2"].get("track_id_source_required"),
            "track_id_source_valid": config["experiment2"].get("track_id_source_required") == "sam3_native",
        },
        "blocking_items": [],
    }
    for dataset, status in report["datasets"].items():
        if not status["ready"]:
            report["blocking_items"].append(f"{dataset}:{status.get('reason', 'missing_or_invalid_structure')}")
    required_models = ("yolo", "yolo_world", "sam3_original", "sam3_adapted_no_replay", "sam3_adapted_with_replay")
    for model in required_models:
        status = report["models"][model]
        if not status["ready"]:
            report["blocking_items"].append(f"{model}:{status.get('reason', 'file_missing')}")
    if not report["invariants"]["track_id_source_valid"]:
        report["blocking_items"].append("experiment2:track_id_source_must_be_sam3_native")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
