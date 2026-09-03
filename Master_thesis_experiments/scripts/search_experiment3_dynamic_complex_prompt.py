#!/usr/bin/env python3
"""Probe difficult noun-phrase prompts on real consecutive UAV frames."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from Master_thesis_experiments.scripts.infer_sam3_visdrone import predict_boxes

ROOT = PROJECT_ROOT / "Master_thesis_experiments"
DATA = ROOT / "datasets/raw/VisDrone2019-MOT-test-dev/sequences/uav0000077_00720_v"
OUT = ROOT / "predictions/experiment3_dynamic_unambiguous_truck_prompt_probe.json"
FRAMES = (414, 421, 428, 435, 442, 449, 456)
PROMPTS = (
    "livestock truck",
    "livestock truck in the center",
    "livestock transport truck",
    "livestock transport truck in the center",
    "cattle transport truck in the center",
    "animal transport truck in the center",
)
MODELS = {
    "Original SAM 3": {
        "checkpoint": Path("/home/alien/.cache/huggingface/hub/models--facebook--sam3/blobs/9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"),
        "threshold": 0.48943960666656494,
    },
    "Ours": {
        "checkpoint": Path("/home/alien/sam3/finetune_sam3/checkpoint_25_merged.pt"),
        "threshold": 0.4773983955383301,
    },
}


def main() -> None:
    result = {
        "dataset": "VisDrone2019-MOT-test-dev",
        "sequence": DATA.name,
        "frames": list(FRAMES),
        "inference": "independent per-frame language-guided detection; no tracker",
        "prompts": list(PROMPTS),
        "models": {},
    }
    for model_name, settings in MODELS.items():
        print(f"loading {model_name}", flush=True)
        model = build_sam3_image_model(
            checkpoint_path=str(settings["checkpoint"]),
            load_from_HF=False,
            enable_segmentation=False,
            device="cuda",
        )
        processor = Sam3Processor(model, device="cuda", confidence_threshold=0.0)
        model_rows = {}
        for frame in FRAMES:
            image = Image.open(DATA / f"{frame:07d}.jpg").convert("RGB")
            state = processor.set_image(image)
            per_prompt = {}
            for prompt in PROMPTS:
                rows = predict_boxes(model, processor, state, prompt, score_floor=0.0)
                per_prompt[prompt] = [
                    row for row in rows if row["score"] >= settings["threshold"]
                ]
            model_rows[str(frame)] = per_prompt
            print(f"{model_name}: frame {frame}", flush=True)
        result["models"][model_name] = {
            "checkpoint": str(settings["checkpoint"]),
            "threshold": settings["threshold"],
            "predictions": model_rows,
        }
        del processor, model
        gc.collect()
        torch.cuda.empty_cache()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
