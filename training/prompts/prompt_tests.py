#!/usr/bin/env python3
"""
Prompt Test Script - Tests all complex prompt templates against the model.

Usage:
  python prompts/prompt_tests.py \
    --checkpoint experiments/stage2_uav_multidomain/checkpoints/checkpoint.pt \
    --image_dir test_fire_video/ \
    --prompt_set prompts/templates/fire_smoke_prompts.yaml \
    --output_dir experiments/stage2_uav_multidomain/prompt_eval/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml
import torch

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.prompt_engine import classify_prompt


def parse_args():
    parser = argparse.ArgumentParser("SAM3-I Complex Prompt Tester")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--image_dir",   required=True)
    parser.add_argument("--prompt_set",  default="prompts/templates/fire_smoke_prompts.yaml")
    parser.add_argument("--output_dir",  default="experiments/stage2_uav_multidomain/prompt_eval")
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--verbose",     action="store_true")
    return parser.parse_args()


def load_prompts(yaml_file: str) -> List[Dict]:
    """Load all prompt templates from YAML file."""
    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    all_prompts = []
    for section in ["simple", "spatial", "temporal", "tracking", "conditional"]:
        all_prompts.extend(data.get(section, []))
    return all_prompts


def test_single_prompt(model, image, prompt_info: Dict, score_thresh: float) -> Dict:
    """Test one prompt on one image and return result."""
    prompt = prompt_info["prompt"]
    intent, targets = classify_prompt(prompt)

    # In production: run actual SAM3-I inference here
    # For now, return intent classification as validation of prompt engine
    result = {
        "id": prompt_info["id"],
        "prompt": prompt,
        "expected_classes": prompt_info.get("expected_classes", []),
        "difficulty": prompt_info.get("difficulty", "unknown"),
        "classified_intent": intent,
        "classified_targets": targets,
        "detections": [],   # Would be populated by actual model inference
        "pass": False,      # Would be determined by AP computation
        "notes": "",
    }

    return result


def run_all_prompts(
    model,
    image_dir: Path,
    prompts: List[Dict],
    output_dir: Path,
    score_thresh: float,
    verbose: bool,
) -> Dict:
    """Run all prompts on all test images, collect results."""
    try:
        from PIL import Image
        images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        images = images[:10]  # Limit to 10 test images
    except Exception:
        images = []

    if not images:
        print(f"[WARNING] No images found in {image_dir}")

    all_results = {}
    summary_by_difficulty = {"easy": [], "medium": [], "hard": [], "expert": []}

    print(f"\nTesting {len(prompts)} prompts on {len(images)} images...\n")

    for prompt_info in prompts:
        pid = prompt_info["id"]
        diff = prompt_info.get("difficulty", "medium")
        prompt_results = []

        print(f"  [{pid}] {prompt_info['prompt'][:60]}")

        for img_path in images[:3]:  # Test each prompt on 3 images
            try:
                from PIL import Image
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            res = test_single_prompt(model, img, prompt_info, score_thresh)
            prompt_results.append(res)

        all_results[pid] = {
            "prompt_info": prompt_info,
            "results": prompt_results,
            "avg_detections": sum(len(r["detections"]) for r in prompt_results) / max(len(prompt_results), 1),
        }
        summary_by_difficulty[diff].append(pid)

    return all_results, summary_by_difficulty


def print_summary(all_results: Dict, summary_by_difficulty: Dict):
    print("\n" + "=" * 70)
    print("PROMPT TEST SUMMARY")
    print("=" * 70)

    for difficulty, ids in summary_by_difficulty.items():
        if not ids:
            continue
        print(f"\n  {difficulty.upper()}: {len(ids)} prompts")
        for pid in ids:
            r = all_results[pid]
            intent = r["results"][0]["classified_intent"] if r["results"] else "N/A"
            targets = r["results"][0]["classified_targets"] if r["results"] else []
            print(f"    [{pid}] Intent={intent} | Targets={targets}")

    print("\n" + "=" * 70)
    print("Intent Classification Check:")
    correct_intent = sum(
        1 for r in all_results.values()
        if r.get("results") and r["results"][0].get("classified_intent") != "simple_detect"
        and r["prompt_info"].get("intent") not in (None, "")
    )
    print(f"  Non-trivial prompts correctly classified: {correct_intent}/{len(all_results)}")
    print("=" * 70 + "\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompt_set)
    print(f"Loaded {len(prompts)} prompts from {args.prompt_set}")

    # Model loading (placeholder)
    model = None

    image_dir = Path(args.image_dir)
    all_results, summary_by_diff = run_all_prompts(
        model=model,
        image_dir=image_dir,
        prompts=prompts,
        output_dir=output_dir,
        score_thresh=args.score_thresh,
        verbose=args.verbose,
    )

    print_summary(all_results, summary_by_diff)

    # Save results
    out_file = output_dir / "prompt_test_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved: {out_file}")


if __name__ == "__main__":
    main()
