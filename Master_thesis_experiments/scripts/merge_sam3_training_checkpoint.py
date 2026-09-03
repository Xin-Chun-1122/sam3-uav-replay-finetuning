#!/usr/bin/env python3
"""Merge a detector-only SAM 3 training checkpoint into the original full model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def state_dict(obj: dict) -> dict:
    return obj["model"] if "model" in obj else obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--finetuned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # mmap avoids materializing the large optimizer state from a training checkpoint.
    base_obj = torch.load(args.base, map_location="cpu", mmap=True, weights_only=False)
    ft_obj = torch.load(args.finetuned, map_location="cpu", mmap=True, weights_only=False)
    base = state_dict(base_obj)
    finetuned = state_dict(ft_obj)

    merged = 0
    unexpected: list[str] = []
    for key, value in finetuned.items():
        full_key = key if key.startswith(("detector.", "tracker.")) else f"detector.{key}"
        if full_key not in base:
            unexpected.append(full_key)
            continue
        if tuple(base[full_key].shape) != tuple(value.shape):
            raise ValueError(
                f"shape mismatch for {full_key}: base={tuple(base[full_key].shape)}, "
                f"finetuned={tuple(value.shape)}"
            )
        base[full_key] = value
        merged += 1

    if unexpected:
        raise ValueError(f"{len(unexpected)} unexpected finetuned keys; first keys: {unexpected[:10]}")
    if merged != len(finetuned):
        raise ValueError(f"merged {merged} of {len(finetuned)} finetuned tensors")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base_obj, args.output)
    print(f"merged_tensors={merged}")
    print(f"tracker_tensors_preserved={sum(key.startswith('tracker.') for key in base)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()

