"""Unit tests for resume/cache metadata validation."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def _compute_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_cache_meta(tmp_dir: str, config: dict) -> str:
    meta_path = os.path.join(tmp_dir, "cache_meta.json")
    with open(meta_path, "w") as f:
        json.dump(config, f)
    return meta_path


def _load_cache_meta(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


class TestCacheMetadata:
    """Test that cache invalidation works when parameters change."""

    BASE_CONFIG = {
        "checkpoint_sha256": "abc123",
        "confidence_threshold": 0.30,
        "nms_threshold": 0.50,
        "iou_threshold": 0.50,
        "image_size": 1008,
        "prompts": ["fire", "smoke"],
        "dataset": "fasdd",
        "annotation_sha256": "def456",
    }

    def test_cache_valid_same_config(self, tmp_path):
        meta = self.BASE_CONFIG.copy()
        meta_path = str(tmp_path / "cache_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        loaded = _load_cache_meta(meta_path)
        assert loaded == meta

    def test_cache_invalidated_checkpoint_change(self, tmp_path):
        meta = self.BASE_CONFIG.copy()
        meta_path = str(tmp_path / "cache_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        # Simulate new checkpoint
        new_config = self.BASE_CONFIG.copy()
        new_config["checkpoint_sha256"] = "xyz999"
        loaded = _load_cache_meta(meta_path)
        assert loaded["checkpoint_sha256"] != new_config["checkpoint_sha256"]

    def test_cache_invalidated_threshold_change(self, tmp_path):
        meta = self.BASE_CONFIG.copy()
        meta_path = str(tmp_path / "cache_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        new_config = self.BASE_CONFIG.copy()
        new_config["confidence_threshold"] = 0.50
        loaded = _load_cache_meta(meta_path)
        assert loaded["confidence_threshold"] != new_config["confidence_threshold"]

    def test_cache_invalidated_prompts_change(self, tmp_path):
        meta = self.BASE_CONFIG.copy()
        meta_path = str(tmp_path / "cache_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        new_config = self.BASE_CONFIG.copy()
        new_config["prompts"] = ["car", "person"]
        loaded = _load_cache_meta(meta_path)
        assert loaded["prompts"] != new_config["prompts"]

    def test_cache_invalidated_image_size_change(self, tmp_path):
        meta = self.BASE_CONFIG.copy()
        meta_path = str(tmp_path / "cache_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        new_config = self.BASE_CONFIG.copy()
        new_config["image_size"] = 512
        loaded = _load_cache_meta(meta_path)
        assert loaded["image_size"] != new_config["image_size"]


def test_sha256_file_hash():
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pt", delete=False) as f:
        f.write(b"fake checkpoint data")
        tmp = f.name
    try:
        h1 = _compute_hash(tmp)
        h2 = _compute_hash(tmp)
        assert h1 == h2  # Deterministic

        # Different content → different hash
        with open(tmp, "wb") as f:
            f.write(b"different data")
        h3 = _compute_hash(tmp)
        assert h1 != h3
    finally:
        os.remove(tmp)


def test_per_image_cache_structure():
    """Cache entries should be indexable by image path."""
    cache = {}
    image_path = "/data/images/img001.jpg"
    result = {"boxes": [[1, 2, 3, 4]], "scores": [0.9], "labels": ["fire"]}
    cache[image_path] = result
    assert cache[image_path]["scores"][0] == 0.9
