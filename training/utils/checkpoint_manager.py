#!/usr/bin/env python3
"""
Checkpoint Manager - Advanced checkpoint saving, loading, and lifecycle management.
Handles: auto-cleanup, best model tracking, multi-metric comparison.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CheckpointManager:
    """
    Manages model checkpoints with:
    - Keep only N most recent checkpoints (disk safety)
    - Track best checkpoint per metric
    - Resume-from-latest logic
    - Checkpoint metadata logging
    """

    def __init__(
        self,
        save_dir: str,
        max_keep: int = 2,            # Keep only 2 numbered checkpoints
        best_metric: str = "AP50",    # Metric to track for "best" checkpoint
        higher_is_better: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.best_metric = best_metric
        self.higher_is_better = higher_is_better

        self.metadata_file = self.save_dir / "checkpoint_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {"checkpoints": [], "best": None, "best_score": None}

    def _save_metadata(self):
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Return path to the most recent checkpoint.pt, or None."""
        ckpt = self.save_dir / "checkpoint.pt"
        return ckpt if ckpt.exists() else None

    def get_best_checkpoint(self) -> Optional[Path]:
        """Return path to best checkpoint by tracked metric."""
        best_info = self.metadata.get("best")
        if best_info:
            path = Path(best_info["path"])
            if path.exists():
                return path
        return None

    def on_epoch_saved(self, epoch: int, metrics: Dict[str, float]):
        """
        Call this after every epoch checkpoint is saved.
        Handles cleanup and best-model tracking.
        """
        ckpt_name = f"checkpoint_{epoch}.pt"
        ckpt_path = self.save_dir / ckpt_name

        # Register in metadata
        entry = {"epoch": epoch, "path": str(ckpt_path), "metrics": metrics}
        self.metadata["checkpoints"].append(entry)

        # Check if this is best
        score = metrics.get(self.best_metric)
        if score is not None:
            best_score = self.metadata.get("best_score")
            is_better = (
                best_score is None or
                (self.higher_is_better and score > best_score) or
                (not self.higher_is_better and score < best_score)
            )
            if is_better:
                self.metadata["best_score"] = score
                self.metadata["best"] = entry
                # Copy as best checkpoint
                best_path = self.save_dir / "checkpoint_best.pt"
                if ckpt_path.exists():
                    shutil.copy2(ckpt_path, best_path)
                print(f"[CheckpointManager] New best {self.best_metric}={score:.4f} at epoch {epoch}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        self._save_metadata()

    def _cleanup_old_checkpoints(self):
        """Keep only the most recent max_keep numbered checkpoints."""
        # Find all checkpoint_{N}.pt files
        ckpts = sorted(
            self.save_dir.glob("checkpoint_[0-9]*.pt"),
            key=lambda p: int(p.stem.split("_")[-1])
        )
        # Delete all but the last max_keep
        for old_ckpt in ckpts[:-self.max_keep]:
            old_ckpt.unlink(missing_ok=True)
            print(f"[CheckpointManager] Removed old checkpoint: {old_ckpt.name}")

    def print_summary(self):
        """Print checkpoint history summary."""
        print("\n===== Checkpoint Summary =====")
        for ckpt in self.metadata.get("checkpoints", []):
            metrics_str = " | ".join(f"{k}={v:.3f}" for k, v in ckpt.get("metrics", {}).items())
            print(f"  Epoch {ckpt['epoch']:3d}: {metrics_str}")
        best = self.metadata.get("best")
        if best:
            print(f"\nBest: Epoch {best['epoch']} ({self.best_metric}={self.metadata['best_score']:.4f})")
        print("==============================\n")


def auto_cleanup_daemon(save_dir: str, max_keep: int = 1, interval_sec: int = 600):
    """
    Background process: runs every interval_sec and cleans up old checkpoints.
    Run this with: nohup python -c "from utils.checkpoint_manager import auto_cleanup_daemon; auto_cleanup_daemon('/path')" &
    """
    import time
    print(f"[AutoCleanup] Watching {save_dir} every {interval_sec}s, keeping {max_keep} checkpoints")
    while True:
        mgr = CheckpointManager(save_dir, max_keep=max_keep)
        mgr._cleanup_old_checkpoints()
        time.sleep(interval_sec)
