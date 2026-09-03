#!/usr/bin/env python3
"""
Experiment Logger - Unified logging for training runs.
Writes to file + console + optional TensorBoard.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    """
    Unified logger for training experiments.
    Outputs: console, file, JSON metrics log, optional TensorBoard.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "experiment",
        use_tensorboard: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.metrics_history: Dict[str, list] = {}
        self.start_time = time.time()

        # File logger
        log_file = self.log_dir / f"{experiment_name}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(str(log_file)),
            ],
        )
        self.logger = logging.getLogger(experiment_name)

        # Metrics JSON file
        self.metrics_file = self.log_dir / "metrics.json"

        # TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / "tensorboard"
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
                self.info(f"TensorBoard logging at: {tb_dir}")
            except ImportError:
                self.logger.warning("TensorBoard not available")

        self.info(f"Experiment: {experiment_name}")
        self.info(f"Log dir: {self.log_dir}")

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log a dict of metrics at a given step."""
        for key, val in metrics.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if full_key not in self.metrics_history:
                self.metrics_history[full_key] = []
            self.metrics_history[full_key].append({"step": step, "value": val})

            if self.tb_writer:
                self.tb_writer.add_scalar(full_key, val, step)

        # Save to JSON
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        # Log to console
        metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.info(f"[Step {step:6d}] {metrics_str}")

    def log_epoch(self, epoch: int, train_loss: float, val_metrics: Optional[Dict] = None):
        """Convenience method to log an epoch summary."""
        elapsed = time.time() - self.start_time
        self.info(f"\n{'='*60}")
        self.info(f"Epoch {epoch} | Loss: {train_loss:.4f} | Time: {elapsed/3600:.2f}h")
        if val_metrics:
            metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            self.info(f"  Val: {metrics_str}")
        self.info(f"{'='*60}\n")
        self.log_metrics({"train/loss": train_loss}, step=epoch, prefix="")
        if val_metrics:
            self.log_metrics(val_metrics, step=epoch, prefix="val")

    def close(self):
        if self.tb_writer:
            self.tb_writer.close()
        self.info("Logger closed.")
