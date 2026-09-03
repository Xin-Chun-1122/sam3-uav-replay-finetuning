#!/usr/bin/env python3
"""
Disk Monitor Utility.
Monitors /work disk usage during training and warns before quota is hit.
"""

import os
import shutil
import time
from pathlib import Path


def get_disk_usage(path: str) -> dict:
    """Return disk usage stats for the given path."""
    total, used, free = shutil.disk_usage(path)
    return {
        "total_gb": total / 1e9,
        "used_gb":  used  / 1e9,
        "free_gb":  free  / 1e9,
        "used_pct": used  / total * 100,
    }


def monitor_disk(
    watch_path: str = "/work/nthujerry123/sam3",
    warn_threshold_pct: float = 90.0,
    critical_threshold_pct: float = 95.0,
    interval_sec: int = 300,
    log_file: str = "/work/nthujerry123/sam3/sam3_v2/experiments/stage2_uav_multidomain/logs/disk_monitor.log",
):
    """Background disk monitor daemon."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[DiskMonitor] Watching {watch_path} every {interval_sec}s")

    while True:
        stats = get_disk_usage(watch_path)
        pct   = stats["used_pct"]
        msg   = (f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                 f"Used: {stats['used_gb']:.1f}GB / {stats['total_gb']:.1f}GB "
                 f"({pct:.1f}%) | Free: {stats['free_gb']:.1f}GB")

        if pct >= critical_threshold_pct:
            msg = "🚨 CRITICAL " + msg
        elif pct >= warn_threshold_pct:
            msg = "⚠️  WARNING  " + msg

        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

        time.sleep(interval_sec)


if __name__ == "__main__":
    monitor_disk()
