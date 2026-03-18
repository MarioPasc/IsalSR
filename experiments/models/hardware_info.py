"""Hardware and environment information capture.

Collects CPU, RAM, Python version, OS, conda env, git hash, and timestamp
for experiment metadata.json.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any


def collect_hardware_info() -> dict[str, Any]:
    """Collect hardware and environment information."""
    return {
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "os_version": platform.version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "git_hash": _git_hash(),
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }


def _cpu_model() -> str:
    """Get CPU model string."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":")[1].strip()
    except (FileNotFoundError, PermissionError):
        pass
    return platform.processor() or "unknown"


def _ram_gb() -> float:
    """Get total RAM in GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 1)
    except (FileNotFoundError, PermissionError):
        pass
    return 0.0


def _git_hash() -> str:
    """Get current git commit hash."""
    try:
        return (
            subprocess.check_output(  # noqa: S603, S607
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
