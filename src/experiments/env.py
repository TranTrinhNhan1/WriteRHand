"""Run metadata: git commit, hardware and library versions (stored with every result)."""

import os
import platform
import subprocess
import sys

import numpy as np

from src.utils.paths import ROOT


def git_info():
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, timeout=10
            ).stdout.strip()
        except Exception:
            return None

    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    return {
        "commit": commit,
        "short": commit[:8] if commit else None,
        "dirty": bool(dirty) if dirty is not None else None,
    }


def _cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def hardware_info():
    import sklearn
    import torch

    return {
        "cpu": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",  # every reported timing is measured on the CPU
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
    }
