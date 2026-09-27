"""
Unit tests for Turn 7 validation and safety mechanisms.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_train_gpu_model_safety_prevents_unconfirmed_full_training() -> None:
    """Verify that train_gpu_model.py safely aborts if full training is requested without confirmation flag."""
    script_path = str(PROJECT_ROOT / "scripts" / "train_gpu_model.py")
    cmd = [sys.executable, script_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "SAFETY GUARD TRIGGERED" in res.stdout or "SAFETY GUARD TRIGGERED" in res.stderr


def test_turn7_cli_help() -> None:
    """Verify CLI interface of validation and benchmark scripts."""
    for script_name in ["run_turn7_validation.py", "benchmark_gpu_pipeline.py", "train_gpu_model.py"]:
        script_path = str(PROJECT_ROOT / "scripts" / script_name)
        res = subprocess.run([sys.executable, script_path, "--help"], capture_output=True, text=True)
        assert res.returncode == 0
        assert "usage:" in res.stdout.lower()
