"""Run ruff and mypy as pytest tests so they fail the suite."""

import subprocess
import sys

PROJECT_ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)


def test_ruff():
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "grape/", "tests/"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, f"ruff:\n{result.stdout}\n{result.stderr}"


def test_mypy():
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "grape/"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, f"mypy:\n{result.stdout}\n{result.stderr}"
