import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("batch", [1, 7, 9])
def test_position_benchmark_rejects_serving_batch_before_imports(batch):
    benchmark = Path(__file__).resolve().parents[1] / "benchmark" / "bench_position_occupancy.py"
    result = subprocess.run(
        [sys.executable, "-S", str(benchmark), "--batch", str(batch)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "B8/ISL4096/OSL16/TP1/EP1" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_position_benchmark_rejects_environment_batch_before_imports():
    benchmark = Path(__file__).resolve().parents[1] / "benchmark" / "bench_position_occupancy.py"
    result = subprocess.run(
        [sys.executable, "-S", str(benchmark)],
        env={**os.environ, "BATCHES": "7"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "ModuleNotFoundError" not in result.stderr
