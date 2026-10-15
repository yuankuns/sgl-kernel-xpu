import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("batch", [1, 7, 9])
def test_memory_benchmark_rejects_serving_batch_override(batch):
    benchmark = (
        Path(__file__).resolve().parents[1] / "benchmark" / "bench_memory_compare.py"
    )
    command = [sys.executable, "-S", str(benchmark), "--batch", str(batch)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "Muse-Spark serving benchmarks require --batch 8" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
