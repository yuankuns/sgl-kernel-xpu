import ast
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    bench = Path("/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-residual-norm-xe35/benchmark/bench_fused_norm_residual.py")
    for path in (bench, root / "wait_gpu_barrier.py", root / "review_scope.py"):
        ast.parse(path.read_text())
    commands = [[sys.executable, str(bench)]]
    commands += [["bash", str(root / name)] for name in ("run_diagnostic_gpu7_xe35.sh", "run_validation_xe35.sh", "run_long_graph_gpu7_xe35.sh")]
    invalid = {name: "7" for name in ("BATCHES", "BATCH", "SERVING_BATCH", "BATCH_SIZE",
               "SGLANG_BATCH_SIZE", "SGLANG_MUSE_BATCH_SIZE", "MUSE_BATCH_SIZE", "SERVING_BATCH_SIZE", "MUSE_BS", "BS")}
    invalid.update(ISL="4095", OSL="15", TP="2", EP="2")
    for command in commands:
        for name, value in invalid.items():
            result = subprocess.run(command, env={**os.environ, name: value}, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
            assert result.returncode == 2, (command, name, result.returncode, result.stdout)
            assert "No module named" not in result.stdout
            print(f"PASS: {Path(command[1]).name} {name}={value} rejected before GPU/barrier/import (exit2).")
    for option, value in (("--batch-size", "7"), ("--isl", "4095"), ("--osl", "15"), ("--tp", "2"), ("--ep", "2")):
        result = subprocess.run(commands[0] + [option, value], capture_output=True, text=True, timeout=10)
        assert result.returncode == 2
        print(f"PASS: benchmark CLI {option}={value} rejected before GPU/import (exit2).")


if __name__ == "__main__":
    main()
