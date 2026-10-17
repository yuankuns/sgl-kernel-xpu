import ast
import hashlib
from pathlib import Path

import torch


def main():
    benchmark = Path("/root/worktrees/sgl-kernel-xpu-up/pr/muse-residual-norm-xe35/benchmark/bench_fused_norm_residual.py")
    tree = ast.parse(benchmark.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in
                 {"round_bfloat16_xe35", "reference_xe35", "check_accuracy_xe35"}]
    namespace = {"torch": torch, "hashlib": hashlib, "POST_EPS": 1e-8, "PRE_EPS": 1e-5}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(benchmark), "exec"), namespace)
    torch.set_num_threads(2)
    rows, hidden = 1031, 8
    branch = torch.zeros(rows, hidden, dtype=torch.bfloat16)
    stream = torch.ones(rows, hidden, dtype=torch.float32)
    empty = torch.empty(0, dtype=torch.bfloat16)
    pre = torch.ones(hidden, dtype=torch.bfloat16)
    alpha = torch.ones(hidden, dtype=torch.float32)
    beta = torch.zeros(hidden, dtype=torch.float32)
    output = torch.ones_like(branch)
    updated = torch.ones_like(stream)
    inputs = (branch, stream, empty, pre, alpha, beta, False, False)
    accuracy, outputs = namespace["check_accuracy_xe35"](inputs, output, updated, output.float())
    assert accuracy["stream_max_abs"] == 0
    assert accuracy["norm_sha256"] == hashlib.sha256(output.view(torch.int16).numpy().tobytes()).hexdigest()
    assert accuracy["stream_sha256"] == hashlib.sha256(updated.numpy().tobytes()).hexdigest()
    torch.testing.assert_close(outputs["norm"], output, rtol=0, atol=0)
    torch.testing.assert_close(outputs["stream"], updated, rtol=0, atol=0)
    updated[-1, -1] += 0.125
    try:
        namespace["check_accuracy_xe35"](inputs, output, updated, None)
    except AssertionError:
        pass
    else:
        raise AssertionError("last partial chunk stream corruption was not rejected")
    updated[-1, -1] = 1
    widened = output.float()
    widened[512, 0] += 0.125
    try:
        namespace["check_accuracy_xe35"](inputs, output, updated, widened)
    except AssertionError:
        pass
    else:
        raise AssertionError("widening mismatch was not rejected")
    print("PASS: CPU-only 512-row streaming preserves full output bytes/hashes, final partial-chunk checks and exact widening; not GPU/production acceptance.")


if __name__ == "__main__":
    main()
