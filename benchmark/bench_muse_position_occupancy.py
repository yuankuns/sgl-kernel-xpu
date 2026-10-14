"""Measure the model's position shape with direct Triton and registered SYCL kernels."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from statistics import median

import sgl_kernel  # noqa: F401
import torch

from sglang.kernels.ops.attention.position import (
    compute_position_kernel,
    compute_position_triton,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-batch", type=int, default=8)
    parser.add_argument("--isl", type=int, default=4096)
    parser.add_argument("--osl", type=int, default=16)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--active-requests", type=int, choices=[7, 8], default=8)
    parser.add_argument(
        "--max-seq-len-hint", type=int, choices=[-1, 4096], default=4096
    )
    parser.add_argument("--samples", type=int, default=31)
    args = parser.parse_args()
    if (args.serving_batch, args.isl, args.osl, args.tp, args.ep) != (
        8,
        4096,
        16,
        1,
        1,
    ) or os.environ.get("BATCHES", "8") != "8":
        parser.error("Muse-Spark serving scope requires B=8, ISL=4096, OSL=16, TP1/EP1")

    active = args.active_requests
    tokens = active * args.isl
    prefix = torch.zeros(active, device="xpu", dtype=torch.int64)
    lengths = torch.full((active,), args.isl, device="xpu", dtype=torch.int64)
    old_pos = torch.empty(tokens, device="xpu", dtype=torch.int64)
    old_start = torch.empty(active, device="xpu", dtype=torch.int32)

    def old():
        compute_position_kernel[(active,)](old_pos, old_start, prefix, lengths, True)

    def new():
        return compute_position_triton(
            prefix, lengths, tokens, max_seq_len_hint=args.max_seq_len_hint
        )

    old()
    new_pos, new_start = new()
    torch.xpu.synchronize()
    torch.testing.assert_close(new_pos, old_pos)
    torch.testing.assert_close(new_start, old_start)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profile:
        for i in range(args.samples):
            old()
            new()
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as dirname:
        path = Path(dirname) / "trace.json"
        profile.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
    old_us = [
        e["dur"]
        for e in events
        if e.get("cat") == "kernel" and "compute_position_kernel" in e["name"]
    ]
    new_us = [
        e["dur"]
        for e in events
        if e.get("cat") == "kernel" and "MuseComputePositionKernel" in e["name"]
    ]
    assert len(old_us) == len(new_us) == args.samples, (len(old_us), len(new_us))
    old_median, new_median = median(old_us), median(new_us)
    print(
        f"position B8 wave{active} T{tokens} hint={args.max_seq_len_hint}: "
        f"Triton={old_median:.3f} us SYCL={new_median:.3f} us "
        f"speedup={old_median / new_median:.3f}x"
    )


if __name__ == "__main__":
    main()
