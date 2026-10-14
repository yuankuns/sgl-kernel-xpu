"""Compare Muse Spark memory kernels at B8 shapes."""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from statistics import median

import sgl_kernel  # noqa: F401
import torch


def measure(label, run_triton, run_sycl, old_name, new_name, check, samples, only):
    if only != "all" and only != label:
        return
    for _ in range(10):
        run_triton()
        run_sycl()
    torch.xpu.synchronize()
    check()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profiler:
        for _ in range(samples):
            run_triton()
        for _ in range(samples):
            run_sycl()
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
    old = [
        event["dur"]
        for event in events
        if event.get("cat") == "kernel" and old_name in event["name"]
    ]
    new = [
        event["dur"]
        for event in events
        if event.get("cat") == "kernel" and new_name in event["name"]
    ]
    assert len(old) == len(new) == samples, (label, len(old), len(new))
    old_us, new_us = median(old), median(new)
    print(
        f"{label}: Triton={old_us:.3f} us SYCL={new_us:.3f} us speedup={old_us / new_us:.3f}x",
        flush=True,
    )
    if label in ("write_req_to_token_pool", "alloc_extend"):
        print(
            f"{label} samples: Triton_min={min(old):.3f} p25={sorted(old)[len(old)//4]:.3f} p75={sorted(old)[3*len(old)//4]:.3f} max={max(old):.3f}; SYCL_min={min(new):.3f} p25={sorted(new)[len(new)//4]:.3f} p75={sorted(new)[3*len(new)//4]:.3f} max={max(new):.3f}",
            flush=True,
        )
    # Let the card settle before changing to the next production shape.
    torch.xpu.synchronize()
    time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-batch", type=int, default=8)
    parser.add_argument("--isl", type=int, default=4096)
    parser.add_argument("--osl", type=int, default=16)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--only",
        choices=("all", "alloc_decode", "alloc_extend", "write_req_to_token_pool"),
        default="all",
    )
    args = parser.parse_args()
    if (args.serving_batch, args.isl, args.osl, args.tp, args.ep) != (
        8,
        4096,
        16,
        1,
        1,
    ) or os.environ.get("BATCHES", "8") != "8":
        parser.error("Muse-Spark serving scope requires B=8, ISL=4096, OSL=16, TP1/EP1")
    from sglang.kernels.ops.memory.allocator import (
        alloc_decode,
        alloc_decode_kernel,
        alloc_extend,
        alloc_extend_kernel,
    )
    from sglang.kernels.ops.memory.common import write_req_to_token_pool_triton

    device = "xpu"

    seq = torch.tensor(
        [4096, 4097, 4098, 4104, 4112, 4096, 4100, 4111],
        dtype=torch.int64,
        device=device,
    )
    last = torch.arange(8, dtype=torch.int32, device=device) + 4095
    free_pages = torch.arange(1, 6001, dtype=torch.int64, device=device)
    old_out = torch.empty(8, dtype=torch.int64, device=device)
    new_out = torch.empty_like(old_out)
    measure(
        "alloc_decode",
        lambda: alloc_decode_kernel[(8,)](seq, last, free_pages, old_out, 8, 64),
        lambda: alloc_decode(seq, last, free_pages, new_out, 8, 64),
        "alloc_decode_kernel",
        "MuseAllocDecodeKernel",
        lambda: torch.testing.assert_close(new_out, old_out),
        args.samples,
        args.only,
    )

    prefix = torch.zeros(7, dtype=torch.int64, device=device)
    lengths = torch.full((7,), 4096, dtype=torch.int64, device=device)
    extend_last = torch.full((7,), -1, dtype=torch.int32, device=device)
    old_out = torch.empty(28672, dtype=torch.int64, device=device)
    new_out = torch.empty_like(old_out)
    measure(
        "alloc_extend",
        lambda: alloc_extend_kernel[(7,)](
            prefix, lengths, extend_last, free_pages, old_out, 8, 64
        ),
        lambda: alloc_extend(prefix, lengths, extend_last, free_pages, new_out, 8, 64),
        "alloc_extend_kernel",
        "MuseAllocExtendKernel",
        lambda: torch.testing.assert_close(new_out, old_out),
        args.samples,
        args.only,
    )

    old_pool = torch.empty((7, 4096), dtype=torch.int32, device=device)
    new_pool = torch.empty_like(old_pool)
    req_indices = torch.arange(7, dtype=torch.int64, device=device)
    dummy_prefix = torch.zeros(1, dtype=torch.int64, device=device)
    prefix_pointers = torch.tensor(
        [dummy_prefix.data_ptr()] * 7, dtype=torch.uint64, device=device
    )
    out_cache_loc = torch.arange(28672, dtype=torch.int64, device=device)
    measure(
        "write_req_to_token_pool",
        lambda: write_req_to_token_pool_triton[(7,)](
            old_pool,
            req_indices,
            prefix_pointers,
            prefix,
            lengths,
            lengths,
            out_cache_loc,
            old_pool.stride(0),
        ),
        lambda: torch.ops.sgl_kernel.muse_write_req_to_token_pool(
            new_pool,
            req_indices,
            prefix_pointers,
            prefix,
            lengths,
            lengths,
            out_cache_loc,
        ),
        "write_req_to_token_pool_triton",
        "MuseWriteReqToTokenPoolKernel",
        lambda: torch.testing.assert_close(new_pool, old_pool),
        args.samples,
        args.only,
    )


if __name__ == "__main__":
    main()
