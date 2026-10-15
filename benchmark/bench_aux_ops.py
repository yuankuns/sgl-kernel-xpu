"""Compare memory kernels at Muse Spark B8 shapes."""

import argparse
import json
import tempfile
from pathlib import Path
from statistics import median

import torch

from bench_memory_compare import MEMORY_SETUP_PARSER, main_memory_ops


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
    torch.xpu.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__, parents=[MEMORY_SETUP_PARSER])
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--only",
        choices=("all", "alloc_decode", "alloc_extend", "write_req_to_token_pool"),
        default="all",
    )
    args = parser.parse_args()
    alloc_extend_kernel, alloc_decode_kernel, write_req_to_token_pool_triton = main_memory_ops()

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
        lambda: torch.ops.sgl_kernel.paged_alloc_decode(seq, last, free_pages, new_out, 64),
        "alloc_decode_kernel",
        "PagedAllocDecodeKernel",
        lambda: torch.testing.assert_close(new_out.cpu(), old_out.cpu(), rtol=0, atol=0),
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
        lambda: torch.ops.sgl_kernel.paged_alloc_extend(prefix, lengths, extend_last, free_pages, new_out, 64),
        "alloc_extend_kernel",
        "PagedAllocExtendKernel",
        lambda: torch.testing.assert_close(new_out.cpu(), old_out.cpu(), rtol=0, atol=0),
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
        lambda: torch.ops.sgl_kernel.write_req_to_token_pool(
            new_pool,
            req_indices,
            prefix_pointers,
            prefix,
            lengths,
            lengths,
            out_cache_loc,
        ),
        "write_req_to_token_pool_triton",
        "WriteReqToTokenPoolKernel",
        lambda: torch.testing.assert_close(new_pool.cpu(), old_pool.cpu(), rtol=0, atol=0),
        args.samples,
        args.only,
    )


if __name__ == "__main__":
    main()
