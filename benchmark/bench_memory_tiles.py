"""Measure 1, 4 and 16 workgroups per request at the same XPU shapes.

Requires historical tile-tuning entry points absent from the production shim.
"""

import argparse
import json
import tempfile
from pathlib import Path
from statistics import median

import torch
import triton

from bench_memory_compare import MEMORY_SETUP_PARSER, main_memory_ops, quantile


def measure(batch, prefix_len, extend_len, page_size, samples, fused):
    alloc_extend_kernel, _, write_req_to_token_pool_triton = main_memory_ops()
    prefix = torch.full((batch,), prefix_len, dtype=torch.int64, device="xpu")
    seq = torch.full((batch,), prefix_len + extend_len, dtype=torch.int64, device="xpu")
    extensions = torch.full((batch,), extend_len, dtype=torch.int64, device="xpu")
    last = torch.full((batch,), prefix_len - 1, dtype=torch.int32, device="xpu")
    free_pages = torch.arange(
        1, batch * ((extend_len + page_size - 1) // page_size) + 2, dtype=torch.int64, device="xpu"
    )
    expected = torch.empty(batch * extend_len, dtype=torch.int64, device="xpu")
    alloc_extend_kernel[(batch,)](
        prefix, seq, last, free_pages, expected, triton.next_power_of_2(batch), page_size
    )
    outputs = {tiles: torch.empty_like(expected) for tiles in (1, 4, 16)}
    if fused:
        req_indices = torch.arange(batch, dtype=torch.int64, device="xpu")
        prefix_values = torch.arange(max(1, batch * prefix_len), dtype=torch.int64, device="xpu")
        pointers = torch.tensor(
            [prefix_values.data_ptr() + index * prefix_len * 8 for index in range(batch)],
            dtype=torch.uint64, device="xpu",
        )
        expected_pool = torch.empty((batch, prefix_len + extend_len), dtype=torch.int32, device="xpu")
        write_req_to_token_pool_triton[(batch,)](
            expected_pool, req_indices, pointers, prefix, seq, extensions, expected, expected_pool.stride(0)
        )
        pools = {tiles: torch.empty_like(expected_pool) for tiles in (1, 4, 16)}

        def run(tiles):
            torch.ops.sgl_kernel.paged_alloc_extend_and_write_tiled_for_benchmark(
                prefix, seq, last, free_pages, outputs[tiles], pools[tiles],
                req_indices, pointers, page_size, tiles
            )

        needle = "PagedAllocExtendAndWriteKernel"
    else:

        def run(tiles):
            torch.ops.sgl_kernel.paged_alloc_extend_tiled_for_benchmark(
                prefix, seq, last, free_pages, outputs[tiles], page_size, tiles
            )

        needle = "PagedAllocExtendKernel"
    for _ in range(5):
        for tiles in (1, 4, 16):
            run(tiles)
    torch.xpu.synchronize()
    for tiles in (1, 4, 16):
        torch.testing.assert_close(outputs[tiles].cpu(), expected.cpu(), rtol=0, atol=0)
        if fused:
            torch.testing.assert_close(pools[tiles].cpu(), expected_pool.cpu(), rtol=0, atol=0)
    order = []
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU]
    ) as profiler:
        for iteration in range(samples):
            choices = (1, 4, 16)
            choices = choices[iteration % 3:] + choices[:iteration % 3]
            for tiles in choices:
                run(tiles)
                order.append(tiles)
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
    kernels = sorted(
        (
            event for event in events
            if event.get("cat") == "kernel" and needle in event.get("name", "")
        ),
        key=lambda event: event["ts"],
    )
    if len(kernels) != len(order):
        raise RuntimeError(f"{needle}: expected {len(order)} device events, got {len(kernels)}")
    durations = {tiles: [] for tiles in (1, 4, 16)}
    for tiles, event in zip(order, kernels):
        durations[tiles].append(event["dur"])
    report = {
        "case": "fused" if fused else "extend",
        "batch": batch,
        "prefix_len": prefix_len,
        "extend_len": extend_len,
        "page_size": page_size,
        "last_dtype": "int32",
        "samples": samples,
        "timing": "torch.profiler XPU kernel event duration; rotated launch order",
        "tiles": {
            tiles: {
                "samples_us": values,
                "p20": quantile(values, 0.2),
                "p50": median(values),
                "p80": quantile(values, 0.8),
                "min": min(values),
                "max": max(values),
            }
            for tiles, values in durations.items()
        },
    }
    print(json.dumps(report), flush=True)
    torch.xpu.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__, parents=[MEMORY_SETUP_PARSER])
    parser.add_argument("--samples", type=int, default=51)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=(0, 4096))
    parser.add_argument(
        "--lengths", type=int, nargs="+",
        default=(64, 128, 256, 384, 511, 512, 513, 768, 1024, 1536, 2047, 2048, 2049, 3072, 4096)
    )
    parser.add_argument("--cases", nargs="+", choices=("extend", "fused"), default=("extend", "fused"))
    args = parser.parse_args()
    if not args.test_library:
        parser.error("--test-library must name a tuning shim")
    if (args.samples < 1 or args.batch < 1 or args.page_size < 1 or
            any(length < 1 for length in args.lengths) or any(prefix < 0 for prefix in args.prefix_lengths)):
        parser.error("samples, batch, page size and lengths must be positive")
    print(json.dumps({"torch": torch.__version__, "device": torch.xpu.get_device_name(0)}), flush=True)
    for prefix in args.prefix_lengths:
        for length in args.lengths:
            for fused in (False, True):
                if ("fused" if fused else "extend") in args.cases:
                    measure(args.batch, prefix, length, args.page_size, args.samples, fused)


if __name__ == "__main__":
    main()
