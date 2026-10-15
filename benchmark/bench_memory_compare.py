"""Compare SGLang main memory kernels with the SYCL ops on XPU.

The fixed B8/4096 cases match Muse Spark serving. Nearby extend lengths
exercise both sides of the SYCL tile dispatch boundaries.
"""

import argparse
import functools
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from statistics import median

serving_scope_parser = argparse.ArgumentParser(add_help=False)
serving_scope_parser.add_argument("--batch", type=int, default=8)
serving_scope, _ = serving_scope_parser.parse_known_args()
if serving_scope.batch != 8:
    serving_scope_parser.error("Muse-Spark serving benchmarks require --batch 8")

import torch
import triton

MEMORY_SETUP_PARSER = argparse.ArgumentParser(add_help=False)
MEMORY_SETUP_PARSER.add_argument(
    "--test-library", type=Path, help="explicit memory-only test registration library"
)
MEMORY_SETUP_PARSER.add_argument(
    "--memory-source", type=Path, help="SGLang memory kernel source directory"
)
memory_setup, _ = MEMORY_SETUP_PARSER.parse_known_args()

if memory_setup.test_library:
    torch.ops.load_library(str(memory_setup.test_library))
elif "--main-only" not in sys.argv:
    import sgl_kernel  # noqa: F401


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


@functools.lru_cache(maxsize=1)
def main_memory_ops():
    source = memory_setup.memory_source
    if source:
        # Load the exact SGLang kernel files when the test image lacks server
        # dependencies. This changes import setup, not the invoked kernels.
        modules = {}
        for filename in ("allocator.py", "common.py"):
            path = Path(source) / filename
            spec = importlib.util.spec_from_file_location(
                f"sglang_memory_bench_{filename[:-3]}", path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules[filename] = module
        return (
            modules["allocator.py"].alloc_extend_kernel,
            modules["allocator.py"].alloc_decode_kernel,
            modules["common.py"].write_req_to_token_pool_triton,
        )
    from sglang.kernels.ops.memory.allocator import (
        alloc_decode_kernel,
        alloc_extend_kernel,
    )
    from sglang.kernels.ops.memory.common import write_req_to_token_pool_triton

    return alloc_extend_kernel, alloc_decode_kernel, write_req_to_token_pool_triton


def profile_pair(name, shape, old, new, old_kernels, new_kernel, samples):
    for _ in range(5):
        old()
        new()
    torch.xpu.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profiler:
        for iteration in range(samples):
            if iteration % 2:
                new()
                old()
            else:
                old()
                new()
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
    durations = {
        needle: [
            event["dur"]
            for event in events
            if event.get("cat") == "kernel" and needle in event.get("name", "")
        ]
        for needle in (*old_kernels, new_kernel)
    }
    for needle, values in durations.items():
        if len(values) != samples:
            raise RuntimeError(
                f"{name} {shape}: expected {samples} kernels for {needle}, got {len(values)}"
            )
    old_us = [
        sum(durations[needle][index] for needle in old_kernels)
        for index in range(samples)
    ]
    new_us = durations[new_kernel]
    report = {
        "case": name,
        **shape,
        "samples": samples,
        "timing": "torch.profiler XPU kernel event duration; alternating paired calls",
        "main_samples_us": old_us,
        "sycl_samples_us": new_us,
        "main_us": {
            "p20": quantile(old_us, 0.2),
            "p50": median(old_us),
            "p80": quantile(old_us, 0.8),
            "min": min(old_us),
            "max": max(old_us),
        },
        "sycl_us": {
            "p20": quantile(new_us, 0.2),
            "p50": median(new_us),
            "p80": quantile(new_us, 0.8),
            "min": min(new_us),
            "max": max(new_us),
        },
    }
    report["speedup"] = report["main_us"]["p50"] / report["sycl_us"]["p50"]
    print(json.dumps(report), flush=True)
    torch.xpu.synchronize()


def profile_main(name, shape, old, old_kernels, samples):
    for _ in range(5):
        old()
    torch.xpu.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profiler:
        for _ in range(samples):
            old()
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
    durations = {
        needle: [
            event["dur"]
            for event in events
            if event.get("cat") == "kernel" and needle in event.get("name", "")
        ]
        for needle in old_kernels
    }
    for needle, values in durations.items():
        if len(values) != samples:
            raise RuntimeError(
                f"{name} {shape}: expected {samples} kernels for {needle}, got {len(values)}"
            )
    old_us = [
        sum(durations[needle][index] for needle in old_kernels)
        for index in range(samples)
    ]
    print(
        json.dumps(
            {
                "case": name,
                **shape,
                "samples": samples,
                "timing": "torch.profiler XPU kernel event duration",
                "main_samples_us": old_us,
                "main_us": {
                    "p20": quantile(old_us, 0.2),
                    "p50": median(old_us),
                    "p80": quantile(old_us, 0.8),
                    "min": min(old_us),
                    "max": max(old_us),
                },
            }
        ),
        flush=True,
    )
    torch.xpu.synchronize()


def compare_extend(
    batch, prefix_len, extend_len, page_size, samples, fused, main_only=False
):
    alloc_extend_kernel, _, write_req_to_token_pool_triton = main_memory_ops()

    prefix = torch.full((batch,), prefix_len, dtype=torch.int64, device="xpu")
    seq = torch.full((batch,), prefix_len + extend_len, dtype=torch.int64, device="xpu")
    extensions = torch.full((batch,), extend_len, dtype=torch.int64, device="xpu")
    last = torch.full((batch,), prefix_len - 1, dtype=torch.int32, device="xpu")
    free_pages = torch.arange(
        1,
        batch * ((extend_len + page_size - 1) // page_size) + 2,
        dtype=torch.int64,
        device="xpu",
    )
    old_out = torch.empty(batch * extend_len, dtype=torch.int64, device="xpu")
    new_out = torch.empty_like(old_out)
    shape = {
        "batch": batch,
        "prefix_len": prefix_len,
        "extend_len": extend_len,
        "page_size": page_size,
        "last_dtype": "int32",
    }
    if fused:
        old_pool = torch.empty(
            (batch, prefix_len + extend_len), dtype=torch.int32, device="xpu"
        )
        new_pool = torch.empty_like(old_pool)
        req_indices = torch.arange(batch, dtype=torch.int64, device="xpu")
        prefix_values = torch.arange(
            max(1, batch * prefix_len), dtype=torch.int64, device="xpu"
        )
        pointers = torch.tensor(
            [
                prefix_values.data_ptr() + index * prefix_len * 8
                for index in range(batch)
            ],
            dtype=torch.uint64,
            device="xpu",
        )

        def old():
            alloc_extend_kernel[(batch,)](
                prefix,
                seq,
                last,
                free_pages,
                old_out,
                triton.next_power_of_2(batch),
                page_size,
            )
            write_req_to_token_pool_triton[(batch,)](
                old_pool,
                req_indices,
                pointers,
                prefix,
                seq,
                extensions,
                old_out,
                old_pool.stride(0),
            )

        def new():
            torch.ops.sgl_kernel.paged_alloc_extend_and_write(
                prefix,
                seq,
                last,
                free_pages,
                new_out,
                new_pool,
                req_indices,
                pointers,
                page_size,
            )

        if main_only:
            profile_main(
                "fused",
                shape,
                old,
                ("alloc_extend_kernel", "write_req_to_token_pool_triton"),
                samples,
            )
            return
        old()
        new()
        torch.xpu.synchronize()
        torch.testing.assert_close(new_pool.cpu(), old_pool.cpu(), rtol=0, atol=0)
        old_kernels = ("alloc_extend_kernel", "write_req_to_token_pool_triton")
        new_kernel = "PagedAllocExtendAndWriteKernel"
    else:

        def old():
            alloc_extend_kernel[(batch,)](
                prefix,
                seq,
                last,
                free_pages,
                old_out,
                triton.next_power_of_2(batch),
                page_size,
            )

        def new():
            torch.ops.sgl_kernel.paged_alloc_extend(
                prefix, seq, last, free_pages, new_out, page_size
            )

        if main_only:
            profile_main("extend", shape, old, ("alloc_extend_kernel",), samples)
            return
        old()
        new()
        torch.xpu.synchronize()
        old_kernels = ("alloc_extend_kernel",)
        new_kernel = "PagedAllocExtendKernel"
    torch.testing.assert_close(new_out.cpu(), old_out.cpu(), rtol=0, atol=0)
    profile_pair(
        "fused" if fused else "extend",
        shape,
        old,
        new,
        old_kernels,
        new_kernel,
        samples,
    )


def compare_decode(samples, main_only=False):
    _, alloc_decode_kernel, _ = main_memory_ops()

    batch, page_size = 8, 64
    seq = torch.tensor(
        [4096, 4097, 4098, 4104, 4112, 4096, 4100, 4111],
        dtype=torch.int64,
        device="xpu",
    )
    last = torch.arange(batch, dtype=torch.int32, device="xpu") + 4095
    free_pages = torch.arange(1, 16, dtype=torch.int64, device="xpu")
    old_out = torch.empty(batch, dtype=torch.int64, device="xpu")
    new_out = torch.empty_like(old_out)

    def old():
        alloc_decode_kernel[(batch,)](seq, last, free_pages, old_out, batch, page_size)

    def new():
        torch.ops.sgl_kernel.paged_alloc_decode(
            seq, last, free_pages, new_out, page_size
        )

    if main_only:
        profile_main(
            "decode",
            {"batch": batch, "page_size": page_size, "last_dtype": "int32"},
            old,
            ("alloc_decode_kernel",),
            samples,
        )
        return
    old()
    new()
    torch.xpu.synchronize()
    torch.testing.assert_close(new_out.cpu(), old_out.cpu(), rtol=0, atol=0)
    profile_pair(
        "decode",
        {"batch": batch, "page_size": page_size, "last_dtype": "int32"},
        old,
        new,
        ("alloc_decode_kernel",),
        "PagedAllocDecodeKernel",
        samples,
    )


def compare_write(samples, prefix_len, extend_len, main_only=False):
    _, _, write_req_to_token_pool_triton = main_memory_ops()

    batch, seq_len = 8, prefix_len + extend_len
    old_pool = torch.empty((batch, seq_len), dtype=torch.int32, device="xpu")
    new_pool = torch.empty_like(old_pool)
    req_indices = torch.arange(batch, dtype=torch.int64, device="xpu")
    prefix_values = torch.arange(
        max(1, batch * prefix_len), dtype=torch.int64, device="xpu"
    )
    pointers = torch.tensor(
        [prefix_values.data_ptr() + index * prefix_len * 8 for index in range(batch)],
        dtype=torch.uint64,
        device="xpu",
    )
    prefix = torch.full((batch,), prefix_len, dtype=torch.int64, device="xpu")
    seq = torch.full((batch,), seq_len, dtype=torch.int64, device="xpu")
    extensions = torch.full((batch,), extend_len, dtype=torch.int64, device="xpu")
    cache_loc = torch.arange(batch * extend_len, dtype=torch.int64, device="xpu")

    def old():
        write_req_to_token_pool_triton[(batch,)](
            old_pool,
            req_indices,
            pointers,
            prefix,
            seq,
            extensions,
            cache_loc,
            old_pool.stride(0),
        )

    def new():
        torch.ops.sgl_kernel.write_req_to_token_pool(
            new_pool, req_indices, pointers, prefix, seq, extensions, cache_loc
        )

    if main_only:
        profile_main(
            "write",
            {
                "batch": batch,
                "prefix_len": prefix_len,
                "extend_len": extend_len,
                "seq_len": seq_len,
                "pool_dtype": "int32",
                "cache_loc_dtype": "int64",
            },
            old,
            ("write_req_to_token_pool_triton",),
            samples,
        )
        return
    old()
    new()
    torch.xpu.synchronize()
    torch.testing.assert_close(new_pool.cpu(), old_pool.cpu(), rtol=0, atol=0)
    profile_pair(
        "write",
        {
            "batch": batch,
            "prefix_len": prefix_len,
            "extend_len": extend_len,
            "seq_len": seq_len,
            "pool_dtype": "int32",
            "cache_loc_dtype": "int64",
        },
        old,
        new,
        ("write_req_to_token_pool_triton",),
        "WriteReqToTokenPoolKernel",
        samples,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, parents=[MEMORY_SETUP_PARSER])
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=(256, 511, 512, 513, 1024, 2047, 2048, 2049, 4096),
    )
    parser.add_argument("--page-sizes", nargs="+", type=int, default=(64,))
    parser.add_argument("--prefix-lengths", nargs="+", type=int, default=(0, 4096))
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("decode", "extend", "write", "fused"),
        default=("decode", "extend", "write", "fused"),
    )
    args = parser.parse_args()
    if (
        args.samples < 1
        or args.batch < 1
        or any(x < 1 for x in (*args.lengths, *args.page_sizes))
        or any(x < 0 for x in args.prefix_lengths)
    ):
        parser.error("samples, batch, lengths and page sizes must be positive")
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "device": torch.xpu.get_device_name(0),
                "main_source": str(memory_setup.memory_source or "installed sglang"),
                "serving_scope": {"batch": 8, "isl": 4096, "osl": 16, "tp": 1, "ep": 1},
            }
        ),
        flush=True,
    )
    if "decode" in args.cases:
        compare_decode(args.samples, args.main_only)
    if "write" in args.cases:
        compare_write(args.samples, 0, 4096, args.main_only)
        compare_write(args.samples, 4096, 16, args.main_only)
    for page_size in args.page_sizes:
        for prefix_len in args.prefix_lengths:
            for extend_len in args.lengths:
                if "extend" in args.cases:
                    compare_extend(
                        args.batch,
                        prefix_len,
                        extend_len,
                        page_size,
                        args.samples,
                        False,
                        args.main_only,
                    )
                if "fused" in args.cases:
                    compare_extend(
                        args.batch,
                        prefix_len,
                        extend_len,
                        page_size,
                        args.samples,
                        True,
                        args.main_only,
                    )


if __name__ == "__main__":
    main()
