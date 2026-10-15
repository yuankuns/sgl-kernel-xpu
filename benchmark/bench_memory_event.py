"""Compare main, production SYCL, and tile variants with batched XPU events."""

import argparse
import json
from statistics import median

import torch
import triton

from bench_memory_compare import MEMORY_SETUP_PARSER, main_memory_ops, quantile


def measure(args):
    alloc_extend_kernel, _, write_req_to_token_pool_triton = main_memory_ops()
    batch, prefix_len, extend_len, page_size = (
        args.batch, args.prefix_len, args.extend_len, args.page_size
    )
    prefix = torch.full((batch,), prefix_len, dtype=torch.int64, device="xpu")
    seq = torch.full((batch,), prefix_len + extend_len, dtype=torch.int64, device="xpu")
    extensions = torch.full((batch,), extend_len, dtype=torch.int64, device="xpu")
    last = torch.full((batch,), prefix_len - 1, dtype=torch.int32, device="xpu")
    free_pages = torch.arange(
        1, batch * ((extend_len + page_size - 1) // page_size) + 2,
        dtype=torch.int64, device="xpu",
    )
    labels = ("main", "production") if args.production_only else (
        "main", "production", "tile1", "tile4", "tile16"
    )
    outputs = {
        label: torch.empty(batch * extend_len, dtype=torch.int64, device="xpu")
        for label in labels
    }
    if args.case == "fused":
        pools = {
            label: torch.empty(
                (batch, prefix_len + extend_len), dtype=torch.int32, device="xpu"
            )
            for label in labels
        }
        req_indices = torch.arange(batch, dtype=torch.int64, device="xpu")
        prefix_values = torch.arange(
            max(1, batch * prefix_len), dtype=torch.int64, device="xpu"
        )
        pointers = torch.tensor(
            [prefix_values.data_ptr() + index * prefix_len * 8 for index in range(batch)],
            dtype=torch.uint64, device="xpu",
        )

        def main():
            alloc_extend_kernel[(batch,)](
                prefix, seq, last, free_pages, outputs["main"],
                triton.next_power_of_2(batch), page_size,
            )
            write_req_to_token_pool_triton[(batch,)](
                pools["main"], req_indices, pointers, prefix, seq, extensions,
                outputs["main"], pools["main"].stride(0),
            )

        def production():
            torch.ops.sgl_kernel.paged_alloc_extend_and_write(
                prefix, seq, last, free_pages, outputs["production"],
                pools["production"], req_indices, pointers, page_size,
            )

        def tiled(label, groups):
            torch.ops.sgl_kernel.paged_alloc_extend_and_write_tiled_for_benchmark(
                prefix, seq, last, free_pages, outputs[label], pools[label],
                req_indices, pointers, page_size, groups,
            )
    else:
        pools = None

        def main():
            alloc_extend_kernel[(batch,)](
                prefix, seq, last, free_pages, outputs["main"],
                triton.next_power_of_2(batch), page_size,
            )

        def production():
            torch.ops.sgl_kernel.paged_alloc_extend(
                prefix, seq, last, free_pages, outputs["production"], page_size,
            )

        def tiled(label, groups):
            torch.ops.sgl_kernel.paged_alloc_extend_tiled_for_benchmark(
                prefix, seq, last, free_pages, outputs[label], page_size, groups,
            )

    calls = {
        "main": main,
        "production": production,
        "tile1": lambda: tiled("tile1", 1),
        "tile4": lambda: tiled("tile4", 4),
        "tile16": lambda: tiled("tile16", 16),
    }
    for _ in range(5):
        for label in labels:
            calls[label]()
    torch.xpu.synchronize()
    expected = outputs["main"].cpu()
    for label in labels[1:]:
        torch.testing.assert_close(outputs[label].cpu(), expected, rtol=0, atol=0)
        if pools is not None:
            torch.testing.assert_close(
                pools[label].cpu(), pools["main"].cpu(), rtol=0, atol=0
            )

    durations = {label: [] for label in labels}
    for iteration in range(args.samples):
        rotated = labels[iteration % len(labels):] + labels[:iteration % len(labels)]
        for label in rotated:
            if args.trace_labels:
                print(f"window_start sample={iteration} variant={label}", flush=True)
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
            for _ in range(args.repeats):
                calls[label]()
            end.record()
            end.synchronize()
            durations[label].append(start.elapsed_time(end) * 1000 / args.repeats)
            if args.trace_labels:
                print(f"window_done sample={iteration} variant={label}", flush=True)

    print(json.dumps({
        "case": args.case, "batch": batch, "prefix_len": prefix_len,
        "extend_len": extend_len, "page_size": page_size,
        "dtype": {"metadata": "int64", "last": "int32", "pool": "int32"},
        "device": torch.xpu.get_device_name(0), "samples": args.samples,
        "repeats_per_sample": args.repeats,
        "timing": "torch.xpu.Event window / repeats, microseconds; includes gaps between enqueues",
        "variants": {
            label: {
                "samples_us": values, "p20": quantile(values, 0.2),
                "p50": median(values), "p80": quantile(values, 0.8),
                "min": min(values), "max": max(values),
            }
            for label, values in durations.items()
        },
    }), flush=True)
    torch.xpu.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__, parents=[MEMORY_SETUP_PARSER])
    parser.add_argument("--case", choices=("extend", "fused"), required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prefix-len", type=int, required=True)
    parser.add_argument("--extend-len", type=int, required=True)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--production-only", action="store_true")
    parser.add_argument("--trace-labels", action="store_true")
    args = parser.parse_args()
    if (args.batch < 1 or args.prefix_len < 0 or args.extend_len < 1 or
            args.page_size < 1 or args.samples < 1 or args.repeats < 1):
        parser.error("shape, samples and repeats are out of range")
    measure(args)


if __name__ == "__main__":
    main()
