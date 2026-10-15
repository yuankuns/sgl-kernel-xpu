"""Measure token-map writer workgroups at the same XPU shapes."""

import argparse
import json
import tempfile
from pathlib import Path
from statistics import median

import torch

from bench_memory_compare import MEMORY_SETUP_PARSER, main_memory_ops, quantile


def measure(batch, prefix_len, extend_len, samples):
    _, _, main_write = main_memory_ops()
    seq_len = prefix_len + extend_len
    prefix_lens = torch.full((batch,), prefix_len, dtype=torch.int64, device="xpu")
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int64, device="xpu")
    extend_lens = torch.full((batch,), extend_len, dtype=torch.int64, device="xpu")
    req_indices = torch.arange(batch, dtype=torch.int64, device="xpu")
    prefix_values = torch.arange(max(1, batch * prefix_len), dtype=torch.int64, device="xpu")
    pointers = torch.tensor(
        [prefix_values.data_ptr() + index * prefix_len * 8 for index in range(batch)],
        dtype=torch.uint64, device="xpu",
    )
    locations = torch.arange(batch * extend_len, dtype=torch.int64, device="xpu")
    expected = torch.empty((batch, seq_len), dtype=torch.int32, device="xpu")
    main_write[(batch,)](
        expected, req_indices, pointers, prefix_lens, seq_lens, extend_lens,
        locations, expected.stride(0)
    )
    outputs = {groups: torch.empty_like(expected) for groups in (1, 2, 4, 8)}

    def run(groups):
        torch.ops.sgl_kernel.write_req_to_token_pool_tiled_for_benchmark(
            outputs[groups], req_indices, pointers, prefix_lens, seq_lens,
            extend_lens, locations, groups
        )

    for _ in range(5):
        for groups in outputs:
            run(groups)
    torch.xpu.synchronize()
    for output in outputs.values():
        torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=0, atol=0)
    order = []
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU]
    ) as profiler:
        for iteration in range(samples):
            choices = (1, 2, 4, 8)
            choices = choices[iteration % 4:] + choices[:iteration % 4]
            for groups in choices:
                run(groups)
                order.append(groups)
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
    kernels = sorted(
        (event for event in events
         if event.get("cat") == "kernel" and "WriteReqToTokenPoolKernel" in event.get("name", "")),
        key=lambda event: event["ts"],
    )
    if len(kernels) != len(order):
        raise RuntimeError(f"expected {len(order)} writer device events, got {len(kernels)}")
    durations = {groups: [] for groups in outputs}
    for groups, event in zip(order, kernels):
        durations[groups].append(event["dur"])
    print(json.dumps({
        "case": "write_groups", "batch": batch, "prefix_len": prefix_len,
        "extend_len": extend_len,
        "samples": samples, "device": torch.xpu.get_device_name(0),
        "timing": "torch.profiler XPU kernel event duration; rotated launch order",
        "groups": {
            groups: {
                "samples_us": values,
                "p20": quantile(values, 0.2),
                "p50": median(values),
                "p80": quantile(values, 0.8),
                "min": min(values), "max": max(values),
            }
            for groups, values in durations.items()
        },
    }), flush=True)
    torch.xpu.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__, parents=[MEMORY_SETUP_PARSER])
    parser.add_argument("--samples", type=int, default=51)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=(0, 4096))
    parser.add_argument(
        "--lengths", type=int, nargs="+",
        default=(256, 511, 512, 513, 1024, 2047, 2048, 2049, 4096)
    )
    args = parser.parse_args()
    if not args.test_library:
        parser.error("--test-library must name a tuning shim")
    if (args.samples < 1 or args.batch < 1 or
            any(length < 1 for length in args.lengths) or
            any(prefix < 0 for prefix in args.prefix_lengths)):
        parser.error("samples, batch and lengths must be positive")
    print(json.dumps({"torch": torch.__version__, "device": torch.xpu.get_device_name(0)}), flush=True)
    for prefix in args.prefix_lengths:
        for length in args.lengths:
            measure(args.batch, prefix, length, args.samples)


if __name__ == "__main__":
    main()
