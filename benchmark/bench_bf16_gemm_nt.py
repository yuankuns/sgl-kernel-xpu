"""Compare BF16 GEMM dispatch and kernels with XPU graph event timing."""

import argparse
import csv
import os
import statistics
import time
from contextlib import nullcontext


def capture_xe35(function, warmup, calls):
    for _ in range(warmup):
        function()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        for _ in range(calls):
            function()
    for _ in range(warmup):
        graph.replay()
    torch.xpu.synchronize()
    return graph


def measure_xe35(graph, calls):
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / calls


def force_kernel_xe35(output, activations, weight, packed):
    torch.ops.sgl_kernel.bf16_gemm_nt_xe35.default(
        output, activations, weight, 2 if packed else 0
    )
    return output


def parse_args_xe35():
    if os.environ.get("BATCHES", "8") != "8":
        raise SystemExit("BATCHES must be 8; rejected before GPU work")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-batch", type=int, choices=(8,), default=8)
    parser.add_argument("--mode", choices=("all", "dispatch", "kernel"), default="all")
    parser.add_argument(
        "--shape", nargs=2, action="append", type=int, metavar=("N", "K")
    )
    parser.add_argument(
        # M counts GEMM rows, not the serving batch size.
        "--m",
        type=int,
        nargs="+",
        default=[8],
    )
    parser.add_argument("--n", type=int, default=32768)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", "--repeats", dest="repeats", type=int, default=20)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="Optional CSV with medians and sample spans")
    args = parser.parse_args()
    args.shapes = args.shape or [(args.n, args.k)]
    if any(
        columns <= 0 or inner <= 0 or columns % 16 or inner % 16
        for columns, inner in args.shapes
    ) or any(rows <= 0 for rows in args.m):
        parser.error("positive shapes with N and K divisible by 16 are required")
    if args.warmup < 0 or args.repeats <= 0 or args.calls <= 0:
        parser.error("--warmup must be nonnegative; repeats and calls must be positive")
    return args


def benchmark_case_xe35(args, weight, packed_weight, rows, names):
    columns, inner = weight.shape
    activations = torch.randn((rows, inner), dtype=torch.bfloat16, device=weight.device)
    native_output = torch.empty(
        (rows, columns), dtype=activations.dtype, device=activations.device
    )
    packed_output = torch.empty(
        (rows, packed_weight.shape[1]),
        dtype=activations.dtype,
        device=activations.device,
    )
    functions = {
        "native_mm": lambda: torch.mm(activations, weight.T),
        "native_cute": lambda: force_kernel_xe35(
            native_output, activations, weight, False
        ),
        "packed_mm": lambda: torch.mm(activations, packed_weight),
        "packed_cute": lambda: force_kernel_xe35(
            packed_output, activations, packed_weight, True
        ),
        "native_dispatch": lambda: bf16_gemm_nt_xe35(activations, weight),
        "packed_dispatch": lambda: bf16_gemm_nt_xe35(
            activations, packed_weight, packed_weight=True
        ),
        "native_mm_after": lambda: torch.mm(activations, weight.T),
    }
    reference = torch.mm(activations.float().cpu(), weight.float().cpu().T)
    for name in names:
        output = functions[name]()
        torch.testing.assert_close(
            output[:, :columns].float().cpu(), reference, atol=5e-3, rtol=5e-3
        )
    graphs = {
        name: capture_xe35(functions[name], args.warmup, args.calls) for name in names
    }
    samples = {name: [] for name in names}
    for index in range(args.repeats):
        comparison_names = tuple(name for name in names if name != "native_mm_after")
        order = (
            comparison_names if index % 2 == 0 else tuple(reversed(comparison_names))
        )
        if "native_mm_after" in names:
            order = (*order, "native_mm_after")
        for name in order:
            samples[name].append(measure_xe35(graphs[name], args.calls))
    result = {"m": rows, "n": columns, "k": inner}
    for name in names:
        result[name + "_ms"] = statistics.median(samples[name])
        result[name + "_span_ms"] = max(samples[name]) - min(samples[name])
    return result


def main_xe35():
    global torch, bf16_gemm_nt_xe35, pack_bf16_gemm_weight_xe35
    args = parse_args_xe35()
    import sgl_kernel
    import torch
    from sgl_kernel import bf16_gemm_nt_xe35, pack_bf16_gemm_weight_xe35

    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        raise RuntimeError("BF16 NT GEMM benchmark requires Xe35 (capability 3.5)")

    names = {
        "kernel": ("native_mm", "native_cute", "packed_mm", "packed_cute"),
        "dispatch": (
            "native_mm",
            "native_dispatch",
            "packed_mm",
            "packed_dispatch",
        ),
        "all": (
            "native_mm",
            "native_cute",
            "packed_mm",
            "packed_cute",
            "native_dispatch",
            "packed_dispatch",
            "native_mm_after",
        ),
    }[args.mode]
    torch.manual_seed(args.seed)
    print(
        f"device={torch.xpu.get_device_name(0)} torch={torch.__version__} "
        f"timing=xpu_graph_events mode={args.mode} "
        f"calls={args.calls} repeats={args.repeats}",
        flush=True,
    )
    fields = ["m", "n", "k"]
    fields += [name + "_ms" for name in names]
    fields += [name + "_span_ms" for name in names]
    output_context = (
        open(args.output, "w", newline="") if args.output else nullcontext(None)
    )
    with output_context as handle:
        writer = csv.DictWriter(handle, fieldnames=fields) if handle else None
        if writer:
            writer.writeheader()
        completed_cases = 0
        for columns, inner in args.shapes:
            weight = torch.randn((columns, inner), dtype=torch.bfloat16, device="xpu")
            packed_weight = pack_bf16_gemm_weight_xe35(weight)
            for rows in args.m:
                if completed_cases:
                    torch.xpu.synchronize()
                    time.sleep(2)
                result = benchmark_case_xe35(args, weight, packed_weight, rows, names)
                if writer:
                    writer.writerow(result)
                    handle.flush()
                timings = " ".join(
                    f"{name}={result[name + '_ms']:.5f}ms" for name in names
                )
                print(f"M={rows} N={columns} K={inner} {timings}", flush=True)
                completed_cases += 1


if __name__ == "__main__":
    main_xe35()
