"""Compare BF16 GEMM backends with repeated XPU graph replays.

Run with --shape N K (repeatable) and --m M [M ...]. N is the unpadded
weight width; both backends receive the same packed weight for packed timing.
"""

import argparse
import csv
import statistics
import time

import torch
from sgl_kernel import pack_bf16_gemm_weight


def capture(fn, calls):
    for _ in range(5):
        fn()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        for _ in range(calls):
            fn()
    for _ in range(4):
        graph.replay()
    torch.xpu.synchronize()
    return graph


def measure(graph, calls):
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", nargs=2, action="append", type=int, required=True)
    parser.add_argument("--m", nargs="+", type=int, required=True)
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        any(n <= 0 or k <= 0 or n % 16 or k % 16 for n, k in args.shape)
        or any(m <= 0 for m in args.m)
        or args.calls <= 0
        or args.repeats <= 0
        or args.cooldown < 2
    ):
        parser.error(
            "positive aligned shapes, positive counts, and cooldown >= 2 are required"
        )

    print(
        f"device={torch.xpu.get_device_name(0)} torch={torch.__version__} "
        f"calls={args.calls} repeats={args.repeats}",
        flush=True,
    )
    fields = ["m", "n", "k"]
    names = ("native_mm", "native_cute", "packed_mm", "packed_cute")
    fields += [f"{name}_ms" for name in names]
    fields += [f"{name}_span_ms" for name in names]
    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for n, k in args.shape:
            b = torch.randn((n, k), dtype=torch.bfloat16, device="xpu")
            packed_b = pack_bf16_gemm_weight(b)
            for m in args.m:
                torch.xpu.synchronize()
                time.sleep(args.cooldown)
                a = torch.randn((m, k), dtype=torch.bfloat16, device="xpu")
                native_out = torch.empty((m, n), dtype=a.dtype, device=a.device)
                packed_out = torch.empty(
                    (m, packed_b.shape[1]), dtype=a.dtype, device=a.device
                )
                functions = {
                    "native_mm": lambda: torch.mm(a, b.T),
                    "native_cute": lambda: torch.ops.sgl_kernel.bf16_gemm_nt.default(
                        native_out, a, b, 0
                    ),
                    "packed_mm": lambda: torch.mm(a, packed_b),
                    "packed_cute": lambda: torch.ops.sgl_kernel.bf16_gemm_nt.default(
                        packed_out, a, packed_b, 2
                    ),
                }
                reference = functions["native_mm"]()
                functions["native_cute"]()
                functions["packed_cute"]()
                torch.testing.assert_close(native_out, reference, rtol=0.02, atol=0.02)
                torch.testing.assert_close(
                    packed_out[:, :n], reference, rtol=0.02, atol=0.02
                )
                graphs = {
                    name: capture(fn, args.calls) for name, fn in functions.items()
                }
                samples = {name: [] for name in names}
                for index in range(args.repeats):
                    order = names if index % 2 == 0 else reversed(names)
                    for name in order:
                        samples[name].append(measure(graphs[name], args.calls))
                medians = [statistics.median(samples[name]) for name in names]
                spans = [max(samples[name]) - min(samples[name]) for name in names]
                writer.writerow([m, n, k, *medians, *spans])
                handle.flush()
                print(
                    f"M={m} N={n} K={k} "
                    f"packed_mm={medians[2]:.5f} packed_cute={medians[3]:.5f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
