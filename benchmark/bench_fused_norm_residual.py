"""Device-event timing of the registered residual normalization operator."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import sgl_kernel as sgl_kernel
import torch

# Muse-Spark B8 uses rows8 decode/rows32768 ISL4096 prefill, hidden8192, BF16 with FP32 stream.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 32768])
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument(
        "--post-mode", choices=("none", "weight", "input_scale"), default="none"
    )
    parser.add_argument("--round-pre-norm-input", action="store_true")
    parser.add_argument("--output-fp32", action="store_true")
    args = parser.parse_args()
    if (
        min(*args.rows, args.hidden_size, args.iterations, args.samples) <= 0
        or args.hidden_size % 8
        or args.warmup < 0
    ):
        parser.error(
            "positive counts and hidden-size divisible by 8 are required; "
            "warmup may be zero"
        )
    torch.manual_seed(42)
    for rows in args.rows:
        branch = torch.randn(rows, args.hidden_size, device="xpu", dtype=torch.bfloat16)
        stream = torch.randn(rows, args.hidden_size, device="xpu", dtype=torch.float32)
        post_weight = (
            torch.randn(args.hidden_size, device="xpu", dtype=torch.bfloat16)
            if args.post_mode != "none"
            else torch.empty(0, device="xpu", dtype=torch.bfloat16)
        )
        pre_weight = torch.randn(args.hidden_size, device="xpu", dtype=torch.bfloat16)
        alpha = torch.rand(args.hidden_size, device="xpu", dtype=torch.float32)
        beta = torch.rand(args.hidden_size, device="xpu", dtype=torch.float32)
        output = torch.empty_like(branch)
        new_stream = torch.empty_like(stream)
        output_fp32 = torch.empty_like(stream) if args.output_fp32 else None

        def invoke():
            torch.ops.sgl_kernel.fused_post_norm_residual_pre_norm(
                branch,
                stream,
                post_weight,
                pre_weight,
                alpha,
                beta,
                output,
                new_stream,
                1e-8,
                1e-5,
                args.post_mode == "input_scale",
                args.round_pre_norm_input,
                output_fp32,
            )

        for _ in range(args.warmup):
            invoke()
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            for _ in range(args.iterations):
                invoke()
        torch.xpu.synchronize()
        for _ in range(3):
            graph.replay()
        torch.xpu.synchronize()
        values = []
        for _ in range(args.samples):
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end) * 1000 / args.iterations)
        print(
            json.dumps(
                {
                    "rows": rows,
                    "hidden_size": args.hidden_size,
                    "device": torch.xpu.get_device_name(0),
                    "dtype": "branch/weights/output BF16; stream/alpha/beta FP32",
                    "layout": "contiguous row-major",
                    "post_mode": args.post_mode,
                    "round_pre_norm_input": args.round_pre_norm_input,
                    "output_fp32": args.output_fp32,
                    "eps": [1e-8, 1e-5],
                    "timing": "XPU event / graph replay iterations; setup outside timing",
                    "iterations": args.iterations,
                    "warmup": args.warmup,
                    "samples_us": values,
                    "p50_us": statistics.median(values),
                    "range_us": [min(values), max(values)],
                }
            ),
            flush=True,
        )
        del graph

    paths = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "libsgl-ops-sycl-FusedNormResidual.so" in line
    }
    for path in sorted(paths):
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        print(f"loaded_lib={path} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
