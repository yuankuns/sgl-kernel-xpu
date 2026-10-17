"""Paired graph replay timings for the residual norm review."""

import argparse
import json
import os
import statistics
import time

import torch

from bench_fused_norm_residual_review import (
    framework_triton_components,
    fused,
    framework_triton,
    framework_triton_w4,
    framework_triton_w16,
    framework_triton_w32,
    inputs,
    main_sycl,
    parse_shape,
)


PROVIDERS = {
    "fused": fused,
    "framework_triton": framework_triton,
    "framework_triton_w4": framework_triton_w4,
    "framework_triton_w16": framework_triton_w16,
    "framework_triton_w32": framework_triton_w32,
    "main_sycl": main_sycl,
}


def capture(op, args, tensors):
    for _ in range(args.warmup):
        op(args, tensors)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        for _ in range(args.graph_iterations):
            op(args, tensors)
    torch.xpu.synchronize()
    for _ in range(3):
        graph.replay()
    torch.xpu.synchronize()
    return graph


def run(args):
    _, resolve, _ = framework_triton_components()
    print(
        json.dumps({
            "device": torch.xpu.get_device_name(0),
            "gpu_index": 0,
            "host_gpu_affinity": os.environ.get("ZE_AFFINITY_MASK"),
            "torch": torch.__version__,
            "timing": "torch.xpu.Event enclosing one graph replay",
            "graph_iterations": args.graph_iterations,
            "samples": args.samples,
            "providers": args.providers,
            "dtype": "branch/weights/output BF16; stream/alpha/beta FP32",
            "layout": "row-major contiguous; identical tensors per provider",
            "eps": [1e-8, 1e-5],
            "round_pre_norm_input": True,
        }),
        flush=True,
    )
    for rows, hidden in args.shape:
        tensors = inputs(rows, hidden)
        m_bucket = 0 if rows <= 256 else (1 if rows <= 2048 else 2)
        warps, stages, extra = resolve(hidden, m_bucket, rows, False)
        print(
            json.dumps({
                "rows": rows,
                "hidden": hidden,
                "triton_resolved": {
                    "num_warps": warps, "num_stages": stages, "extra": extra
                },
                "post_mode": args.post_mode,
            }),
            flush=True,
        )
        graphs = {
            name: capture(PROVIDERS[name], args, tensors)
            for name in args.providers
        }
        samples = {name: [] for name in graphs}
        for repeat in range(args.samples):
            names = list(graphs)
            if repeat % 2:
                names.reverse()
            for name in names:
                start = torch.xpu.Event(enable_timing=True)
                end = torch.xpu.Event(enable_timing=True)
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(
                    start.elapsed_time(end) * 1000 / args.graph_iterations
                )
        for name, values in samples.items():
            print(
                json.dumps({
                    "provider": name,
                    "rows": rows,
                    "hidden": hidden,
                    "post_mode": args.post_mode,
                    "samples_us": values,
                    "p50_us": statistics.median(values),
                    "range_us": [min(values), max(values)],
                }),
                flush=True,
            )
        del graphs, tensors
        torch.xpu.synchronize()
        time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=parse_shape, action="append", required=True)
    parser.add_argument("--post-mode", choices=["none", "weight"], default="none")
    parser.add_argument("--providers", nargs="+", choices=PROVIDERS, default=list(PROVIDERS))
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--graph-iterations", type=int, default=40)
    parser.add_argument("--samples", type=int, default=11)
    args = parser.parse_args()
    args.round_pre_norm_input = True
    run(args)
