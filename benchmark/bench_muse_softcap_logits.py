"""Compare Triton and SYCL logits softcap; defaults match Muse Spark B8."""

import argparse
import json
import tempfile
from statistics import median

import sgl_kernel  # noqa: F401
import torch
import triton

from sglang.kernels.ops.activation.softcap import softcap_inplace_logits_kernel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=202048)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--order", choices=("triton-first", "sycl-first"), default="triton-first"
    )
    args = parser.parse_args()
    if args.batch < 1 or args.vocab < 1 or args.samples < 1:
        parser.error("batch, vocab, and samples must be positive")
    cap = 20.0
    source = torch.randn(args.batch, args.vocab, device="xpu", dtype=torch.float32)
    triton_logits = source.clone()
    sycl_logits = source.clone()

    def run_triton() -> None:
        softcap_inplace_logits_kernel[(triton.cdiv(triton_logits.numel(), 1024), 1)](
            triton_logits, cap, triton_logits.numel(), triton_logits.numel(), 1024
        )

    def run_sycl() -> None:
        torch.ops.sgl_kernel.muse_softcap_logits(sycl_logits, cap)

    run_triton()
    run_sycl()
    torch.testing.assert_close(sycl_logits, triton_logits, atol=3e-5, rtol=3e-5)
    torch.xpu.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profiler:
        runs = (
            (run_triton, run_sycl)
            if args.order == "triton-first"
            else (run_sycl, run_triton)
        )
        for _ in range(args.samples):
            for run in runs:
                run()
        torch.xpu.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as trace_file:
        profiler.export_chrome_trace(trace_file.name)
        with open(trace_file.name) as stream:
            events = json.load(stream)["traceEvents"]
    durations = {
        "triton": [
            event["dur"]
            for event in events
            if event.get("cat") == "kernel"
            and event["name"] == "softcap_inplace_logits_kernel"
        ],
        "sycl": [
            event["dur"]
            for event in events
            if event.get("cat") == "kernel" and "MuseSoftcapLogits" in event["name"]
        ],
    }
    logical_bytes = 2 * source.numel() * source.element_size()
    for name, samples in durations.items():
        assert len(samples) == args.samples, (name, len(samples))
        latency_us = median(samples)
        print(
            f"{name}: p50={latency_us:.3f} us, logical_bandwidth={logical_bytes / (latency_us * 1e-6) / 1e9:.1f} GB/s"
        )
    print(f"speedup={median(durations['triton']) / median(durations['sycl']):.3f}x")


if __name__ == "__main__":
    main()
