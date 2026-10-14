"""Measure the Muse Spark B8 vocabulary mask against the model Triton path."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from statistics import median

import sgl_kernel  # noqa: F401
import torch

from sglang.srt.layers.vocab_parallel_embedding import (
    _get_masked_input_and_mask_compiled,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-batch", type=int, default=8)
    parser.add_argument("--isl", type=int, default=4096)
    parser.add_argument("--osl", type=int, default=16)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if (args.serving_batch, args.isl, args.osl, args.tp, args.ep) != (
        8,
        4096,
        16,
        1,
        1,
    ) or os.environ.get("BATCHES", "8") != "8":
        parser.error("Muse-Spark serving scope requires B=8, ISL=4096, OSL=16, TP1/EP1")
    if args.samples < 1:
        parser.error("--samples must be positive")

    ids = torch.tensor(
        [0, 1, 99, 100, 200, 201, 206, 207], dtype=torch.int64, device="xpu"
    )
    mask_args = (0, 100, 16, 200, 207)

    def run_triton():
        return _get_masked_input_and_mask_compiled(ids, *mask_args)

    def run_sycl():
        return torch.ops.sgl_kernel.muse_vocab_mask(ids, *mask_args)

    for _ in range(10):
        run_triton()
        run_sycl()
    torch.xpu.synchronize()
    for actual, expected in zip(run_sycl(), run_triton()):
        torch.testing.assert_close(actual, expected)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profiler:
        for _ in range(args.samples):
            run_triton()
        for _ in range(args.samples):
            run_sycl()
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
    old = [
        event["dur"]
        for event in events
        if event.get("cat") == "kernel" and "triton_poi_fused" in event["name"]
    ]
    new = [
        event["dur"]
        for event in events
        if event.get("cat") == "kernel" and "MuseVocabMaskKernel" in event["name"]
    ]
    assert len(old) == len(new) == args.samples, (len(old), len(new))
    old_us, new_us = median(old), median(new)
    print(
        f"B8 vocab mask: Triton={old_us:.3f} us "
        f"SYCL={new_us:.3f} us speedup={old_us / new_us:.3f}x"
    )


if __name__ == "__main__":
    main()
