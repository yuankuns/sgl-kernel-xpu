"""Compare vocabulary mask paths; defaults match Muse Spark B8 decode."""

import argparse
import json
import tempfile
from pathlib import Path
from statistics import median

import sgl_kernel  # noqa: F401
import torch

from sglang.srt.layers.vocab_parallel_embedding import get_masked_input_and_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    count = args.serving_batch if args.tokens is None else args.tokens
    if count < 1 or args.samples < 1:
        parser.error("tokens and samples must be positive")

    values = [0, 1, 99, 100, 200, 201, 206, 207]
    ids = torch.tensor(
        (values * ((count + len(values) - 1) // len(values)))[:count],
        dtype=torch.int64,
        device="xpu",
    )
    mask_args = (0, 100, 16, 200, 207)

    def run_triton():
        return get_masked_input_and_mask(ids, *mask_args)

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
        f"tokens={count} vocab mask: compiled={old_us:.3f} us "
        f"SYCL={new_us:.3f} us speedup={old_us / new_us:.3f}x"
    )


if __name__ == "__main__":
    main()
