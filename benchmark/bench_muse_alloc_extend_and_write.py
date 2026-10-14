"""B8 prefill: compare the public fused allocation path with the split kernels."""

import argparse
import os
import statistics
import time

import torch
import triton.testing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-batch", type=int, default=8)
    parser.add_argument("--isl", type=int, default=4096)
    parser.add_argument("--osl", type=int, default=16)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--samples", type=int, default=9)
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

    import sgl_kernel  # noqa: F401

    from sglang.kernels.ops.memory.allocator import (
        alloc_extend_and_write,
        alloc_extend_kernel,
    )
    from sglang.kernels.ops.memory.common import write_req_to_token_pool_triton

    batch, tokens_per_request, page_size = 8, 4096, 64
    total_tokens = batch * tokens_per_request
    prefix = torch.zeros(batch, dtype=torch.int64, device="xpu")
    lengths = torch.full((batch,), tokens_per_request, dtype=torch.int64, device="xpu")
    last = torch.full((batch,), -1, dtype=torch.int32, device="xpu")
    free_pages = torch.arange(1, 6001, dtype=torch.int64, device="xpu")
    req_indices = torch.arange(batch, dtype=torch.int64, device="xpu")
    dummy_prefix = torch.zeros(1, dtype=torch.int64, device="xpu")
    prefix_pointers = torch.tensor(
        [dummy_prefix.data_ptr()] * batch, dtype=torch.uint64, device="xpu"
    )
    split_out = torch.empty(total_tokens, dtype=torch.int64, device="xpu")
    fused_out = torch.empty_like(split_out)
    split_pool = torch.empty(
        (batch, tokens_per_request), dtype=torch.int32, device="xpu"
    )
    fused_pool = torch.empty_like(split_pool)

    def split():
        alloc_extend_kernel[(batch,)](
            prefix, lengths, last, free_pages, split_out, batch, page_size
        )
        write_req_to_token_pool_triton[(batch,)](
            split_pool,
            req_indices,
            prefix_pointers,
            prefix,
            lengths,
            lengths,
            split_out,
            split_pool.stride(0),
        )

    def fused():
        alloc_extend_and_write(
            prefix,
            lengths,
            last,
            free_pages,
            fused_out,
            fused_pool,
            req_indices,
            prefix_pointers,
            page_size,
        )

    split()
    fused()
    torch.xpu.synchronize()
    torch.testing.assert_close(fused_out, split_out)
    torch.testing.assert_close(fused_pool, split_pool)
    timings = {}
    for index, (label, fn) in enumerate((("split", split), ("fused", fused))):
        if index:
            torch.xpu.synchronize()
            time.sleep(2)
        timings[label] = statistics.median(
            triton.testing.do_bench(fn, warmup=1, rep=1) for _ in range(args.samples)
        )
        print(f"{label} p50={timings[label]:.6f} ms")
    print(f"fused_speedup={timings['split'] / timings['fused']:.3f}x")


if __name__ == "__main__":
    main()
