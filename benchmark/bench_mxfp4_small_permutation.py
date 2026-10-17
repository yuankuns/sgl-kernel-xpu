"""Device time for the registered Muse Spark B8 small permutation kernel."""

import argparse
import os


def main_xe35() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-batch", type=int, default=8)
    parser.add_argument("--isl", type=int, default=4096)
    parser.add_argument("--osl", type=int, default=16)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--experts", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument("--alignment", type=int, choices=[4, 16], default=4)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=500)
    args = parser.parse_args()
    if (args.serving_batch, args.isl, args.osl, args.tp, args.ep) != (
        8,
        4096,
        16,
        1,
        1,
    ) or os.environ.get("BATCHES", "8") != "8":
        parser.error("Muse-Spark serving scope requires B=8, ISL=4096, OSL=16, TP1/EP1")
    if args.warmup_ms <= 0 or args.rep_ms <= 0:
        parser.error("warmup and repetition times must be positive")

    import sgl_kernel
    import torch
    import triton.testing
    from sgl_kernel import mxfp4_small_permutation_xe35

    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        raise RuntimeError("Small permutation benchmark requires Xe35 (capability 3.5)")

    tokens, topk = 8, 8
    pairs = tokens * topk
    rows = pairs * args.alignment
    topk_ids = (
        torch.arange(pairs, device="xpu", dtype=torch.int32).reshape(tokens, topk)
        % args.experts
    )
    padded_counts = torch.empty(args.experts, device="xpu", dtype=torch.int32)
    perm_src = torch.empty(rows, device="xpu", dtype=torch.int32)
    expanded_to_permuted = torch.empty(pairs, device="xpu", dtype=torch.int32)
    scale_map = torch.empty(rows, 2, device="xpu", dtype=torch.int32)

    def run_xe35() -> None:
        mxfp4_small_permutation_xe35(
            topk_ids,
            padded_counts,
            perm_src,
            expanded_to_permuted,
            scale_map,
            args.alignment,
        )

    run_xe35()
    torch.xpu.synchronize()
    p50, p20, p80 = triton.testing.do_bench(
        run_xe35, warmup=args.warmup_ms, rep=args.rep_ms, quantiles=[0.5, 0.2, 0.8]
    )
    print(
        f"B8 topk={topk} experts={args.experts} alignment={args.alignment}: "
        f"p20={p20:.6f} ms p50={p50:.6f} ms p80={p80:.6f} ms"
    )


if __name__ == "__main__":
    main_xe35()
