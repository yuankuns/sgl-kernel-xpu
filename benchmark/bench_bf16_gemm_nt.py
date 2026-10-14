import argparse
import os
import statistics
import time

import torch
from sgl_kernel import bf16_gemm_nt, pack_bf16_gemm_weight


def measure(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def force_packed_kernel(a, packed_b):
    output = torch.empty(
        (a.shape[0], packed_b.shape[1]), dtype=a.dtype, device=a.device
    )
    torch.ops.sgl_kernel.bf16_gemm_nt.default(output, a, packed_b, 2)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--isl", type=int, default=4096)
    parser.add_argument("--osl", type=int, default=16)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument(
        # M is the GEMM row count, not the serving batch size.
        "--m",
        type=int,
        nargs="+",
        default=[8],
    )
    parser.add_argument("--n", type=int, default=202048)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    if (args.batch_size, args.isl, args.osl, args.tp, args.ep) != (
        8,
        4096,
        16,
        1,
        1,
    ) or os.environ.get("BATCHES", "8") != "8":
        parser.error(
            "Muse-Spark serving scope requires B=8, ISL=4096, OSL=16, TP=1, EP=1"
        )
    if len(args.m) > 1 and args.cooldown < 2.0:
        parser.error("--cooldown must be at least 2 seconds for a shape sweep")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be nonnegative and --repeat positive")

    b = torch.randn((args.n, args.k), dtype=torch.bfloat16, device="xpu")
    packed_b = pack_bf16_gemm_weight(b)
    for index, m in enumerate(args.m):
        if index:
            torch.xpu.synchronize()
            time.sleep(args.cooldown)
        a = torch.randn((m, args.k), dtype=torch.bfloat16, device="xpu")
        reference = torch.mm(a, b.T)
        torch.testing.assert_close(
            force_packed_kernel(a, packed_b)[:, : args.n],
            reference,
            rtol=0.02,
            atol=0.02,
        )
        torch.testing.assert_close(
            torch.mm(a, packed_b)[:, : args.n],
            reference,
            rtol=0.02,
            atol=0.02,
        )
        reference_ms = measure(lambda: torch.mm(a, b.T), args.warmup, args.repeat)
        kernel_ms = measure(lambda: bf16_gemm_nt(a, b), args.warmup, args.repeat)
        packed_onednn_ms = measure(
            lambda: torch.mm(a, packed_b), args.warmup, args.repeat
        )
        packed_dispatch_ms = measure(
            lambda: bf16_gemm_nt(a, packed_b, packed_weight=True),
            args.warmup,
            args.repeat,
        )
        packed_kernel_ms = measure(
            lambda: force_packed_kernel(a, packed_b), args.warmup, args.repeat
        )
        reference_after_ms = measure(lambda: torch.mm(a, b.T), args.warmup, args.repeat)
        print(
            f"M={m} N={args.n} K={args.k}: torch={reference_ms:.4f} ms, "
            f"cutlass={kernel_ms:.4f} ms "
            f"({reference_ms / kernel_ms:.3f}x), "
            f"packed_onednn={packed_onednn_ms:.4f} ms "
            f"({reference_ms / packed_onednn_ms:.3f}x), "
            f"packed_dispatch={packed_dispatch_ms:.4f} ms "
            f"({reference_ms / packed_dispatch_ms:.3f}x), "
            f"cutlass_packed_forced={packed_kernel_ms:.4f} ms "
            f"({reference_ms / packed_kernel_ms:.3f}x), "
            f"torch_after={reference_after_ms:.4f} ms"
        )


if __name__ == "__main__":
    main()
