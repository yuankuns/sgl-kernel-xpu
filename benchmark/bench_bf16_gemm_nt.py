import argparse
import statistics

import torch
from sgl_kernel import bf16_gemm_nt, pack_bf16_gemm_weight


def measure(fn, warmup=5, repeat=20, calls=20):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        for _ in range(calls):
            fn()
    for _ in range(warmup):
        graph.replay()
    torch.xpu.synchronize()
    starts = [torch.xpu.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.xpu.Event(enable_timing=True) for _ in range(repeat)]
    for start, end in zip(starts, ends):
        start.record()
        graph.replay()
        end.record()
    torch.xpu.synchronize()
    return (
        statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends))
        / calls
    )


def force_packed_kernel(a, packed_b):
    output = torch.empty(
        (a.shape[0], packed_b.shape[1]), dtype=a.dtype, device=a.device
    )
    torch.ops.sgl_kernel.bf16_gemm_nt.default(output, a, packed_b, 2)
    return output


def force_native_kernel(a, b):
    output = torch.empty((a.shape[0], b.shape[0]), dtype=a.dtype, device=a.device)
    torch.ops.sgl_kernel.bf16_gemm_nt.default(output, a, b, 0)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serving-batch", type=int, choices=(8,), default=8)
    parser.add_argument(
        # M is the GEMM row count, not the serving batch size.
        "--m",
        type=int,
        nargs="+",
        default=[1, 8, 9, 16, 17, 32, 33, 64],
    )
    parser.add_argument("--n", type=int, default=32768)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--calls", type=int, default=20)
    args = parser.parse_args()
    if args.n <= 0 or args.k <= 0 or any(m <= 0 for m in args.m):
        parser.error("M, N and K must be positive")
    if args.k % 16:
        parser.error("K must be a multiple of 16 for the forced kernel comparison")
    if args.warmup < 0 or args.repeat <= 0 or args.calls <= 0:
        parser.error("--warmup must be nonnegative; --repeat and --calls positive")

    print(
        f"device={torch.xpu.get_device_name(0)} timing=xpu_graph_events "
        f"calls={args.calls} repeats={args.repeat}"
    )
    b = torch.randn((args.n, args.k), dtype=torch.bfloat16, device="xpu")
    packed_b = pack_bf16_gemm_weight(b)
    for index, m in enumerate(args.m):
        if index:
            torch.xpu.synchronize()
        a = torch.randn((m, args.k), dtype=torch.bfloat16, device="xpu")
        reference = torch.mm(a.float().cpu(), b.float().cpu().T)
        torch.testing.assert_close(
            force_packed_kernel(a, packed_b)[:, : args.n].float().cpu(),
            reference,
            rtol=5e-3,
            atol=5e-3,
        )
        torch.testing.assert_close(
            force_native_kernel(a, b).float().cpu(),
            reference,
            rtol=5e-3,
            atol=5e-3,
        )
        torch.testing.assert_close(
            torch.mm(a, packed_b)[:, : args.n].float().cpu(),
            reference,
            rtol=5e-3,
            atol=5e-3,
        )
        reference_ms = measure(
            lambda: torch.mm(a, b.T), args.warmup, args.repeat, args.calls
        )
        native_dispatch_ms = measure(
            lambda: bf16_gemm_nt(a, b), args.warmup, args.repeat, args.calls
        )
        native_kernel_ms = measure(
            lambda: force_native_kernel(a, b),
            args.warmup,
            args.repeat,
            args.calls,
        )
        packed_onednn_ms = measure(
            lambda: torch.mm(a, packed_b), args.warmup, args.repeat, args.calls
        )
        packed_dispatch_ms = measure(
            lambda: bf16_gemm_nt(a, packed_b, packed_weight=True),
            args.warmup,
            args.repeat,
            args.calls,
        )
        packed_kernel_ms = measure(
            lambda: force_packed_kernel(a, packed_b),
            args.warmup,
            args.repeat,
            args.calls,
        )
        reference_after_ms = measure(
            lambda: torch.mm(a, b.T), args.warmup, args.repeat, args.calls
        )
        print(
            f"M={m} N={args.n} K={args.k}: torch={reference_ms:.4f} ms, "
            f"native_dispatch={native_dispatch_ms:.4f} ms "
            f"({reference_ms / native_dispatch_ms:.3f}x), "
            f"native_cute_forced={native_kernel_ms:.4f} ms "
            f"({reference_ms / native_kernel_ms:.3f}x), "
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
