"""Device-event timing of repeated QK norm/RoPE XPU graph replay."""

import argparse
import hashlib
import statistics
from pathlib import Path

import torch
from sgl_kernel import fused_qk_norm_rope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--q-heads", type=int, default=128)
    parser.add_argument("--k-heads", type=int, default=16)
    parser.add_argument("--tail-heads", type=int, default=144)
    parser.add_argument("--head-dim", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()
    if (
        min(args.tokens, args.repeats, args.samples, args.q_heads, args.k_heads) <= 0
        or min(args.warmup, args.tail_heads) < 0
    ):
        parser.error("counts must be positive; warmup and tail-heads may be zero")
    torch.manual_seed(42)
    tokens, repeats, samples, warmup = (
        args.tokens,
        args.repeats,
        args.samples,
        args.warmup,
    )
    nq, nk, nv, hd = args.q_heads, args.k_heads, args.tail_heads, args.head_dim
    # Muse-Spark B8 uses T8/T32768, Q128/K16/D64 and gated tail144; other dimensions are generic corners.
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    scale = 43.7840518911 / (hd**0.5)
    qkv = torch.randn(tokens, (nq + nk + nv) * hd, device="xpu:0", dtype=dtype)
    initial = qkv.clone()
    q_weight = torch.full((hd,), scale, device="xpu:0", dtype=dtype)
    k_weight = torch.ones(hd, device="xpu:0", dtype=dtype)
    positions = torch.arange(tokens, device="xpu:0", dtype=torch.int32) % 4096

    def fn():
        fused_qk_norm_rope(
            qkv,
            nq,
            nk,
            nv,
            hd,
            1e-5,
            q_weight,
            k_weight,
            500000.0,
            False,
            positions,
            rotary_dim=hd,
        )

    for _ in range(5):
        fn()
    torch.xpu.synchronize()
    qkv.copy_(initial)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        for _ in range(repeats):
            fn()
    torch.xpu.synchronize()

    paths = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "libsgl-ops-sycl-FusedQKNormRope.so" in line
    }
    for path in sorted(paths):
        print(
            f"loaded_lib={path} sha256={hashlib.sha256(Path(path).read_bytes()).hexdigest()}",
            flush=True,
        )

    for _ in range(warmup):
        qkv.copy_(initial)
        graph.replay()
    torch.xpu.synchronize()
    values = []
    for _ in range(samples):
        qkv.copy_(initial)
        torch.xpu.synchronize()
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / repeats)

    ordered = sorted(values)
    print(
        f"graph tokens={tokens} dtype={args.dtype} heads={nq},{nk},{nv} hd={hd} "
        f"repeats={repeats} samples={samples} warmup={warmup} "
        f"device={torch.xpu.get_device_name(0)}",
        flush=True,
    )
    print(
        f"tokens={tokens} raw_us={','.join(f'{x:.3f}' for x in values)} "
        f"p50_us={statistics.median(values):.3f} "
        f"mad_us={statistics.median(abs(x - statistics.median(values)) for x in values):.3f} "
        f"p20_us={ordered[int((samples - 1) * 0.2)]:.3f} "
        f"p80_us={ordered[int((samples - 1) * 0.8)]:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
