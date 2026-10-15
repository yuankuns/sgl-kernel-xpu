"""Repeated device-event comparison of QK norm/RoPE implementations.

Run with an installed sgl_kernel package and the selected target library in
that package. --impl selects the fused operator or unfused reference path.
"""

import hashlib
import json
import argparse
import statistics
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from sgl_kernel import fused_qk_norm_rope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("sglang", "unfused"), default="sglang")
    parser.add_argument(
        "--tokens", type=int, nargs="+",
        default=[1, 4, 8, 16, 32, 64, 128, 256, 512, 2048, 32768],
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--q-heads", type=int, default=128)
    parser.add_argument("--k-heads", type=int, default=16)
    parser.add_argument("--tail-heads", type=int, default=144)
    parser.add_argument("--head-dim", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if (
        min(*args.tokens, args.samples, args.q_heads, args.k_heads) <= 0
        or min(args.warmup, args.tail_heads) < 0
    ):
        parser.error("counts must be positive; warmup and tail-heads may be zero")
    torch.manual_seed(42)
    impl, tokens_list = args.impl, args.tokens
    samples, warmup = args.samples, args.warmup
    nq, nk, hd = args.q_heads, args.k_heads, args.head_dim
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    # Muse's gated QKV projection carries V16 plus an untouched O-gate128.
    nv = args.tail_heads
    eps, base = 1e-5, 500000.0
    scale = 43.7840518911 / (hd**0.5)
    device = "xpu:0"
    print(
        f"impl={impl} dtype={args.dtype} heads={nq},{nk},{nv} hd={hd} "
        f"eps={eps} base={base} samples={samples} warmup={warmup} "
        f"device={torch.xpu.get_device_name(0)}",
        flush=True,
    )
    if impl == "sglang":
        paths = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libsgl-ops-sycl-FusedQKNormRope.so" in line
        }
        if not paths:
            print("loaded_lib=not_mapped_at_import", flush=True)
        for path in sorted(paths):
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            print(f"loaded_lib={path} sha256={digest}", flush=True)
    for tokens in tokens_list:
        qkv = torch.randn(
            tokens, (nq + nk + nv) * hd, device=device, dtype=dtype
        )
        original_qkv = qkv.clone()
        q_weight = torch.full((hd,), scale, device=device, dtype=dtype)
        k_weight = torch.ones(hd, device=device, dtype=dtype)
        positions = torch.arange(tokens, device=device, dtype=torch.int32) % 4096
        if impl == "unfused":
            inv_freq = 1.0 / (
                base ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)
            )
            freqs = torch.outer(torch.arange(32768, dtype=torch.float32), inv_freq)
            cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(
                device=device, dtype=dtype
            )
            rope_positions = positions.to(torch.long)

            def fn():
                q = qkv[:, : nq * hd]
                k = qkv[:, nq * hd : (nq + nk) * hd]
                q = (
                    F.rms_norm(q.reshape(-1, hd), (hd,), None, eps)
                    .reshape_as(q)
                    .to(dtype)
                    * scale
                )
                k = (
                    F.rms_norm(k.reshape(-1, hd), (hd,), None, eps)
                    .reshape_as(k)
                    .to(dtype)
                )
                torch.ops.sgl_kernel.rotary_embedding(
                    rope_positions, q, k, hd, cache, False
                )

        elif impl == "sglang":

            def fn():
                fused_qk_norm_rope(
                    qkv,
                    nq,
                    nk,
                    nv,
                    hd,
                    eps,
                    q_weight,
                    k_weight,
                    base,
                    False,
                    positions,
                    rotary_dim=hd,
                )

        else:
            raise ValueError(impl)

        for _ in range(warmup):
            qkv.copy_(original_qkv)
            fn()
        torch.xpu.synchronize()
        values = []
        for _ in range(samples):
            qkv.copy_(original_qkv)
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end) * 1000)
        print(
            f"tokens={tokens} raw_us={','.join(f'{x:.3f}' for x in values)} "
            f"p50_us={statistics.median(values):.3f} "
            f"mad_us={statistics.median(abs(x - statistics.median(values)) for x in values):.3f} "
            f"p20_us={sorted(values)[int((samples-1)*0.2)]:.3f} "
            f"p80_us={sorted(values)[int((samples-1)*0.8)]:.3f}",
            flush=True,
        )
        if args.profile:
            qkv.copy_(original_qkv)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.XPU]
            ) as profiler:
                fn()
                torch.xpu.synchronize()
            with tempfile.TemporaryDirectory() as directory:
                trace = Path(directory) / "qk-trace.json"
                profiler.export_chrome_trace(str(trace))
                events = json.loads(trace.read_text())["traceEvents"]
            names = [
                event.get("name", "")
                for event in events
                if event.get("cat") == "kernel"
            ]
            print(f"tokens={tokens} kernel_names={names}", flush=True)
        qkv = None
        original_qkv = None
        torch.xpu.synchronize()


if __name__ == "__main__":
    main()
