"""CRI Muse-Spark B8 FMHA benchmark with production-path event timing."""

import argparse
import os
import statistics
import time

import sgl_kernel  # noqa: F401
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

COOLDOWN_SECONDS = 2


def make_case(batch: int, seqlen_q: int, seqlen_k: int, window: int, page_size: int):
    device = "xpu"
    dtype = torch.bfloat16
    heads_q, heads_kv, head_dim = 128, 16, 64
    pages_per_seq = (seqlen_k + page_size - 1) // page_size
    num_pages = batch * pages_per_seq
    q = torch.randn(batch * seqlen_q, heads_q, head_dim, device=device, dtype=dtype)
    k = torch.randn(
        num_pages, page_size, heads_kv, head_dim, device=device, dtype=dtype
    )
    v = torch.randn_like(k)
    page_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(
        batch, pages_per_seq
    )
    cache_seqlens = torch.full((batch,), seqlen_k, device=device, dtype=torch.int32)
    cu_seqlens_q = torch.arange(batch + 1, device=device, dtype=torch.int32) * seqlen_q
    out = torch.empty_like(q)

    def launch():
        return flash_attn_with_kvcache(
            q,
            k,
            v,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            softmax_scale=head_dim**-0.5,
            causal=True,
            window_size=(window, 0) if window >= 0 else (-1, -1),
            rotary_interleaved=False,
            out=out,
        )

    return launch


def measure(launch, warmup: int, iterations: int):
    samples = []
    for iteration in range(warmup + iterations):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        torch.xpu.synchronize()
        if iteration >= warmup:
            samples.append(start.elapsed_time(end))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serving-batch", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--sweep-local", action="store_true")
    parser.add_argument(
        "--cases",
        default="prefill-b8-wave1-swa2047-p64,prefill-b8-wave7-swa2047-p64,"
        "decode-b8-global-p64,decode-b8-swa2047-p64",
    )
    args = parser.parse_args()
    if args.serving_batch != 8 or os.environ.get("BATCHES", "8") != "8":
        parser.error("Muse-Spark serving benchmark requires B=8")

    cases = {
        "prefill-b8-wave1-global-p64": (1, 4096, 4096, -1, 64),
        "prefill-b8-wave7-global-p64": (7, 4096, 4096, -1, 64),
        "prefill-b8-wave1-swa2046-p64": (1, 4096, 4096, 2046, 64),
        "prefill-b8-wave1-swa2047-p64": (1, 4096, 4096, 2047, 64),
        "prefill-b8-wave1-swa2048-p64": (1, 4096, 4096, 2048, 64),
        "prefill-b8-wave1-swa2046-p128": (1, 4096, 4096, 2046, 128),
        "prefill-b8-wave1-swa2047-p128": (1, 4096, 4096, 2047, 128),
        "prefill-b8-wave1-swa2048-p128": (1, 4096, 4096, 2048, 128),
        "prefill-b8-wave7-swa2046-p64": (7, 4096, 4096, 2046, 64),
        "prefill-b8-wave7-swa2047-p64": (7, 4096, 4096, 2047, 64),
        "prefill-b8-wave7-swa2048-p64": (7, 4096, 4096, 2048, 64),
        "prefill-b8-wave7-swa2046-p128": (7, 4096, 4096, 2046, 128),
        "prefill-b8-wave7-swa2047-p128": (7, 4096, 4096, 2047, 128),
        "prefill-b8-wave7-swa2048-p128": (7, 4096, 4096, 2048, 128),
        "decode-b8-global-p64": (8, 1, 4097, -1, 64),
        "decode-b8-swa2047-p64": (8, 1, 4097, 2047, 64),
    }
    for wave in (1, 7):
        for page_size in (64, 128):
            for window in (0, 63, 511, 1023, 4095):
                cases[f"prefill-b8-wave{wave}-swa{window}-p{page_size}"] = (
                    wave,
                    4096,
                    4096,
                    window,
                    page_size,
                )
    names = (
        [
            f"prefill-b8-wave{wave}-swa{window}-p{page_size}"
            for wave in (1, 7)
            for page_size in (64, 128)
            for window in (0, 63, 511, 1023, 2046, 2047, 2048, 4095)
        ]
        if args.sweep_local
        else args.cases.split(",")
    )
    for name in names:
        batch, seqlen_q, seqlen_k, window, page_size = cases[name]
        samples = measure(
            make_case(batch, seqlen_q, seqlen_k, window, page_size),
            args.warmup,
            args.iterations,
        )
        median_ms = statistics.median(samples)
        effective_k = min(seqlen_k, window + 1) if window >= 0 else seqlen_k
        logical_kv_bytes = batch * effective_k * 16 * 64 * 2 * 2
        logical_kv_gbps = logical_kv_bytes / (median_ms * 1e6)
        print(
            f"{name}: median={median_ms:.4f} ms "
            f"min={min(samples):.4f} ms max={max(samples):.4f} ms "
            f"logical_kv={logical_kv_gbps:.1f} GB/s "
            f"samples={len(samples)} shape_cooldown={COOLDOWN_SECONDS}s"
        )
        # Shape sweeps must cool between cases. Keep this outside the measured
        # interval so each case still represents steady-state decode.
        torch.xpu.synchronize()
        time.sleep(COOLDOWN_SECONDS)


if __name__ == "__main__":
    main()
