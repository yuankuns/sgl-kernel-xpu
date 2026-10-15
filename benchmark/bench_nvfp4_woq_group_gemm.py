# SPDX-License-Identifier: Apache-2.0
"""Benchmark NVFP4 weight-only grouped GEMMs on Intel XPU.

Xe3P has no native NVFP4 MMA, so NVFP4 weights are requantized to MXFP8 inside
the mainloop. Two configurations are timed:

  mxfp8  Upper bound: the plain MXFP8 grouped GEMM over weights that are already
         MXFP8. Not an NVFP4 path — the reference to beat.
  woq    NVFP4 requantized to MXFP8 inside the mainloop. Weight traffic stays at
         4 bits/element; the cost is the in-register conversion.

The gap between them is the price of the emulation, and the weight-byte columns
show what it buys.

Config mode measures the NVFP4 MXFP8 and BF16 paths at B8 decode and
ISL4096 prefill shapes.

Usage:
    python bench_nvfp4_woq_group_gemm.py                 # default sweep
    python bench_nvfp4_woq_group_gemm.py --m 64 --n 512 --k 512 --groups 4
"""

import argparse
import json
import statistics
from typing import List, Tuple

import torch
from accuracy_reference import verify_grouped_output

MXFP8_BLOCK_SIZE = 32
NVFP4_BLOCK_SIZE = 16
FP8_E4M3_MAX = 448.0
E2M1_MAX = 6.0
UE8M0_BIAS = 127


def synchronize_before_shape() -> None:
    torch.xpu.synchronize()


def is_cri_device() -> bool:
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return False
    try:
        from sgl_kernel import is_xe3_arch

        return is_xe3_arch()
    except ImportError:
        return False


# --- minimal quantizers (self-contained so the benchmark has no test deps) ---

_E2M1_MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def quantize_mxfp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    blocks = xf.reshape(
        *xf.shape[:-1], xf.shape[-1] // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE
    )
    amax = blocks.abs().amax(dim=-1)
    m, e = torch.frexp(amax)
    kexp = torch.where(m <= 0.875, e - 9, e - 8).float()
    kexp = torch.where(amax > 0, kexp, torch.zeros_like(kexp)).clamp(-UE8M0_BIAS, 127)
    q = (blocks / torch.exp2(kexp).unsqueeze(-1)).reshape(*xf.shape)
    return q.to(torch.float8_e4m3fn), (kexp + UE8M0_BIAS).to(torch.uint8)


def quantize_nvfp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, k = x.shape
    xf = x.float()
    amax = xf.abs().amax().clamp_min(1e-12)
    gs = ((FP8_E4M3_MAX * E2M1_MAX) / amax).to(torch.float32)
    blocks = xf.reshape(n, k // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    sf = ((blocks.abs().amax(dim=-1) / E2M1_MAX) * gs).clamp(max=FP8_E4M3_MAX)
    sf16 = sf.to(torch.float8_e4m3fn)
    inv = (sf16.float() / gs).clamp_min(1e-30).unsqueeze(-1)
    scaled = (blocks / inv).reshape(n, k)

    mags = torch.tensor(_E2M1_MAGS)
    mid = (mags[:-1] + mags[1:]) / 2
    a = scaled.abs().clamp(max=E2M1_MAX)
    bucket = torch.bucketize(a, mid, right=False).to(torch.int64)
    at_mid = (a.unsqueeze(-1) == mid).any(dim=-1)
    idx = torch.where(at_mid, torch.where(bucket % 2 == 0, bucket, bucket + 1), bucket)
    nib = (((scaled < 0).to(torch.uint8)) << 3) | idx.to(torch.uint8)
    packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
    return packed.to(torch.uint8).contiguous(), sf16, gs


def build_inputs(groups: int, m_per_group: int, n: int, k: int, device: str):
    torch.manual_seed(0)
    w = torch.randn(groups, n, k) * 0.05
    qs, sfs, gss = [], [], []
    for e in range(groups):
        q, sf, gs = quantize_nvfp4(w[e])
        qs.append(q)
        sfs.append(sf)
        gss.append(1.0 / gs)
    b_fp4 = torch.stack(qs).to(device)
    b_sf = torch.stack(sfs).to(device)
    b_gs = torch.stack(gss).float().to(device)

    # MXFP8 weights for the upper-bound configuration, quantized straight from
    # the same FP32 source rather than requantized from NVFP4.
    b_mx, b_smx = quantize_mxfp8(w)
    b_mx, b_smx = b_mx.to(device), b_smx.to(device)

    m_total = groups * m_per_group
    a_q, a_s = quantize_mxfp8(torch.randn(m_total, k) * 0.5)
    a_q, a_s = a_q.to(device), a_s.to(device)

    problem_sizes = torch.tensor(
        [[m_per_group, n, k]] * groups, dtype=torch.int32, device=device
    )
    expert_offsets = torch.tensor(
        [e * m_per_group for e in range(groups)], dtype=torch.int32, device=device
    )
    return (
        a_q,
        a_s,
        b_fp4,
        b_sf,
        b_gs,
        b_mx,
        b_smx,
        problem_sizes,
        expert_offsets,
        m_total,
    )


def timeit_samples(fn, num_warmup: int, num_run: int) -> List[float]:
    for _ in range(num_warmup):
        fn()
    torch.xpu.synchronize()
    starts = [torch.xpu.Event(enable_timing=True) for _ in range(num_run)]
    ends = [torch.xpu.Event(enable_timing=True) for _ in range(num_run)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.xpu.synchronize()
    return [start.elapsed_time(end) * 1e3 for start, end in zip(starts, ends)]


def timeit(fn, num_warmup: int, num_run: int) -> float:
    return statistics.mean(timeit_samples(fn, num_warmup, num_run))


def bench_one(
    groups: int, m_per_group: int, n: int, k: int, num_warmup: int, num_run: int
):
    from sgl_kernel import (
        fp8_blockwise_scaled_grouped_mm,
        nvfp4_woq_mxfp8_scaled_grouped_mm,
        nvfp4_woq_prep,
    )

    device = "xpu"
    (
        a_q,
        a_s,
        b_fp4,
        b_sf,
        b_gs,
        b_mx,
        b_smx,
        problem_sizes,
        expert_offsets,
        m_total,
    ) = build_inputs(groups, m_per_group, n, k, device)
    workspace = torch.zeros((64 * 1024 * 1024,), dtype=torch.uint8, device=device)
    empty = torch.empty((0,), dtype=torch.int64, device=device)
    zeros_stride = torch.zeros((groups,), dtype=torch.int64, device=device)
    zeros_layout = torch.zeros((groups, 5), dtype=torch.int32, device=device)
    out = torch.zeros((m_total, n), dtype=torch.float32, device=device)

    c_b, sfmx_b = nvfp4_woq_prep(b_sf, b_gs)

    def mxfp8_gemm():
        fp8_blockwise_scaled_grouped_mm(
            out,
            empty,
            empty,
            empty,
            empty,
            empty,
            a_q,
            b_mx,
            a_s,
            b_smx,
            zeros_stride,
            zeros_stride,
            zeros_stride,
            zeros_layout,
            zeros_layout,
            problem_sizes,
            expert_offsets,
            workspace,
        )

    def woq_gemm():
        nvfp4_woq_mxfp8_scaled_grouped_mm(
            out, a_q, b_fp4, a_s, sfmx_b, c_b, problem_sizes, expert_offsets, workspace
        )

    flops = 2.0 * m_total * n * k
    res = {"groups": groups, "m_per_group": m_per_group, "n": n, "k": k}
    for name, fn in (("mxfp8", mxfp8_gemm), ("woq", woq_gemm)):
        fn()
        torch.xpu.synchronize()
        if name == "woq":
            verify_grouped_output(
                out, a_q, b_fp4, c_b, sfmx_b, [m_per_group] * groups, "mxfp8", a_s
            )
        else:
            verify_grouped_output(
                out, a_q, b_mx, None, b_smx, [m_per_group] * groups, "plain_mxfp8", a_s
            )
        us = timeit(fn, num_warmup, num_run)
        res[f"{name}_us"] = us
        res[f"{name}_tflops"] = flops / (us * 1e-6) / 1e12
    res["woq_vs_mxfp8"] = res["woq_us"] / res["mxfp8_us"]
    # NVFP4 weight bytes vs MXFP8 weight bytes actually streamed per GEMM.
    res["weight_mb_nvfp4"] = groups * n * k * 0.5 / 1e6
    res["weight_mb_mxfp8"] = groups * n * k * 1.0 / 1e6
    return res


def effective_metrics(
    groups: int,
    total_m: int,
    n: int,
    k: int,
    time_us: float,
    compute: str = "mxfp8",
) -> dict:
    """Logical GEMM traffic and FLOPs for a prepared grouped-GEMM path.

    Traffic counts tensors used by active experts once. The figure is an
    effective bandwidth, not a hardware-counter measurement of cache/DRAM.
    """
    active_groups = min(groups, total_m)
    if compute == "mxfp8":
        a_bytes = total_m * k
        a_scale_bytes = total_m * (k // MXFP8_BLOCK_SIZE)
        b_mult_bytes = active_groups * n * (k // NVFP4_BLOCK_SIZE) * 2
        output_bytes = total_m * n * 4
    elif compute == "bf16":
        a_bytes = total_m * k * 2
        a_scale_bytes = 0
        b_mult_bytes = active_groups * n * (k // NVFP4_BLOCK_SIZE) * 2
        output_bytes = total_m * n * 4
    else:
        raise ValueError(f"unknown compute path: {compute}")
    b_bytes = active_groups * n * k // 2
    b_sfmx_bytes = active_groups * n * (k // MXFP8_BLOCK_SIZE)
    total_bytes = (
        a_bytes + a_scale_bytes + b_bytes + b_sfmx_bytes + b_mult_bytes + output_bytes
    )
    flops = 2 * total_m * n * k
    time_s = time_us * 1e-6
    return {
        "total_bytes": total_bytes,
        "bandwidth_gbs": total_bytes / time_s / 1e9,
        "tflops": flops / time_s / 1e12,
    }


def bench_config_projection(
    compute: str,
    name: str,
    groups: int,
    topk: int,
    token_counts: List[int],
    n: int,
    k: int,
    num_warmup: int,
    num_run: int,
    cooldown_before_first: bool = False,
) -> List[dict]:
    """Benchmark a config-derived projection without host-side quantization."""
    from sgl_kernel import nvfp4_woq_bf16_grouped_mm, nvfp4_woq_mxfp8_scaled_grouped_mm

    if cooldown_before_first:
        synchronize_before_shape()
    device = "xpu"
    # Materialize packed weights directly on device to avoid a many-GB FP32
    # host tensor while retaining nontrivial, incompressible payload traffic.
    torch.manual_seed(groups + n + k)
    b_fp4 = torch.randint(0, 256, (groups, n, k // 2), dtype=torch.uint8, device=device)
    sfmx_b = torch.full(
        (groups, k // MXFP8_BLOCK_SIZE, n), UE8M0_BIAS, dtype=torch.uint8, device=device
    )
    c_b = torch.ones(
        (groups, n, k // NVFP4_BLOCK_SIZE), dtype=torch.bfloat16, device=device
    )
    workspace = torch.empty((64 * 1024 * 1024,), dtype=torch.uint8, device=device)

    rows = []
    for index, tokens in enumerate(token_counts):
        if index:
            synchronize_before_shape()
        routed_rows = tokens * topk
        base_rows, extra_rows = divmod(routed_rows, groups)
        generator = torch.Generator().manual_seed(20261015 + routed_rows)
        extra_experts = set(
            torch.randperm(groups, generator=generator)[:extra_rows].tolist()
        )
        counts = [base_rows + int(expert in extra_experts) for expert in range(groups)]
        m_per_group = max(counts)
        total_m = routed_rows
        a_dtype = torch.float8_e4m3fn if compute == "mxfp8" else torch.bfloat16
        torch.manual_seed(total_m + k)
        a_q = torch.randint(-4, 5, (total_m, k), dtype=torch.int8, device=device).to(
            a_dtype
        )
        a_s = (
            torch.full(
                (total_m, k // MXFP8_BLOCK_SIZE),
                UE8M0_BIAS,
                dtype=torch.uint8,
                device=device,
            )
            if compute == "mxfp8"
            else None
        )
        out = torch.empty((total_m, n), dtype=torch.float32, device=device)
        problem_sizes = torch.tensor(
            [[count, n, k] for count in counts], dtype=torch.int32, device=device
        )
        offsets = []
        next_offset = 0
        for count in counts:
            offsets.append(next_offset)
            next_offset += count
        expert_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)

        def grouped_gemm():
            if compute == "mxfp8":
                nvfp4_woq_mxfp8_scaled_grouped_mm(
                    out,
                    a_q,
                    b_fp4,
                    a_s,
                    sfmx_b,
                    c_b,
                    problem_sizes,
                    expert_offsets,
                    workspace,
                )
            else:
                nvfp4_woq_bf16_grouped_mm(
                    out,
                    a_q,
                    b_fp4,
                    sfmx_b,
                    c_b,
                    problem_sizes,
                    expert_offsets,
                    workspace,
                )

        grouped_gemm()
        torch.xpu.synchronize()
        verify_grouped_output(out, a_q, b_fp4, c_b, sfmx_b, counts, compute, a_s)
        samples_us = timeit_samples(grouped_gemm, num_warmup, num_run)
        us = statistics.median(samples_us)
        metrics = effective_metrics(groups, total_m, n, k, us, compute)
        row = {
            "compute": compute,
            "projection": name,
            "tokens": tokens,
            "groups": groups,
            "topk": topk,
            "m_per_group": m_per_group,
            "active_experts": min(groups, routed_rows),
            "n": n,
            "k": k,
            "samples_us": samples_us,
            "p50_us": us,
            "min_us": min(samples_us),
            "max_us": max(samples_us),
            "us": us,
            **metrics,
        }
        rows.append(row)
        print(
            f"{compute:5s} {name:7s} tokens={tokens:5d} "
            f"E={groups} m/g={m_per_group:3d} "
            f"N={n} K={k}  p50={us:9.1f} us "
            f"range=[{min(samples_us):.1f},{max(samples_us):.1f}] us "
            f"raw_us={samples_us}  "
            f"{metrics['bandwidth_gbs']:8.2f} effective GB/s  "
            f"{metrics['tflops']:8.2f} TFLOP/s",
            flush=True,
        )

        del a_q, a_s, out, problem_sizes, expert_offsets

    return rows


def bench_from_config(args) -> None:
    with open(args.config, encoding="utf-8") as config_file:
        config = json.load(config_file)

    moe = config["moe"]
    groups = int(moe["num_experts"])
    topk = int(moe["topk"])
    hidden = int(config["hidden_size"])
    intermediate = int(moe["override_routed_hidden_dim"])
    token_counts = [int(value) for value in args.tokens.split(",")]

    projections = []
    if args.projection in ("gate_up", "both"):
        projections.append(("gate_up", 2 * intermediate, hidden))
    if args.projection in ("down", "both"):
        projections.append(("down", hidden, intermediate))

    print(
        f"Config workload: E={groups}, topk={topk}, hidden={hidden}, "
        f"intermediate={intermediate}"
    )
    if args.compute == "both":
        compute_paths = ("mxfp8", "bf16")
    else:
        compute_paths = (args.compute,)
    first = True
    for compute in compute_paths:
        for name, n, k in projections:
            if not first:
                print()
            bench_config_projection(
                compute,
                name,
                groups,
                topk,
                token_counts,
                n,
                k,
                args.warmup,
                args.runs,
                cooldown_before_first=not first,
            )
            first = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=None)
    ap.add_argument("--serving-batch", type=int, choices=(8,), default=8)
    ap.add_argument("--m", type=int, default=None, help="M per group")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument(
        "--config",
        help="Model config whose MoE dimensions define gate/up and down projections",
    )
    ap.add_argument(
        "--tokens",
        default="8,32768",
        help="Comma-separated token counts for --config mode",
    )
    ap.add_argument(
        "--projection",
        choices=("gate_up", "down", "both"),
        default="both",
        help="Projection(s) to run in --config mode",
    )
    ap.add_argument(
        "--compute",
        choices=("mxfp8", "bf16", "both"),
        default="mxfp8",
        help="NVFP4 compute path(s) to run in --config mode",
    )
    args = ap.parse_args()
    if args.warmup < 0 or args.runs <= 0:
        ap.error("--warmup must be nonnegative and --runs must be positive")
    if args.config:
        try:
            token_counts = [int(value) for value in args.tokens.split(",")]
        except ValueError:
            ap.error("--tokens must be a comma-separated list of integers")
        if not token_counts or any(value <= 0 for value in token_counts):
            ap.error("--tokens must contain positive token counts")
        if any(value not in (8, 8 * 4096) for value in token_counts):
            ap.error(
                "Muse config mode requires B8 decode tokens=8 or ISL4096 prefill tokens=32768"
            )

    if not is_cri_device():
        print("NVFP4 WoQ kernels require an Xe3P (CRI) XPU; skipping.")
        return

    if args.config:
        bench_from_config(args)
        return

    if args.n is not None:
        shapes = [(args.groups or 2, args.m or 64, args.n, args.k or args.n)]
    else:
        shapes = [
            (2, 64, 128, 256),
            (2, 64, 256, 512),
            (4, 64, 256, 512),
            (2, 256, 512, 512),
        ]

    rows: List[dict] = []
    for index, (groups, m, n, k) in enumerate(shapes):
        if index:
            synchronize_before_shape()
        rows.append(bench_one(groups, m, n, k, args.warmup, args.runs))
        r = rows[-1]
        print(
            f"groups={r['groups']} m/g={r['m_per_group']} n={r['n']} k={r['k']}  "
            f"mxfp8={r['mxfp8_us']:.1f}us ({r['mxfp8_tflops']:.2f} TF)  "
            f"woq={r['woq_us']:.1f}us ({r['woq_tflops']:.2f} TF)  "
            f"woq/mxfp8={r['woq_vs_mxfp8']:.2f}x  "
            f"weights {r['weight_mb_nvfp4']:.1f}MB vs {r['weight_mb_mxfp8']:.1f}MB",
            flush=True,
        )


if __name__ == "__main__":
    main()
