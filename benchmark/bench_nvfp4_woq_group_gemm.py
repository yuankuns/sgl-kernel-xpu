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
import importlib
import json
import pathlib
import statistics
import sysconfig
from typing import List, Tuple

import torch
from accuracy_reference import verify_grouped_output

MXFP8_BLOCK_SIZE = 32
NVFP4_BLOCK_SIZE = 16
FP8_E4M3_MAX = 448.0
E2M1_MAX = 6.0
UE8M0_BIAS = 127
CTIMER_DIRECTORY = pathlib.Path(__file__).resolve().parent / "ctimer"
CTIMER_EXTENSION = (
    CTIMER_DIRECTORY / f"nvfp4_ctimer_ext{sysconfig.get_config_var('EXT_SUFFIX')}"
)
CTIMER_WORKSPACE_BYTES = 128 * 1024 * 1024


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


def load_extension():
    try:
        return importlib.import_module("nvfp4_ctimer_ext")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"missing installed nvfp4_ctimer_ext; build {CTIMER_EXTENSION} "
            "with build_nvfp4_woq_ctimer.sh and install it into Python purelib"
        ) from error


def make_inputs(
    ext,
    groups: int,
    m: int,
    n: int,
    k: int,
    suite: str,
    randomize: bool = False,
    routed_rows: int | None = None,
    match_weights: bool = False,
    shuffle_experts: bool = False,
    skip_bf16_copy: bool = False,
):
    dev = "xpu"
    total_m = groups * m if routed_rows is None else routed_rows
    if randomize:
        b_nvfp4 = torch.randint(
            0, 256, (groups, n, k // 2), dtype=torch.uint8, device=dev
        )
    else:
        b_nvfp4 = torch.zeros((groups, n, k // 2), dtype=torch.uint8, device=dev)
    sfmx_b = torch.full((groups, k // 32, n), 127, dtype=torch.uint8, device=dev)
    c_b = torch.ones((groups, n, k // 16), dtype=torch.bfloat16, device=dev)
    base_rows, extra_rows = divmod(total_m, groups)
    if shuffle_experts:
        generator = torch.Generator().manual_seed(20261015 + total_m)
        extra_experts = set(
            torch.randperm(groups, generator=generator)[:extra_rows].tolist()
        )
    else:
        extra_experts = set(range(extra_rows))
    counts = [base_rows + int(i in extra_experts) for i in range(groups)]
    offsets = []
    next_offset = 0
    for count in counts:
        offsets.append(next_offset)
        next_offset += count
    problem_sizes = torch.tensor(
        [[count, n, k] for count in counts], dtype=torch.int32, device=dev
    )
    expert_offsets = torch.tensor(offsets, dtype=torch.int32, device=dev)
    rows_per_expert = torch.tensor(counts, dtype=torch.int32, device=dev)
    output = torch.empty((total_m, n), dtype=torch.float32, device=dev)
    workspace = torch.empty((CTIMER_WORKSPACE_BYTES,), dtype=torch.uint8, device=dev)
    inputs = {
        "output": output,
        "b_nvfp4": b_nvfp4,
        "sfmx_b": sfmx_b,
        "c_b": c_b,
        "problem_sizes": problem_sizes,
        "expert_offsets": expert_offsets,
        "rows_per_expert": rows_per_expert,
        "workspace": workspace,
    }
    if suite in ("mxfp8", "all"):
        scales_a = torch.full((total_m, k // 32), 127, dtype=torch.uint8, device=dev)
        if randomize:
            a_mxfp8 = torch.randint(
                -4, 5, (total_m, k), dtype=torch.int8, device=dev
            ).to(torch.float8_e4m3fn)
        else:
            a_mxfp8 = torch.zeros((total_m, k), dtype=torch.float8_e4m3fn, device=dev)
        if match_weights:
            b_mxfp8 = torch.empty((groups, n, k), dtype=torch.float8_e4m3fn, device=dev)
        elif randomize:
            b_mxfp8 = torch.randint(
                -4, 5, (groups, n, k), dtype=torch.int8, device=dev
            ).to(torch.float8_e4m3fn)
        else:
            b_mxfp8 = torch.zeros((groups, n, k), dtype=torch.float8_e4m3fn, device=dev)
        inputs.update(
            a_mxfp8=a_mxfp8,
            scales_a=scales_a,
            b_mxfp8=b_mxfp8,
            scales_b_mxfp8=torch.full(
                (groups, n, k // 32), 127, dtype=torch.uint8, device=dev
            ),
            output_mxfp8=torch.empty((total_m, n), dtype=torch.float32, device=dev),
            empty_ptrs=torch.empty((0,), dtype=torch.int64, device=dev),
            zero_stride=torch.empty((0,), dtype=torch.int64, device=dev),
            zero_layout=torch.empty((0,), dtype=torch.int64, device=dev),
        )
        if match_weights:
            lut = torch.tensor(
                [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                dtype=torch.float32,
                device=dev,
            )
            # Convert one expert at a time, outside the timed region. Both
            # kernels then multiply the same logical weights.
            for expert in range(groups):
                packed = b_nvfp4[expert]
                inputs["b_mxfp8"][expert, :, 0::2].copy_(
                    lut[(packed & 15).long()].to(torch.float8_e4m3fn)
                )
                inputs["b_mxfp8"][expert, :, 1::2].copy_(
                    lut[(packed >> 4).long()].to(torch.float8_e4m3fn)
                )
    if suite in ("bf16", "all"):
        inputs.update(
            a_bf16=(
                torch.randint(-4, 5, (total_m, k), dtype=torch.int8, device=dev).to(
                    torch.bfloat16
                )
                if randomize
                else torch.zeros((total_m, k), dtype=torch.bfloat16, device=dev)
            ),
            output_bf16=torch.empty((total_m, n), dtype=torch.bfloat16, device=dev),
        )
        if not skip_bf16_copy:
            inputs["b_bf16"] = (
                torch.randint(-4, 5, (groups, k, n), dtype=torch.int8, device=dev).to(
                    torch.bfloat16
                )
                if randomize
                else torch.zeros((groups, k, n), dtype=torch.bfloat16, device=dev)
            )
    torch.xpu.synchronize()
    return inputs


def logical_bytes(path: str, groups: int, m: int, n: int, k: int) -> float:
    activations = groups * m * k
    # Sparse routed rows only load weights for experts with a nonzero count.
    weights = min(groups, int(round(groups * m))) * n * k
    output = groups * m * n
    if path == "main_mxfp8":
        return activations * (1 + 1 / 32) + weights * (1 + 1 / 32) + output * 4
    if path.startswith("nvfp4_mxfp8"):
        return (
            activations * (1 + 1 / 32)
            + weights * (1 / 2 + 1 / 32 + 2 / 16)
            + output * 4
        )
    if path == "nvfp4_bf16":
        return activations * 2 + weights * (1 / 2 + 1 / 32 + 2 / 16) + output * 4
    if path == "bf16_copy":
        return activations * 2 + weights * 2 + output * 2
    raise ValueError(f"unknown path {path}")


def print_results(results, groups: int, m: int, n: int, k: int):
    flops = 2.0 * groups * m * n * k
    print("path                    device_ms  effective_GB/s  TFLOP/s  cute_ms")
    for name, result in results:
        milliseconds = result["device_span_ms"]
        bandwidth = logical_bytes(name, groups, m, n, k) / milliseconds / 1e6
        tflops = flops / milliseconds / 1e9
        print(
            f"{name:23s}  {milliseconds:9.4f}  {bandwidth:14.2f}  "
            f"{tflops:7.2f}  {result['cute_ms']:7.4f}"
        )


def verify_bf16_reference(x, groups: int, n: int, k: int):
    if groups * n * k > 1 << 24:
        raise ValueError("--verify-bf16 is for small directed shapes")
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device="xpu",
    )
    total_rows = x["a_bf16"].size(0)
    counts = x["problem_sizes"][:, 0].cpu().tolist()
    offset = 0
    for expert, rows in enumerate(counts):
        if not rows:
            continue
        packed = x["b_nvfp4"][expert]
        codes = torch.stack((packed & 15, packed >> 4), dim=-1)
        weight = lut[codes.reshape(n, k).long()]
        reference = x["a_bf16"][offset : offset + rows].float().cpu() @ weight.cpu().T
        actual = x["output"][offset : offset + rows].cpu()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        offset += rows
    print(f"bf16_reference exact_match rows={total_rows} E={groups} N={n} K={k}")


def run_case_once(ext, args, case: str, m: int, n: int, k: int, x):
    suite = "bf16" if args.only_bf16 else args.suite
    results = []
    if suite in ("mxfp8", "all"):
        mx = ext.benchmark_mxfp8(
            x["output_mxfp8"],
            x["a_mxfp8"],
            x["b_mxfp8"],
            x["scales_a"],
            x["scales_b_mxfp8"],
            x["empty_ptrs"],
            x["zero_stride"],
            x["zero_layout"],
            x["problem_sizes"],
            x["expert_offsets"],
            x["workspace"],
            args.warmup,
            args.iterations,
        )
        results.append(("main_mxfp8", mx))
        torch.xpu.synchronize()
        verify_grouped_output(
            x["output_mxfp8"],
            x["a_mxfp8"],
            x["b_mxfp8"],
            None,
            x["scales_b_mxfp8"],
            x["rows_per_expert"],
            "plain_mxfp8",
            x["scales_a"],
        )
        woq_mxfp8 = ext.benchmark_nvfp4(
            x["output"],
            x["a_mxfp8"],
            x["b_nvfp4"],
            x["scales_a"],
            x["sfmx_b"],
            x["c_b"],
            x["problem_sizes"],
            x["expert_offsets"],
            x["workspace"],
            args.warmup,
            args.iterations,
        )
        results.append(("nvfp4_mxfp8", woq_mxfp8))
        torch.xpu.synchronize()
        verify_grouped_output(
            x["output"],
            x["a_mxfp8"],
            x["b_nvfp4"],
            x["c_b"],
            x["sfmx_b"],
            x["rows_per_expert"],
            "mxfp8",
            x["scales_a"],
        )
        if args.verify_main:
            torch.xpu.synchronize()
            difference = (x["output"] - x["output_mxfp8"]).abs()
            mismatches = torch.count_nonzero(x["output"] != x["output_mxfp8"]).item()
            max_abs = difference.max().item()
            print(
                f"main_equivalence mismatches={mismatches}/{difference.numel()} "
                f"max_abs={max_abs}",
                flush=True,
            )
            if mismatches:
                raise AssertionError("NVFP4 MXFP8 differs from main MXFP8")

    if suite in ("bf16", "all"):
        if not args.skip_bf16_copy:
            bf16 = ext.benchmark_bf16(
                x["output_bf16"],
                x["a_bf16"],
                x["b_bf16"],
                x["rows_per_expert"],
                args.warmup,
                args.iterations,
            )
        woq_bf16 = ext.benchmark_nvfp4_bf16(
            x["output"],
            x["a_bf16"],
            x["b_nvfp4"],
            x["sfmx_b"],
            x["c_b"],
            x["problem_sizes"],
            x["expert_offsets"],
            x["workspace"],
            args.warmup,
            args.iterations,
        )
        if not args.skip_bf16_copy:
            results.append(("bf16_copy", bf16))
            torch.xpu.synchronize()
            verify_grouped_output(
                x["output_bf16"],
                x["a_bf16"],
                x["b_bf16"].transpose(1, 2),
                None,
                None,
                x["rows_per_expert"],
                "plain_bf16",
            )
        results.append(("nvfp4_bf16", woq_bf16))
        torch.xpu.synchronize()
        verify_grouped_output(
            x["output"],
            x["a_bf16"],
            x["b_nvfp4"],
            x["c_b"],
            x["sfmx_b"],
            x["rows_per_expert"],
            "bf16",
        )
        if args.verify_bf16:
            verify_bf16_reference(x, args.groups, n, k)

    print(
        f"case={case} shape=E{args.groups} M{m} N{n} K{k} "
        f"routed_rows={args.routed_rows or args.groups * m}"
    )
    print_results(
        results,
        args.groups,
        (args.routed_rows / args.groups) if args.routed_rows is not None else m,
        n,
        k,
    )


def run_case(ext, args, case: str, m: int, n: int, k: int):
    suite = "bf16" if args.only_bf16 else args.suite
    x = make_inputs(
        ext,
        args.groups,
        m,
        n,
        k,
        suite,
        randomize=args.randomize,
        routed_rows=args.routed_rows,
        match_weights=args.match_weights,
        shuffle_experts=args.shuffle_experts,
        skip_bf16_copy=args.skip_bf16_copy,
    )
    for repeat in range(args.repeats):
        if repeat:
            synchronize_before_shape()
        print(f"repeat={repeat + 1}/{args.repeats}", flush=True)
        run_case_once(ext, args, case, m, n, k, x)


def run_ctimer_backend(args):
    args.groups = args.groups or 2
    args.n = args.n or 256
    args.k = args.k or 512
    args.iterations = args.runs
    args.suite = args.suite or ("all" if args.compute == "both" else args.compute)
    if args.only_bf16:
        args.suite = "bf16"
    if args.routed_rows is not None and args.routed_rows <= 0:
        raise ValueError("--routed-rows must be positive")
    if args.match_weights and (not args.randomize or args.suite == "bf16"):
        raise ValueError("--match-weights requires --randomize and an MXFP8 suite")
    if args.verify_main and not args.match_weights:
        raise ValueError("--verify-main requires --match-weights")
    if args.verify_bf16 and not args.randomize:
        raise ValueError("--verify-bf16 requires --randomize")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    if not torch.xpu.is_available():
        raise RuntimeError("XPU is unavailable")
    ext = load_extension()
    print(f"device={torch.xpu.get_device_name()}")
    cases = []
    if args.config:
        with open(args.config, encoding="utf-8") as config_file:
            config = json.load(config_file)
        args.groups = int(config["moe"]["num_experts"])
        topk = int(config["moe"]["topk"])
        hidden = int(config["hidden_size"])
        intermediate = int(config["moe"]["override_routed_hidden_dim"])
        projections = []
        if args.projection in ("gate_up", "both"):
            projections.append(("gate_up", 2 * intermediate, hidden))
        if args.projection in ("down", "both"):
            projections.append(("down", hidden, intermediate))
        for token_count in (int(value) for value in args.tokens.split(",")):
            args.routed_rows = token_count * topk
            for name, output_width, hidden_width in projections:
                run_case(
                    ext,
                    args,
                    f"{name}_tokens{token_count}",
                    (args.routed_rows + args.groups - 1) // args.groups,
                    output_width,
                    hidden_width,
                )
        return
    if args.shapes:
        for value in args.shapes.split(","):
            try:
                m, n, k = (int(dim) for dim in value.split(":"))
            except ValueError as exc:
                raise ValueError(f"invalid shape {value!r}; expected M:N:K") from exc
            cases.append(("sweep", m, n, k))
    else:
        if args.case in ("decode", "both"):
            cases.append(("decode", args.m or args.decode_m, args.n, args.k))
        if args.case in ("prefill", "both"):
            cases.append(("prefill", args.m or args.prefill_m, args.n, args.k))
    for index, (case, m, n, k) in enumerate(cases):
        if index:
            print()
        if index:
            synchronize_before_shape()
        run_case(ext, args, case, m, n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=None)
    ap.add_argument("--serving-batch", type=int, choices=(8,), default=8)
    ap.add_argument("--reference-threads", type=int, default=10)
    ap.add_argument(
        "--timing-backend", choices=("device-event", "ctimer"), default="device-event"
    )
    ap.add_argument("--m", type=int, default=None, help="M per group")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--runs", "--iterations", dest="runs", type=int)
    ap.add_argument("--case", choices=("decode", "prefill", "both"), default="both")
    ap.add_argument("--decode-m", type=int, default=1)
    ap.add_argument("--prefill-m", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--suite", choices=("mxfp8", "bf16", "all"))
    ap.add_argument("--randomize", action="store_true", default=True)
    ap.add_argument("--zero-inputs", action="store_false", dest="randomize")
    ap.add_argument("--match-weights", action="store_true")
    ap.add_argument("--verify-main", action="store_true")
    ap.add_argument("--verify-bf16", action="store_true")
    ap.add_argument("--routed-rows", type=int)
    ap.add_argument("--shuffle-experts", action="store_true")
    ap.add_argument("--skip-bf16-copy", action="store_true")
    ap.add_argument("--shapes", help="CTimer comma-separated M:N:K tuples")
    ap.add_argument("--only-bf16", action="store_true")
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
    if args.reference_threads <= 0:
        ap.error("--reference-threads must be positive")
    torch.set_num_threads(args.reference_threads)
    if args.warmup is None:
        args.warmup = 1 if args.timing_backend == "ctimer" else 2
    if args.runs is None:
        args.runs = 3 if args.timing_backend == "ctimer" else 5
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
    if args.timing_backend == "ctimer":
        if args.config and args.shapes:
            ap.error("--config and --shapes are mutually exclusive")
        run_ctimer_backend(args)
        return
    if (
        args.suite is not None
        or args.only_bf16
        or args.verify_main
        or args.verify_bf16
        or args.match_weights
        or args.routed_rows is not None
        or args.shapes
        or args.shuffle_experts
        or args.skip_bf16_copy
        or not args.randomize
        or args.case != "both"
    ):
        ap.error("CTimer-specific options require --timing-backend ctimer")
    if not args.config and args.compute != "mxfp8":
        ap.error(
            "device-event BF16 comparison requires --config; use --timing-backend ctimer for explicit shapes"
        )

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
