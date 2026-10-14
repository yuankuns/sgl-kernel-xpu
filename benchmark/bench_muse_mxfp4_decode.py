#!/usr/bin/env python3
"""Benchmark Muse Spark B=8 MXFP4 MoE GEMMs through the production op."""

import argparse
import json
import os
import statistics
import time

import torch
from sgl_kernel import is_xe3_arch, moe_grouped_mm_nt_xe35_mxfp4_w4a4

BATCH = 8
TOPK = 8
EXPERTS = 256
ALIGNMENT = 4
HIDDEN = 4096
INTERMEDIATE = 4096
UPPER_BOUND_ROWS = BATCH * TOPK + (ALIGNMENT - 1) * BATCH * TOPK


def make_padded_counts(seed: int, active_experts: int | None) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if active_experts is not None:
        if not 1 <= active_experts <= BATCH * TOPK:
            raise ValueError(f"active_experts must be in [1, {BATCH * TOPK}]")
        experts = torch.randperm(EXPERTS, generator=generator)[:active_experts]
        routed = experts[torch.arange(BATCH * TOPK, dtype=torch.int64) % active_experts]
        counts = torch.bincount(routed, minlength=EXPERTS).to(torch.int32)
        return (
            torch.div(counts + ALIGNMENT - 1, ALIGNMENT, rounding_mode="floor")
            * ALIGNMENT
        )

    # top-k never selects the same expert twice for one token.
    routed = torch.stack(
        [torch.randperm(EXPERTS, generator=generator)[:TOPK] for _ in range(BATCH)]
    )
    counts = torch.bincount(routed.flatten(), minlength=EXPERTS).to(torch.int32)
    return (
        torch.div(counts + ALIGNMENT - 1, ALIGNMENT, rounding_mode="floor") * ALIGNMENT
    )


def make_sweep_counts(total_rows: int, experts: int = EXPERTS) -> torch.Tensor:
    if total_rows <= 0 or total_rows % ALIGNMENT:
        raise ValueError("sweep rows must be a positive multiple of 4")
    if experts <= 0:
        raise ValueError("experts must be positive")
    quads = total_rows // ALIGNMENT
    counts = torch.full((experts,), quads // experts, dtype=torch.int32)
    counts[: quads % experts] += 1
    return counts * ALIGNMENT


def event_samples(launch, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        launch()
    torch.xpu.synchronize()
    starts = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        launch()
        end.record()
    torch.xpu.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def logical_bytes(active_experts: int, padded_rows: int, n: int, k: int) -> int:
    a = padded_rows * (k // 2 + k // 32)
    weights = active_experts * n * (k // 2 + k // 32)
    output = padded_rows * n * 2
    return a + weights + output


def benchmark_projection(
    name: str,
    n: int,
    k: int,
    counts: torch.Tensor,
    warmup: int,
    iterations: int,
    allocated_rows: int,
) -> None:
    device = "xpu"
    experts = counts.numel()
    active_experts = int((counts != 0).sum())
    padded_rows = int(counts.sum())
    rows = counts.to(device)

    a = torch.randint(
        0, 256, (allocated_rows, k // 2), dtype=torch.uint8, device=device
    )
    a_scale = torch.randint(
        112, 143, (allocated_rows, k // 32), dtype=torch.uint8, device=device
    )
    weights = torch.randint(
        0, 256, (experts, n, k // 2), dtype=torch.uint8, device=device
    )
    weight_scales = torch.randint(
        112, 143, (experts, k // 32, n), dtype=torch.uint8, device=device
    )
    output = torch.empty((allocated_rows, n), dtype=torch.bfloat16, device=device)

    def launch():
        moe_grouped_mm_nt_xe35_mxfp4_w4a4(
            output,
            a,
            a_scale,
            weights,
            weight_scales,
            rows,
            n,
            k,
            experts,
        )

    bytes_per_call = logical_bytes(active_experts, padded_rows, n, k)
    samples = event_samples(launch, warmup, iterations)
    median_ms = statistics.median(samples)
    bandwidth = bytes_per_call / (median_ms * 1e6)
    print(
        json.dumps(
            {
                "projection": name,
                "policy": "auto",
                "active_experts": active_experts,
                "experts": experts,
                "padded_rows": padded_rows,
                "allocated_rows": allocated_rows,
                "n": n,
                "k": k,
                "median_ms": median_ms,
                "min_ms": min(samples),
                "max_ms": max(samples),
                "p20_ms": sorted(samples)[int(0.2 * (len(samples) - 1))],
                "p80_ms": sorted(samples)[int(0.8 * (len(samples) - 1))],
                "logical_GBs": bandwidth,
                "samples_ms": samples,
            }
        ),
        flush=True,
    )

    del a, a_scale, weights, weight_scales, output, rows
    torch.xpu.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allocated-rows", type=int, default=UPPER_BOUND_ROWS)
    parser.add_argument("--sweep-rows", type=int)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--active-experts",
        type=int,
        help="spread the 64 routed rows evenly over this many experts",
    )
    parser.add_argument(
        "--projection", choices=("gate_up", "down", "both"), default="both"
    )
    args = parser.parse_args()

    if os.environ.get("BATCHES", "8") != "8":
        parser.error("Muse-Spark serving workload requires BATCHES=8")
    if not torch.xpu.is_available() or not is_xe3_arch():
        raise RuntimeError("Muse MXFP4 decode benchmark requires a CRI XPU")
    counts = (
        make_sweep_counts(args.sweep_rows)
        if args.sweep_rows is not None
        else make_padded_counts(args.seed, args.active_experts)
    )
    if args.allocated_rows < int(counts.sum()):
        parser.error("allocated rows must cover all routed rows")
    if args.allocated_rows % ALIGNMENT:
        parser.error("allocated rows must be a multiple of 4")
    projections = []
    if args.projection in ("gate_up", "both"):
        projections.append(("gate_up", 2 * INTERMEDIATE, HIDDEN))
    if args.projection in ("down", "both"):
        projections.append(("down", HIDDEN, INTERMEDIATE))

    for index, projection in enumerate(projections):
        benchmark_projection(
            *projection, counts, args.warmup, args.iterations, args.allocated_rows
        )
        if index + 1 < len(projections):
            time.sleep(2)


if __name__ == "__main__":
    main()
