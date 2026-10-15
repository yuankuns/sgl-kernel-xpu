#!/usr/bin/env python3
"""Compare the main blockwise op and W4A4 op from identical MXFP4 bytes."""

import argparse
import json
import statistics

import torch
from accuracy_reference import verify_grouped_output
from bench_mxfp4_decode import (
    EXPERTS,
    UPPER_BOUND_ROWS,
    event_samples,
    make_padded_counts,
    make_sweep_counts,
)
from sgl_kernel import (
    moe_grouped_mm_nt_xe35_mxfp4_w4a4,
    mxfp4_blockwise_scaled_grouped_mm,
    mxfp4_moe_scale_to_mn_major,
)


def run_case(n: int, allocated_rows: int, warmup: int, iterations: int, rounds: int):
    k = 4096
    rows_cpu = (
        make_padded_counts(0, None)
        if allocated_rows == UPPER_BOUND_ROWS
        else make_sweep_counts(allocated_rows)
    )
    used = int(rows_cpu.sum())
    rows = rows_cpu.to("xpu")
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int32), rows_cpu.cumsum(0, dtype=torch.int32)[:-1])
    ).to("xpu")
    problems = torch.stack(
        (rows_cpu, torch.full_like(rows_cpu, n), torch.full_like(rows_cpu, k)), 1
    ).to("xpu")
    a = torch.randint(0, 256, (allocated_rows, k // 2), dtype=torch.uint8, device="xpu")
    sa_rm = torch.randint(
        124, 129, (allocated_rows, k // 32), dtype=torch.uint8, device="xpu"
    )
    b = torch.randint(0, 256, (EXPERTS, n, k // 2), dtype=torch.uint8, device="xpu")
    sb_rm = torch.randint(
        124, 129, (EXPERTS, n, k // 32), dtype=torch.uint8, device="xpu"
    )
    sa_mn = torch.zeros_like(sa_rm)
    mxfp4_moe_scale_to_mn_major(sa_mn, sa_rm, rows)
    # The legacy runner checks for a contiguous 3D expert dimension; explicit
    # pointer arrays carry the ragged offsets and strides of this flat buffer.
    assert allocated_rows % EXPERTS == 0
    sa_legacy = sa_mn.view(EXPERTS, k // 32, allocated_rows // EXPERTS)
    sb_mn = sb_rm.transpose(1, 2).contiguous()
    out_branch = torch.empty((allocated_rows, n), dtype=torch.bfloat16, device="xpu")
    out_main = torch.empty((allocated_rows, n), dtype=torch.float32, device="xpu")
    workspace = torch.empty((64 * 1024 * 1024,), dtype=torch.uint8, device="xpu")
    empty_ptrs = torch.empty((0,), dtype=torch.int64, device="xpu")
    starts = offsets.cpu().tolist()
    groups = k // 32

    def pointers(base, strides):
        return torch.tensor(
            [base + offset for offset in strides], dtype=torch.uint64, device="xpu"
        )

    a_ptrs = pointers(a.data_ptr(), [m * (k // 2) for m in starts])
    b_ptrs = pointers(b.data_ptr(), [e * n * (k // 2) for e in range(EXPERTS)])
    out_ptrs = pointers(out_main.data_ptr(), [m * n * 4 for m in starts])
    sa_ptrs = pointers(sa_mn.data_ptr(), [m * groups for m in starts])
    sb_ptrs = pointers(sb_mn.data_ptr(), [e * n * groups for e in range(EXPERTS)])

    def branch():
        moe_grouped_mm_nt_xe35_mxfp4_w4a4(
            out_branch, a, sa_mn, b, sb_mn, rows, n, k, EXPERTS
        )

    def main():
        mxfp4_blockwise_scaled_grouped_mm(
            out_main,
            empty_ptrs,
            empty_ptrs,
            empty_ptrs,
            empty_ptrs,
            empty_ptrs,
            a,
            b,
            sa_rm,
            sb_rm,
            problems,
            offsets,
            workspace,
        )

    def main_bf16():
        main()
        return out_main.to(torch.bfloat16)

    def main_legacy_bf16():
        mxfp4_blockwise_scaled_grouped_mm(
            out_main,
            a_ptrs,
            b_ptrs,
            out_ptrs,
            sa_ptrs,
            sb_ptrs,
            a,
            b,
            sa_legacy,
            sb_mn,
            problems,
            offsets,
            workspace,
        )
        return out_main.to(torch.bfloat16)

    main()
    assert empty_ptrs.numel() == 0, "main changed the empty-pointer sentinel"
    branch()
    torch.xpu.synchronize()
    verify_grouped_output(out_branch, a, sa_mn, b, sb_mn, rows_cpu)
    got = out_branch[:used].float()
    want = out_main[:used].clone()
    main_legacy_bf16()
    torch.xpu.synchronize()
    torch.testing.assert_close(out_main[:used], want, rtol=0, atol=0)
    torch.testing.assert_close(got, want, rtol=5e-3, atol=5e-3)
    abs_error = (got - want).abs()
    relative_error = abs_error / want.abs().clamp_min(1e-6)
    assert torch.isfinite(got).all() and torch.isfinite(want).all()
    accuracy = {
        "kind": "accuracy",
        "n": n,
        "k": k,
        "allocated_rows": allocated_rows,
        "used_rows": used,
        "active_experts": int((rows_cpu > 0).sum()),
        "elements": got.numel(),
        "max_abs_vs_main_fp32": abs_error.max().item(),
        "max_rel_vs_main_fp32": relative_error.max().item(),
        "main_fp32_abs_p50": want.abs().median().item(),
        "main_fp32_abs_max": want.abs().max().item(),
    }
    print(json.dumps(accuracy), flush=True)
    paths = (
        ("branch_gemm", branch),
        ("main_legacy_bf16", main_legacy_bf16),
        ("main_full_bf16", main_bf16),
    )
    for round_idx in range(rounds):
        order = paths if round_idx % 2 == 0 else tuple(reversed(paths))
        for label, call in order:
            samples = event_samples(call, warmup, iterations)
            ordered = sorted(samples)
            print(
                json.dumps(
                    {
                        "round": round_idx,
                        "path": label,
                        "n": n,
                        "k": k,
                        "experts": EXPERTS,
                        "allocated_rows": allocated_rows,
                        "used_rows": used,
                        "median_ms": statistics.median(samples),
                        "p20_ms": ordered[int(0.2 * (len(samples) - 1))],
                        "p80_ms": ordered[int(0.8 * (len(samples) - 1))],
                        "min_ms": min(samples),
                        "max_ms": max(samples),
                        "samples_ms": samples,
                    }
                ),
                flush=True,
            )
            torch.xpu.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serving-batch", type=int, choices=(8,), default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--rows", type=int, nargs="+", default=[UPPER_BOUND_ROWS])
    args = parser.parse_args()
    for allocated_rows in args.rows:
        for n in (4096, 8192):
            run_case(n, allocated_rows, args.warmup, args.iterations, args.rounds)
