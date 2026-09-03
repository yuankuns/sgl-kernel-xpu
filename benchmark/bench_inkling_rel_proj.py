import argparse
from dataclasses import dataclass

import torch
import triton
from sgl_kernel import rel_proj_small_t


@dataclass(frozen=True)
class RelProjCase:
    name: str
    t: int
    h: int
    kv_heads: int
    e: int


def _make_r(case: RelProjCase, d: int) -> torch.Tensor:
    q_width = case.h * 128
    kv_width = 2 * case.kv_heads * 128
    packed = torch.randn(
        case.t, q_width + kv_width + case.h * d, device="xpu", dtype=torch.bfloat16
    )
    return packed[:, q_width + kv_width :].view(case.t, case.h, d)


def _cases(suite: str) -> list[RelProjCase]:
    if suite == "quick":
        return [
            RelProjCase("global_decode_tp2", 1, 24, 2, 1024),
            RelProjCase("global_verify_tp4", 9, 12, 1, 1024),
            RelProjCase("global_extend_tp8", 32, 6, 1, 1024),
            RelProjCase("local_verify_tp4", 9, 12, 1, 512),
        ]
    return [
        RelProjCase("global_decode_tp1", 1, 48, 4, 1024),
        RelProjCase("global_decode_tp2", 1, 24, 2, 1024),
        RelProjCase("global_decode_tp4", 1, 12, 1, 1024),
        RelProjCase("global_decode_tp8", 1, 6, 1, 1024),
        RelProjCase("global_verify_tp2", 9, 24, 2, 1024),
        RelProjCase("global_verify_tp4", 9, 12, 1, 1024),
        RelProjCase("global_verify_tp8", 9, 6, 1, 1024),
        RelProjCase("global_extend_tp2", 32, 24, 2, 1024),
        RelProjCase("global_extend_tp4", 32, 12, 1, 1024),
        RelProjCase("global_extend_tp8", 32, 6, 1, 1024),
        RelProjCase("local_decode_tp4", 1, 12, 1, 512),
        RelProjCase("local_verify_tp4", 9, 12, 1, 512),
        RelProjCase("local_extend_tp4", 32, 12, 1, 512),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["quick", "inkling"], default="quick")
    args = parser.parse_args()

    d = 16
    torch.xpu.set_device(0)
    for case in _cases(args.suite):
        proj = torch.randn(d, case.e, device="xpu", dtype=torch.bfloat16)
        r = _make_r(case, d)
        tau = 1.0 + 0.1 * torch.rand(case.t, device="xpu", dtype=torch.float32)
        out = torch.empty(case.t, case.h, case.e, device="xpu", dtype=torch.bfloat16)
        elapsed_ms, _, _ = triton.testing.do_bench(
            lambda: rel_proj_small_t(r, proj, tau, out),
            quantiles=[0.5, 0.2, 0.8],
        )
        flops = case.t * case.h * d * case.e * 2
        tops = flops / (elapsed_ms * 1e-3) / 1e12
        print(
            f"{case.name:20s} T={case.t:<2d} H={case.h:<2d} D={d} E={case.e} "
            f"time={elapsed_ms * 1000.0:.2f} us throughput={tops:.3f} TOPS"
        )


if __name__ == "__main__":
    main()
