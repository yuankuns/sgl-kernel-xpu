import argparse
from dataclasses import dataclass

import torch
from sgl_kernel import rel_proj_small_t


@dataclass(frozen=True)
class RelProjCase:
    name: str
    t: int
    h: int
    kv_heads: int


def _make_r(case: RelProjCase, d: int) -> torch.Tensor:
    q_width = case.h * 128
    kv_width = 2 * case.kv_heads * 128
    packed = torch.randn(
        case.t, q_width + kv_width + case.h * d, device="xpu", dtype=torch.bfloat16
    )
    return packed[:, q_width + kv_width :].view(case.t, case.h, d)


def _time_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) / iterations


def _time_ms_graph(fn, warmup: int, iterations: int, reps: int = 15) -> float:
    """Time `fn` from a captured graph replay.

    At production Inkling shapes this kernel runs in a few microseconds, which is
    less than the eager per-call dispatch cost, so `_time_ms` measures the Python
    and runtime submit path instead of the kernel. Replaying `iterations` captured
    launches keeps the submit cost out of the timed region.
    """
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        for _ in range(iterations):
            fn()
    for _ in range(5):
        graph.replay()
    torch.xpu.synchronize()

    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    best = float("inf")
    for _ in range(reps):
        start.record()
        graph.replay()
        end.record()
        torch.xpu.synchronize()
        best = min(best, start.elapsed_time(end) / iterations)
    return best


def _cases(suite: str) -> list[RelProjCase]:
    if suite == "quick":
        return [
            RelProjCase("decode_tp2", 1, 24, 2),
            RelProjCase("verify_tp4", 9, 12, 1),
            RelProjCase("extend_tp8", 32, 6, 1),
        ]
    return [
        RelProjCase("decode_tp1", 1, 48, 4),
        RelProjCase("decode_tp2", 1, 24, 2),
        RelProjCase("decode_tp4", 1, 12, 1),
        RelProjCase("decode_tp8", 1, 6, 1),
        RelProjCase("verify_tp2", 9, 24, 2),
        RelProjCase("verify_tp4", 9, 12, 1),
        RelProjCase("verify_tp8", 9, 6, 1),
        RelProjCase("extend_tp2", 32, 24, 2),
        RelProjCase("extend_tp4", 32, 12, 1),
        RelProjCase("extend_tp8", 32, 6, 1),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["quick", "inkling"], default="quick")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=["eager", "graph"],
        default="graph",
        help="graph replays captured launches so the kernel, not the dispatch "
        "path, is what gets timed",
    )
    args = parser.parse_args()

    timer = _time_ms_graph if args.mode == "graph" else _time_ms

    d, e = 16, 1024
    torch.xpu.set_device(0)
    proj = torch.randn(d, e, device="xpu", dtype=torch.bfloat16)
    for case in _cases(args.suite):
        r = _make_r(case, d)
        tau = 1.0 + 0.1 * torch.rand(case.t, device="xpu", dtype=torch.float32)
        out = torch.empty(case.t, case.h, e, device="xpu", dtype=torch.bfloat16)
        elapsed_ms = timer(
            lambda: rel_proj_small_t(r, proj, tau, out),
            args.warmup,
            args.iterations,
        )
        flops = case.t * case.h * d * e * 2
        tops = flops / (elapsed_ms * 1e-3) / 1e12
        print(
            f"{case.name:12s} T={case.t:<2d} H={case.h:<2d} D={d} E={e} "
            f"time={elapsed_ms * 1000.0:.2f} us throughput={tops:.3f} TOPS"
        )


if __name__ == "__main__":
    main()
