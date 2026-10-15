"""Measure Muse Spark FMHA and the neighboring Xe3 dispatch shapes.

Run against an installed sgl_kernel package. Each JSON line includes the raw
device-event samples so separate builds can be compared without rounding loss.
Graph-batched timing captures repeated calls in one graph to keep short kernels
from being limited by Python replay submission.
"""

import argparse
import json
import re
import statistics
from dataclasses import dataclass

import sgl_kernel  # noqa: F401 - registers the production attention op
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache


@dataclass(frozen=True)
class Case:
    name: str
    requests: int
    q_len: int
    k_len: int
    heads_q: int = 128
    heads_kv: int = 16
    window_left: int = -1
    page_size: int = 64
    num_splits: int = 0


def cases_for(suite):
    if suite == "production":
        for requests in (1, 7, 8):
            for window in (-1, 2047):
                yield Case(
                    f"prefill-wave{requests}-w{window}",
                    requests,
                    4096,
                    4096,
                    window_left=window,
                )
        for k_len in range(4097, 4113):
            for window in (-1, 2047):
                yield Case(
                    f"decode-osl{k_len - 4096}-w{window}",
                    8,
                    1,
                    k_len,
                    window_left=window,
                )
    elif suite == "prefill-window":
        for requests in (1, 7, 8):
            for page in (64, 128):
                for window in (2046, 2047, 2048):
                    yield Case(
                        f"prefill-wave{requests}-p{page}-w{window}",
                        requests,
                        4096,
                        4096,
                        window_left=window,
                        page_size=page,
                    )
    elif suite == "decode-window":
        for k_len in (4097, 4112):
            for page in (64, 128):
                for window in (-1, 2046, 2047, 2048):
                    yield Case(
                        f"decode-k{k_len}-p{page}-w{window}",
                        8,
                        1,
                        k_len,
                        window_left=window,
                        page_size=page,
                    )
    elif suite == "decode-splits":
        for requests in (1, 3, 4, 8):
            for window in (-1, 2047):
                for splits in (0, 1, 2, 4, 8):
                    yield Case(
                        f"decode-b{requests}-w{window}-s{splits}",
                        requests,
                        1,
                        4097,
                        window_left=window,
                        num_splits=splits,
                    )
    elif suite == "decode-gqa":
        for requests in (3, 4, 8):
            for group in (7, 8, 9):
                for window in (-1, 2047):
                    yield Case(
                        f"decode-b{requests}-gqa{group}-w{window}",
                        requests,
                        1,
                        4097,
                        heads_q=16 * group,
                        window_left=window,
                    )
    else:
        raise ValueError(f"unknown suite: {suite}")


def make_launch(case):
    torch.manual_seed(42)
    device, dtype, dim = "xpu", torch.bfloat16, 64
    pages_per_seq = (case.k_len + case.page_size - 1) // case.page_size
    num_pages = case.requests * pages_per_seq
    q = torch.randn(
        case.requests * case.q_len, case.heads_q, dim, device=device, dtype=dtype
    )
    k = torch.randn(
        num_pages, case.page_size, case.heads_kv, dim, device=device, dtype=dtype
    )
    v = torch.randn_like(k)
    page_table = torch.randperm(num_pages, device=device, dtype=torch.int32).view(
        case.requests, pages_per_seq
    )
    cache_seqlens = torch.full(
        (case.requests,), case.k_len, device=device, dtype=torch.int32
    )
    cu_seqlens_q = (
        torch.arange(case.requests + 1, device=device, dtype=torch.int32) * case.q_len
    )
    out = torch.empty_like(q)

    def launch():
        return flash_attn_with_kvcache(
            q,
            k,
            v,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=case.q_len,
            max_seqlen_k=case.k_len,
            softmax_scale=dim**-0.5,
            causal=True,
            window_size=(case.window_left, 0) if case.window_left >= 0 else (-1, -1),
            rotary_interleaved=False,
            num_splits=case.num_splits,
            out=out,
        )

    return launch


def measure(launch, warmup, iterations, timing, replays_per_sample):
    for _ in range(warmup):
        launch()
    torch.xpu.synchronize()
    if timing in ("graph", "graph-batched"):
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            if timing == "graph-batched":
                for _ in range(replays_per_sample):
                    launch()
            else:
                launch()
        torch.xpu.synchronize()
        run = graph.replay
    else:
        run = launch
    for _ in range(warmup):
        run()
    torch.xpu.synchronize()

    samples = []
    for _ in range(iterations):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        if timing == "graph-batched":
            run()
        else:
            for _ in range(replays_per_sample):
                run()
        end.record()
        torch.xpu.synchronize()
        samples.append(start.elapsed_time(end) / replays_per_sample)
    return samples


@torch.inference_mode()
def check_tail_shapes():
    """Kernel accuracy only: paged GQA and partial query/key tiles.

    The production Xe3.5 Muse-Spark path uses BF16.
    """
    for dtype in (torch.bfloat16,):
        for length in (33, 129, 257):
            torch.xpu.synchronize()
            torch.manual_seed(42)
            batch = 8
            page, hq, hkv, dim = 64, 128, 16, 64
            pages_per_seq = (length + page - 1) // page
            pages = batch * pages_per_seq
            q = torch.randn(batch * length, hq, dim, device="xpu", dtype=dtype)
            k = torch.randn(pages, page, hkv, dim, device="xpu", dtype=dtype)
            v = torch.randn_like(k)
            table = (
                torch.randperm(pages, device="xpu")
                .to(torch.int32)
                .view(batch, pages_per_seq)
            )
            cu = torch.arange(batch + 1, device="xpu", dtype=torch.int32) * length
            lengths = torch.full((batch,), length, dtype=torch.int32, device="xpu")
            # Independent FP32 CPU reference, using the actual shuffled pages.
            order = table.cpu().long()
            keys = k.cpu()[order].reshape(batch, -1, hkv, dim)[:, :length].float()
            values = v.cpu()[order].reshape(batch, -1, hkv, dim)[:, :length].float()
            keys = keys.repeat_interleave(hq // hkv, dim=2).transpose(1, 2)
            values = values.repeat_interleave(hq // hkv, dim=2).transpose(1, 2)
            queries = q.cpu().float().view(batch, length, hq, dim).transpose(1, 2)
            scores = queries @ keys.transpose(2, 3)
            scores *= dim**-0.5
            scores.masked_fill_(
                torch.ones(length, length, dtype=torch.bool).triu(1), -float("inf")
            )
            expected = (scores.softmax(dim=-1) @ values).transpose(1, 2)
            expected = expected.reshape(batch * length, hq, dim)
            actual = flash_attn_with_kvcache(
                q=q,
                k_cache=k,
                v_cache=v,
                page_table=table,
                cache_seqlens=lengths,
                cu_seqlens_q=cu,
                max_seqlen_q=length,
                softmax_scale=dim**-0.5,
                causal=True,
                window_size=(-1, -1),
            )
            torch.testing.assert_close(
                actual.float().cpu(), expected, atol=5e-3, rtol=5e-3
            )
            print(
                json.dumps(
                    {
                        "accuracy": "PASS",
                        "length": length,
                        "dtype": str(dtype),
                        "reference": "FP32 CPU causal GQA",
                        "path": "flash_attn_with_kvcache",
                    }
                ),
                flush=True,
            )


def legacy_cases():
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
    return {
        name: Case(name, *shape[:3], window_left=shape[3], page_size=shape[4])
        for name, shape in cases.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-batch", type=int, default=8)
    parser.add_argument(
        "--mode", choices=("prefill", "decode", "dispatch"), default="dispatch"
    )
    parser.add_argument(
        "--suite",
        choices=(
            "production",
            "prefill-window",
            "decode-window",
            "decode-splits",
            "decode-gqa",
        ),
        default="production",
    )
    parser.add_argument(
        "--cases",
        help="Comma-separated legacy case names; retains all B8 wave/page/window cases",
    )
    parser.add_argument("--sweep-local", action="store_true")
    parser.add_argument("--check-tails", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument(
        "--timing",
        choices=("graph", "graph-batched", "eager", "triton"),
        default="graph",
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=500)
    parser.add_argument("--replays-per-sample", type=int, default=8)
    parser.add_argument("--label", default="production")
    parser.add_argument(
        "--case-filter", default=".", help="Regex matched against case name"
    )
    args = parser.parse_args()
    if args.serving_batch != 8:
        parser.error("Muse-Spark serving workload requires B=8")
    if (
        min(
            args.warmup,
            args.iterations,
            args.replays_per_sample,
            args.warmup_ms,
            args.rep_ms,
        )
        < 1
    ):
        parser.error(
            "warmup, iterations, replays, and millisecond timing budgets must be positive"
        )
    if args.cases or args.sweep_local:
        available = legacy_cases()
        names = (
            args.cases.split(",")
            if args.cases
            else [
                f"prefill-b8-wave{wave}-swa{window}-p{page}"
                for wave in (1, 7)
                for page in (64, 128)
                for window in (0, 63, 511, 1023, 2046, 2047, 2048, 4095)
            ]
        )
        if any(name not in available for name in names):
            parser.error("unknown legacy case name")
        selected = [available[name] for name in names]
    else:
        selected = list(cases_for(args.suite))
    selected = [
        case
        for case in selected
        if re.search(args.case_filter, case.name)
        and (args.mode == "dispatch" or (case.q_len > 1) == (args.mode == "prefill"))
    ]
    if not selected:
        parser.error("mode/case filter selected no shapes")
    if not sgl_kernel.is_xe3_arch():
        parser.error("requires Xe3.5 attention capability")
    if args.check_tails:
        check_tail_shapes()
    for case in selected:
        launch = make_launch(case)
        if args.timing == "triton":
            import triton.testing

            p50_ms, p20_ms, p80_ms = triton.testing.do_bench(
                launch,
                warmup=args.warmup_ms,
                rep=args.rep_ms,
                quantiles=[0.5, 0.2, 0.8],
            )
            summary = {
                "p50_ms": p50_ms,
                "p20_ms": p20_ms,
                "p80_ms": p80_ms,
                "warmup_ms": args.warmup_ms,
                "rep_ms": args.rep_ms,
            }
        else:
            samples = measure(
                launch,
                args.warmup,
                args.iterations,
                args.timing,
                args.replays_per_sample,
            )
            p50_ms = statistics.median(samples)
            summary = {
                "samples_ms": samples,
                "p50_ms": p50_ms,
                "mad_ms": statistics.median(abs(sample - p50_ms) for sample in samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
            }
        effective_keys = (
            min(case.k_len, case.window_left + 1)
            if case.window_left >= 0
            else case.k_len
        )
        logical_kv_bytes = case.requests * effective_keys * case.heads_kv * 64 * 2 * 2
        print(
            json.dumps(
                {
                    "label": args.label,
                    "mode": args.mode,
                    "suite": args.suite,
                    "serving_batch": 8,
                    "case": case.__dict__,
                    "device": torch.xpu.get_device_properties(0).name,
                    "dtype": "bfloat16",
                    "path": "flash_attn_with_kvcache",
                    "method": f"{args.timing}, XPU device timing, {args.replays_per_sample} replays/sample",
                    "logical_kv_gbps": logical_kv_bytes / (p50_ms * 1e6),
                    **summary,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
