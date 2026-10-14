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
import time
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=(
            "production",
            "prefill-window",
            "decode-window",
            "decode-splits",
            "decode-gqa",
        ),
        required=True,
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument(
        "--timing", choices=("graph", "graph-batched", "eager"), default="graph"
    )
    parser.add_argument("--replays-per-sample", type=int, default=8)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--case-filter", default=".", help="Regex matched against case name"
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1 or args.replays_per_sample < 1:
        parser.error("warmup, iterations, and replays-per-sample must be positive")
    if torch.xpu.get_device_properties(0).device_id != 0x674F:
        parser.error("requires CRI 0x674f")

    selected = [
        case for case in cases_for(args.suite) if re.search(args.case_filter, case.name)
    ]
    if not selected:
        parser.error("case filter selected no shapes")
    for index, case in enumerate(selected):
        if index:
            torch.xpu.synchronize()
            time.sleep(2)
        launch = make_launch(case)
        samples = measure(
            launch,
            args.warmup,
            args.iterations,
            args.timing,
            args.replays_per_sample,
        )
        print(
            json.dumps(
                {
                    "label": args.label,
                    "suite": args.suite,
                    "case": case.__dict__,
                    "device": torch.xpu.get_device_properties(0).name,
                    "dtype": "bfloat16",
                    "method": f"{args.timing}, torch.xpu.Event, {args.replays_per_sample} replays/sample",
                    "samples_ms": samples,
                    "p50_ms": statistics.median(samples),
                    "mad_ms": statistics.median(
                        abs(sample - statistics.median(samples)) for sample in samples
                    ),
                    "min_ms": min(samples),
                    "max_ms": max(samples),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
