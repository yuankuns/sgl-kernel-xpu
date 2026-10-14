"""Paired XPU device-time probes for routing and MXFP4 quantization."""

import argparse
import json
import statistics
import time

import torch
from sgl_kernel import sgl_per_token_group_quant_fp4, silu_and_mul, topk_sigmoid

GRAPH_REPLAY = 1


def check_route(weights, ids, logits, bias, multiplier):
    scores = torch.sigmoid((logits * multiplier).float())
    rank_scores = scores + bias
    assert bool(((ids >= 0) & (ids < logits.size(1))).all())
    assert bool((torch.diff(torch.sort(ids, dim=-1).values, dim=-1) != 0).all())
    torch.testing.assert_close(
        rank_scores.gather(1, ids.long()),
        torch.topk(rank_scores, ids.size(1), dim=-1, sorted=True).values,
        rtol=0,
        atol=3e-4,
    )
    selected = scores.gather(1, ids.long())
    torch.testing.assert_close(
        weights,
        selected / (selected.sum(dim=-1, keepdim=True) + 1e-15),
        rtol=3e-4,
        atol=3e-4,
    )


def bench(name, fn, warmup, rep):
    global GRAPH_REPLAY
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    if GRAPH_REPLAY > 1:
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            fn()
        torch.xpu.synchronize()
    samples = []
    for _ in range(rep):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        if GRAPH_REPLAY > 1:
            for _ in range(GRAPH_REPLAY):
                graph.replay()
        else:
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / GRAPH_REPLAY)
    ordered = sorted(samples)
    print(
        json.dumps(
            {
                "name": name,
                "graph_replay": GRAPH_REPLAY,
                "samples_us": samples,
                "p20_us": ordered[round(0.2 * (len(ordered) - 1))],
                "p50_us": statistics.median(ordered),
                "p80_us": ordered[round(0.8 * (len(ordered) - 1))],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", action="store_true")
    parser.add_argument("--triton-route", action="store_true")
    parser.add_argument("--quant-only", action="store_true")
    parser.add_argument("--lane-only", action="store_true")
    parser.add_argument("--fused-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=21)
    parser.add_argument("--graph-replay", type=int, default=1)
    parser.add_argument("--shapes", nargs="+", type=int, default=[64, 128, 252, 256, 260, 368, 372, 376, 512, 1024])
    parser.add_argument("--hidden", type=int, default=4096)
    args = parser.parse_args()
    global GRAPH_REPLAY
    GRAPH_REPLAY = args.graph_replay
    if args.branch:
        from sgl_kernel import (
            muse_spark_moe_routing,
            mxfp4_moe_scale_to_mn_major,
            sgl_padded_gather_per_token_group_quant_fp4,
        )

    torch.manual_seed(7)
    props = torch.xpu.get_device_properties(0)
    print(json.dumps({"device": props.name, "device_id": props.device_id, "branch": args.branch, "hidden": args.hidden}), flush=True)
    for i, rows in enumerate(args.shapes):
        if args.graph_replay > 1:
            torch.manual_seed(7000 + rows)
        source_rows = 32768 if rows >= 32768 * 8 else 8
        x = torch.randn((source_rows, args.hidden), device="xpu", dtype=torch.bfloat16)
        index = torch.arange(rows, device="xpu", dtype=torch.int32) % source_rows
        gathered = x.index_select(0, index)
        q_ref, s_ref = sgl_per_token_group_quant_fp4(gathered)
        if args.branch:
            q, s = sgl_padded_gather_per_token_group_quant_fp4(x, index)
            assert torch.equal(q, q_ref) and torch.equal(s, s_ref)
        if not args.lane_only:
            bench(f"index_quant_rows{rows}", lambda: sgl_per_token_group_quant_fp4(x.index_select(0, index)), args.warmup, args.rep)
        if not args.fused_only:
            bench(f"plain_quant_rows{rows}", lambda: sgl_per_token_group_quant_fp4(gathered), args.warmup, args.rep)
        if args.branch:
            if not args.fused_only:
                bench(f"gather_quant_rows{rows}", lambda: sgl_padded_gather_per_token_group_quant_fp4(x, index), args.warmup, args.rep)
            routed_rows = source_rows * 8
            active_experts = min(256, routed_rows)
            counts = torch.zeros(256, dtype=torch.int32)
            base, rem = divmod(routed_rows, active_experts)
            counts[:active_experts] = base
            counts[:rem] += 1
            counts = (counts + 3) // 4 * 4
            used_rows = int(counts.sum())
            if used_rows <= rows and not args.lane_only:
                counts_xpu = counts.to("xpu")

                def index_quant_relayout():
                    q, s = sgl_per_token_group_quant_fp4(x.index_select(0, index))
                    s_mn = torch.empty_like(s)
                    mxfp4_moe_scale_to_mn_major(s_mn, s, counts_xpu)
                    return q, s_mn

                def gather_quant_relayout():
                    q, s = sgl_padded_gather_per_token_group_quant_fp4(x, index)
                    s_mn = torch.empty_like(s)
                    mxfp4_moe_scale_to_mn_major(s_mn, s, counts_xpu)
                    return q, s_mn

                iq, is_mn = index_quant_relayout()
                gq, gs_mn = gather_quant_relayout()
                assert torch.equal(iq, gq) and torch.equal(
                    is_mn.flatten()[: used_rows * (args.hidden // 32)],
                    gs_mn.flatten()[: used_rows * (args.hidden // 32)],
                )
                bench(
                    f"index_quant_relayout_rows{rows}",
                    index_quant_relayout,
                    args.warmup,
                    args.rep,
                )
                bench(
                    f"gather_quant_relayout_rows{rows}",
                    gather_quant_relayout,
                    args.warmup,
                    args.rep,
                )
            gate_up = torch.randn((rows, 2 * args.hidden), device="xpu", dtype=torch.bfloat16)
            activated = silu_and_mul(gate_up)
            fused_q, fused_s = sgl_per_token_group_quant_fp4(
                gate_up, fuse_silu_and_mul=True
            )
            separate_q, separate_s = sgl_per_token_group_quant_fp4(activated)
            assert torch.equal(fused_q, separate_q) and torch.equal(
                fused_s, separate_s
            )
            if not args.lane_only:
                bench(
                    f"separate_silu_quant_rows{rows}",
                    lambda: sgl_per_token_group_quant_fp4(silu_and_mul(gate_up)),
                    args.warmup,
                    args.rep,
                )
            bench(
                f"fused_silu_quant_rows{rows}",
                lambda: sgl_per_token_group_quant_fp4(
                    gate_up, fuse_silu_and_mul=True
                ),
                args.warmup,
                args.rep,
            )
        if not args.quant_only and rows in (256, 512, 32768 * 8 + 768):
            route_tokens = source_rows if rows != 512 else 16
            if args.graph_replay > 1:
                torch.manual_seed(17000 + route_tokens)
            logits = torch.randn((route_tokens, 256), device="xpu", dtype=torch.float32)
            bias = torch.randn((256,), device="xpu", dtype=torch.float32)
            multiplier = 0.3535533905932738
            def native():
                scores = torch.sigmoid((logits * multiplier).float())
                ids = torch.topk(scores + bias, 8, dim=-1, sorted=True)[1]
                weights = scores.gather(1, ids)
                return weights / (weights.sum(dim=-1, keepdim=True) + 1e-15), ids
            main_weights = torch.empty((route_tokens, 8), device="xpu", dtype=torch.float32)
            main_ids = torch.empty((route_tokens, 8), device="xpu", dtype=torch.int32)
            def main_topk():
                scaled_logits = logits * multiplier
                topk_sigmoid(main_weights, main_ids, scaled_logits, True, bias)
                return main_weights, main_ids
            main_w, main_i = main_topk()
            check_route(main_w, main_i, logits, bias, multiplier)
            if args.branch:
                weights = torch.empty((route_tokens, 8), device="xpu", dtype=torch.float32)
                ids = torch.empty((route_tokens, 8), device="xpu", dtype=torch.int32)
                def fused():
                    muse_spark_moe_routing(weights, ids, logits, bias, multiplier, 1.0, 1e-15)
                fused()
                check_route(weights, ids, logits, bias, multiplier)
            bench(f"native_route_tokens{route_tokens}", native, args.warmup, args.rep)
            bench(f"main_topk_route_tokens{route_tokens}", main_topk, args.warmup, args.rep)
            if args.triton_route:
                try:
                    from sglang_meta_common.ops.fused_muse_spark_routing import (
                        fused_muse_spark_routing,
                    )
                    triton_weights = torch.empty_like(main_weights)
                    triton_ids = torch.empty_like(main_ids)

                    def triton_route():
                        return fused_muse_spark_routing(
                            logits,
                            bias,
                            8,
                            1.0,
                            router_output_multiplier=multiplier,
                            router_score_normalization="sigmoid",
                            router_score_division_eps=1e-15,
                            topk_weights=triton_weights,
                            topk_ids=triton_ids,
                        )

                    triton_w, triton_i = triton_route()
                    check_route(triton_w, triton_i, logits, bias, multiplier)
                    bench(
                        f"framework_auto_route_tokens{route_tokens}",
                        triton_route,
                        args.warmup,
                        args.rep,
                    )
                except Exception as exc:
                    print(
                        json.dumps(
                            {"name": f"framework_auto_route_tokens{route_tokens}", "error": repr(exc)}
                        ),
                        flush=True,
                    )
            if args.branch:
                bench(f"fused_route_tokens{route_tokens}", fused, args.warmup, args.rep)
        torch.xpu.synchronize()
        if i + 1 < len(args.shapes):
            time.sleep(2)


if __name__ == "__main__":
    main()
