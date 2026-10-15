import argparse
import statistics

import torch
from sgl_kernel import (
    mxfp4_moe_scale_to_mn_major,
    sgl_padded_gather_per_token_group_quant_fp4,
    sgl_per_token_group_quant_fp4,
)


def measure(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    samples = []
    for _ in range(repeat):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def baseline(hidden_states, row_map, rows_per_expert):
    gathered = hidden_states.index_select(0, row_map)
    output_q, scales_rm = sgl_per_token_group_quant_fp4(gathered, group_size=32)
    scales_mn = torch.empty_like(scales_rm)
    mxfp4_moe_scale_to_mn_major(scales_mn, scales_rm, rows_per_expert)
    return output_q, scales_mn


def gather_quant_relayout(hidden_states, row_map, rows_per_expert):
    output_q, scales_rm = sgl_padded_gather_per_token_group_quant_fp4(
        hidden_states, row_map, group_size=32
    )
    scales_mn = torch.empty_like(scales_rm)
    mxfp4_moe_scale_to_mn_major(scales_mn, scales_rm, rows_per_expert)
    return output_q, scales_mn


def gather_quant_only(hidden_states, row_map):
    return sgl_padded_gather_per_token_group_quant_fp4(
        hidden_states, row_map, group_size=32
    )


def relayout_only(scales_rm, rows_per_expert):
    scales_mn = torch.empty_like(scales_rm)
    mxfp4_moe_scale_to_mn_major(scales_mn, scales_rm, rows_per_expert)
    return scales_mn


def swiglu_quant_relayout(gate_up, rows_per_expert):
    output_q, scales_rm = sgl_per_token_group_quant_fp4(
        gate_up, group_size=32, fuse_silu_and_mul=True
    )
    scales_mn = torch.empty_like(scales_rm)
    mxfp4_moe_scale_to_mn_major(scales_mn, scales_rm, rows_per_expert)
    return output_q, scales_mn


def logical_bytes(rows, hidden, path):
    # Count each logical stream once. The baseline first reads and writes a
    # gathered BF16 surface, then reads it for quantization. Both paths write
    # packed FP4 values and one UE8M0 byte per group of 32 values. The legacy
    # paths additionally read and rewrite the row-major scale surface.
    bf16_bytes = 2
    fp4_bytes = 0.5
    scale_bytes = 1 / 32
    if path == "baseline":
        streams_per_element = 3 * bf16_bytes + fp4_bytes + 3 * scale_bytes
    elif path == "gather_relayout":
        streams_per_element = bf16_bytes + fp4_bytes + 3 * scale_bytes
    elif path == "swiglu_relayout":
        streams_per_element = 2 * bf16_bytes + fp4_bytes + 3 * scale_bytes
    else:
        raise ValueError(f"unknown path: {path}")
    return rows * hidden * streams_per_element


def routing_metadata(routed_rows, experts, alignment, rows, device):
    active_experts = min(experts, routed_rows)
    base, remainder = divmod(routed_rows, active_experts)
    counts = torch.zeros(experts, dtype=torch.int32)
    counts[:active_experts] = base
    counts[:remainder] += 1
    padded_counts = (
        torch.div(counts + alignment - 1, alignment, rounding_mode="floor") * alignment
    )
    used_rows = int(padded_counts.sum())
    assert used_rows <= rows
    return padded_counts.to(device), used_rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare MXFP4 index-select and padded-gather quant paths."
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=None,
        help="unrouted token counts; rows are tokens*topk plus alignment slack",
    )
    parser.add_argument("--serving-batch", type=int, default=8)
    parser.add_argument("--isl", type=int, default=4096)
    parser.add_argument("--osl", type=int, default=16)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--alignment", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()
    if any(
        value <= 0
        for value in (args.serving_batch, args.isl, args.osl, args.tp, args.ep)
    ):
        parser.error("serving batch, ISL, OSL, TP and EP must be positive")
    if args.tokens is None:
        args.tokens = [args.serving_batch, args.serving_batch * args.isl]

    if any(token_count <= 0 for token_count in args.tokens):
        parser.error("--tokens values must be positive")
    if (
        args.topk <= 0
        or args.experts <= 0
        or args.alignment <= 0
        or args.topk > args.experts
    ):
        parser.error("--topk and --alignment must be positive, with topk <= experts")
    if args.hidden <= 0 or args.hidden % 32:
        parser.error("--hidden must be positive and divisible by 32")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be nonnegative and --repeat must be positive")

    if not torch.xpu.is_available():
        raise RuntimeError("an XPU is required")
    device = torch.xpu.get_device_properties(0)
    print(
        f"device={device.name} device_id=0x{device.device_id:04x} "
        f"B={args.serving_batch} ISL={args.isl} OSL={args.osl} TP={args.tp} EP={args.ep}"
    )

    for token_count in args.tokens:
        routed_rows = token_count * args.topk
        rows = routed_rows + (args.alignment - 1) * min(args.experts, routed_rows)
        hidden_states = torch.randn(
            (token_count, args.hidden), dtype=torch.bfloat16, device="xpu"
        )
        gate_up = torch.randn(
            (rows, 2 * args.hidden), dtype=torch.bfloat16, device="xpu"
        )
        padded_counts, used_rows = routing_metadata(
            routed_rows,
            args.experts,
            args.alignment,
            rows,
            hidden_states.device,
        )
        row_map_i32 = torch.arange(rows, dtype=torch.int32, device="xpu") % token_count
        expected_q, expected_s = baseline(hidden_states, row_map_i32, padded_counts)
        gather_q, gather_s = gather_quant_relayout(
            hidden_states, row_map_i32, padded_counts
        )
        used_scale_bytes = used_rows * (args.hidden // 32)
        if not torch.equal(expected_q, gather_q) or not torch.equal(
            expected_s.view(-1)[:used_scale_bytes],
            gather_s.view(-1)[:used_scale_bytes],
        ):
            raise AssertionError(
                f"optimized output differs for tokens={token_count}, rows={rows}"
            )
        del expected_q, expected_s, gather_q, gather_s
        torch.xpu.synchronize()

        baseline_ms = measure(
            lambda: baseline(hidden_states, row_map_i32, padded_counts),
            args.warmup,
            args.repeat,
        )
        gather_relayout_ms = measure(
            lambda: gather_quant_relayout(hidden_states, row_map_i32, padded_counts),
            args.warmup,
            args.repeat,
        )
        _, scales_rm = gather_quant_only(hidden_states, row_map_i32)
        gather_quant_ms = measure(
            lambda: gather_quant_only(hidden_states, row_map_i32),
            args.warmup,
            args.repeat,
        )
        relayout_ms = measure(
            lambda: relayout_only(scales_rm, padded_counts),
            args.warmup,
            args.repeat,
        )
        swiglu_relayout_ms = measure(
            lambda: swiglu_quant_relayout(gate_up, padded_counts),
            args.warmup,
            args.repeat,
        )
        baseline_gbps = logical_bytes(rows, args.hidden, "baseline") / baseline_ms / 1e6
        gather_relayout_gbps = (
            logical_bytes(rows, args.hidden, "gather_relayout")
            / gather_relayout_ms
            / 1e6
        )
        swiglu_relayout_gbps = (
            logical_bytes(rows, args.hidden, "swiglu_relayout")
            / swiglu_relayout_ms
            / 1e6
        )

        print(
            f"tokens={token_count} routed_rows={routed_rows} "
            f"used_rows={used_rows} padded_upper_rows={rows}: "
            f"index_select+quant+relayout={baseline_ms:.4f} ms/"
            f"{baseline_gbps:.1f} logical GB/s, "
            f"gather+quant+relayout={gather_relayout_ms:.4f} ms/"
            f"{gather_relayout_gbps:.1f} logical GB/s, "
            f"gather+quant={gather_quant_ms:.4f} ms, "
            f"scale_relayout={relayout_ms:.4f} ms, "
            f"speedup={baseline_ms / gather_relayout_ms:.3f}x"
        )
        print(
            f"  swiglu+quant+relayout={swiglu_relayout_ms:.4f} ms/"
            f"{swiglu_relayout_gbps:.1f} logical GB/s"
        )

        del gate_up, hidden_states, row_map_i32, scales_rm
        del padded_counts
        torch.xpu.synchronize()


if __name__ == "__main__":
    main()
