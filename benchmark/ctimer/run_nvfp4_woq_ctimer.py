#!/usr/bin/env python3
"""Compare NVFP4 WoQ with stock BF16/MXFP8 using CUTLASS GPU_Clock."""

import argparse
import importlib.util
import pathlib
import sysconfig

import torch

HERE = pathlib.Path(__file__).resolve().parent
EXTENSION = HERE / f"nvfp4_ctimer_ext{sysconfig.get_config_var('EXT_SUFFIX')}"
WORKSPACE_BYTES = 128 * 1024 * 1024


def synchronize_before_shape() -> None:
    torch.xpu.synchronize()


def load_extension():
    if not EXTENSION.exists():
        raise RuntimeError(f"missing {EXTENSION}; run build_nvfp4_woq_ctimer.sh first")
    spec = importlib.util.spec_from_file_location("nvfp4_ctimer_ext", EXTENSION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_inputs(
    ext, groups: int, m: int, n: int, k: int, suite: str, randomize: bool = False,
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
    workspace = torch.empty((WORKSPACE_BYTES,), dtype=torch.uint8, device=dev)
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
            b_mxfp8 = torch.empty(
                (groups, n, k), dtype=torch.float8_e4m3fn, device=dev
            )
        elif randomize:
            b_mxfp8 = torch.randint(
                -4, 5, (groups, n, k), dtype=torch.int8, device=dev
            ).to(torch.float8_e4m3fn)
        else:
            b_mxfp8 = torch.zeros(
                (groups, n, k), dtype=torch.float8_e4m3fn, device=dev
            )
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
                [0, 0.5, 1, 1.5, 2, 3, 4, 6,
                 -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                dtype=torch.float32, device=dev,
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
                torch.randint(-4, 5, (total_m, k), dtype=torch.int8, device=dev)
                .to(torch.bfloat16)
                if randomize
                else torch.zeros((total_m, k), dtype=torch.bfloat16, device=dev)
            ),
            output_bf16=torch.empty((total_m, n), dtype=torch.bfloat16, device=dev),
        )
        if not skip_bf16_copy:
            inputs["b_bf16"] = (
                torch.randint(-4, 5, (groups, n, k), dtype=torch.int8, device=dev)
                .to(torch.bfloat16)
                if randomize
                else torch.zeros((groups, n, k), dtype=torch.bfloat16, device=dev)
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
        [0, 0.5, 1, 1.5, 2, 3, 4, 6,
         -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32, device="xpu",
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
        reference = x["a_bf16"][offset:offset + rows].float() @ weight.T
        actual = x["output"][offset:offset + rows]
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
        if args.verify_main:
            torch.xpu.synchronize()
            difference = (x["output"] - x["output_mxfp8"]).abs()
            mismatches = torch.count_nonzero(
                x["output"] != x["output_mxfp8"]
            ).item()
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
        results.append(("nvfp4_bf16", woq_bf16))
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
        n, k,
    )


def run_case(ext, args, case: str, m: int, n: int, k: int):
    suite = "bf16" if args.only_bf16 else args.suite
    x = make_inputs(
        ext, args.groups, m, n, k, suite,
        randomize=args.randomize, routed_rows=args.routed_rows,
        match_weights=args.match_weights,
        shuffle_experts=args.shuffle_experts,
        skip_bf16_copy=args.skip_bf16_copy,
    )
    for repeat in range(args.repeats):
        if repeat:
            synchronize_before_shape()
        print(f"repeat={repeat + 1}/{args.repeats}", flush=True)
        run_case_once(ext, args, case, m, n, k, x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("decode", "prefill", "both"), default="both")
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument(
        "--randomize", action="store_true",
        help="Use nontrivial payloads for memory traffic measurements",
    )
    parser.add_argument(
        "--match-weights", action="store_true",
        help="Expand the same E2M1 weights for the main MXFP8 path before timing",
    )
    parser.add_argument(
        "--verify-main", action="store_true",
        help="Assert exact output agreement with main MXFP8 on matched weights",
    )
    parser.add_argument(
        "--verify-bf16", action="store_true",
        help="Assert exact BF16 WOQ agreement with direct dequantization",
    )
    parser.add_argument(
        "--routed-rows", type=int,
        help="Total routed rows, including sparse expert counts (Muse B8 is 64)",
    )
    parser.add_argument(
        "--shuffle-experts", action="store_true",
        help="Spread sparse routed rows across experts with a fixed seed",
    )
    parser.add_argument("--decode-m", type=int, default=1, help="decode M per group")
    parser.add_argument("--prefill-m", type=int, default=64, help="prefill M per group")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--skip-bf16-copy", action="store_true",
        help="Omit non-equivalent BF16 copy-policy weights to reduce peak memory",
    )
    parser.add_argument(
        "--suite",
        choices=("mxfp8", "bf16", "all"),
        default="mxfp8",
        help="allocate and compare only the selected compute paths",
    )
    parser.add_argument(
        "--shapes",
        help="comma-separated M:N:K tuples; overrides the individual shape options",
    )
    parser.add_argument(
        "--only-bf16",
        action="store_true",
        help="skip the unchanged MXFP8 reference paths",
    )
    args = parser.parse_args()
    if args.routed_rows is not None and args.routed_rows <= 0:
        parser.error("--routed-rows must be positive")
    if args.match_weights and (not args.randomize or args.suite == "bf16"):
        parser.error("--match-weights requires --randomize and an MXFP8 suite")
    if args.verify_main and not args.match_weights:
        parser.error("--verify-main requires --match-weights")
    if args.verify_bf16 and not args.randomize:
        parser.error("--verify-bf16 requires --randomize")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    if not torch.xpu.is_available():
        raise RuntimeError("XPU is unavailable")
    ext = load_extension()
    print(f"device={torch.xpu.get_device_name()}")
    cases = []
    if args.shapes:
        for value in args.shapes.split(","):
            try:
                m, n, k = (int(dim) for dim in value.split(":"))
            except ValueError as exc:
                raise ValueError(f"invalid shape {value!r}; expected M:N:K") from exc
            cases.append(("sweep", m, n, k))
    else:
        if args.case in ("decode", "both"):
            cases.append(("decode", args.decode_m, args.n, args.k))
        if args.case in ("prefill", "both"):
            cases.append(("prefill", args.prefill_m, args.n, args.k))
    for index, (case, m, n, k) in enumerate(cases):
        if index:
            print()
        if index:
            synchronize_before_shape()
        run_case(ext, args, case, m, n, k)


if __name__ == "__main__":
    main()
