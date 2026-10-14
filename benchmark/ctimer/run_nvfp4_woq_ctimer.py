#!/usr/bin/env python3
"""Compare NVFP4 WoQ with stock BF16/MXFP8 using CUTLASS GPU_Clock."""

import argparse
import importlib.util
import os
import pathlib
import sysconfig
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
EXTENSION = HERE / f"nvfp4_ctimer_ext{sysconfig.get_config_var('EXT_SUFFIX')}"
WORKSPACE_BYTES = 128 * 1024 * 1024


def cool_down_before_shape() -> None:
    torch.xpu.synchronize()
    time.sleep(2)


def load_extension():
    if not EXTENSION.exists():
        raise RuntimeError(f"missing {EXTENSION}; run build_nvfp4_woq_ctimer.sh first")
    spec = importlib.util.spec_from_file_location("nvfp4_ctimer_ext", EXTENSION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_inputs(
    ext, groups: int, m: int, n: int, k: int, suite: str, randomize: bool = False
):
    dev = "xpu"
    total_m = groups * m
    if randomize:
        b_nvfp4 = torch.randint(
            0, 256, (groups, n, k // 2), dtype=torch.uint8, device=dev
        )
    else:
        b_nvfp4 = torch.zeros((groups, n, k // 2), dtype=torch.uint8, device=dev)
    sfmx_b = torch.full((groups, k // 32, n), 127, dtype=torch.uint8, device=dev)
    c_b = torch.ones((groups, n, k // 16), dtype=torch.bfloat16, device=dev)
    problem_sizes = torch.tensor([[m, n, k]] * groups, dtype=torch.int32, device=dev)
    expert_offsets = torch.arange(groups, dtype=torch.int32, device=dev) * m
    rows_per_expert = torch.full((groups,), m, dtype=torch.int32, device=dev)
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
        scales_a_mn = torch.empty(
            (total_m + 3 * groups, k // 32), dtype=torch.uint8, device=dev
        )
        ext.prepare_mxfp8_scales(scales_a_mn, scales_a, rows_per_expert)
        if randomize:
            a_mxfp8 = torch.randint(
                -4, 5, (total_m, k), dtype=torch.int8, device=dev
            ).to(torch.float8_e4m3fn)
        else:
            a_mxfp8 = torch.zeros((total_m, k), dtype=torch.float8_e4m3fn, device=dev)
        inputs.update(
            a_mxfp8=a_mxfp8,
            scales_a=scales_a,
            scales_a_mn=scales_a_mn,
            b_mxfp8=torch.zeros((groups, n, k), dtype=torch.float8_e4m3fn, device=dev),
            scales_b_mxfp8_mn=torch.full(
                (groups, k // 32, n), 127, dtype=torch.uint8, device=dev
            ),
            output_mxfp8=torch.empty((total_m, n), dtype=torch.bfloat16, device=dev),
        )
    if suite in ("bf16", "all"):
        inputs.update(
            a_bf16=torch.zeros((total_m, k), dtype=torch.bfloat16, device=dev),
            b_bf16=torch.zeros((groups, n, k), dtype=torch.bfloat16, device=dev),
            output_bf16=torch.empty((total_m, n), dtype=torch.bfloat16, device=dev),
        )
    torch.xpu.synchronize()
    return inputs


def logical_bytes(path: str, groups: int, m: int, n: int, k: int) -> float:
    activations = groups * m * k
    weights = groups * n * k
    output = groups * m * n
    if path == "mxfp8":
        return activations * (1 + 1 / 32) + weights * (1 + 1 / 32) + output * 2
    if path.startswith("nvfp4_mxfp8"):
        return (
            activations * (1 + 1 / 32)
            + weights * (1 / 2 + 1 / 32 + 2 / 16)
            + output * 4
        )
    if path == "nvfp4_bf16":
        return activations * 2 + weights * (1 / 2 + 1 / 32 + 2 / 16) + output * 4
    if path == "bf16":
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


def verify_cache_modes(
    ext, groups: int, m: int, n: int, k: int, modes: tuple[str, ...]
):
    if "register" not in modes:
        raise ValueError("verification modes must include register")
    x = make_inputs(ext, groups, m, n, k, "mxfp8", randomize=True)
    outputs = {}
    for mode in modes:
        print(f"running verification mode={mode}", flush=True)
        os.environ["SGLANG_NVFP4_WOQ_WIDE_CACHE"] = mode
        x["output"].fill_(float("nan"))
        ext.run_nvfp4(
            x["output"],
            x["a_mxfp8"],
            x["b_nvfp4"],
            x["scales_a"],
            x["sfmx_b"],
            x["c_b"],
            x["problem_sizes"],
            x["expert_offsets"],
            x["workspace"],
        )
        torch.xpu.synchronize()
        outputs[mode] = x["output"].clone()

    reference = outputs["register"]
    print(f"verification shape=E{groups} M{m} N{n} K{k}")
    for mode in modes:
        if mode == "register":
            continue
        difference = (outputs[mode] - reference).abs()
        mismatches = torch.count_nonzero(outputs[mode] != reference).item()
        print(
            f"{mode:8s} max_abs={difference.max().item():.6g} "
            f"mismatches={mismatches}/{reference.numel()}"
        )


def run_case(ext, args, case: str, m: int, n: int, k: int):
    suite = "bf16" if args.only_bf16 else args.suite
    x = make_inputs(ext, args.groups, m, n, k, suite)
    results = []
    if suite in ("mxfp8", "all"):
        mx = ext.benchmark_mxfp8_stock(
            x["output_mxfp8"],
            x["a_mxfp8"],
            x["b_mxfp8"],
            x["scales_a_mn"],
            x["scales_b_mxfp8_mn"],
            x["rows_per_expert"],
            args.warmup,
            args.iterations,
        )
        results.append(("mxfp8", mx))
        cache_modes = ("register", "slm") if args.cache_sweep else (args.wide_cache,)
        for mode in cache_modes:
            os.environ["SGLANG_NVFP4_WOQ_WIDE_CACHE"] = mode
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
            results.append((f"nvfp4_mxfp8_{mode}", woq_mxfp8))

    if suite in ("bf16", "all"):
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
        results.extend((("bf16", bf16), ("nvfp4_bf16", woq_bf16)))

    print(f"case={case} shape=E{args.groups} M{m} N{n} K{k}")
    print_results(results, args.groups, m, n, k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("decode", "prefill", "both"), default="both")
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--decode-m", type=int, default=1, help="decode M per group")
    parser.add_argument("--prefill-m", type=int, default=64, help="prefill M per group")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
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
    parser.add_argument(
        "--wide-cache",
        choices=("register", "l1", "slm"),
        default="register",
        help="select the NVFP4-to-BF16 staging location (l1 is experimental)",
    )
    parser.add_argument(
        "--cache-sweep",
        action="store_true",
        help="benchmark the runnable register and SLM staging modes",
    )
    parser.add_argument(
        "--verify-cache-modes",
        action="store_true",
        help="compare all cache modes on random exact-representable inputs",
    )
    parser.add_argument(
        "--verify-modes",
        default="register,slm",
        help="comma-separated modes used by --verify-cache-modes",
    )
    parser.add_argument(
        "--force-tile",
        choices=(
            "auto",
            "m8n128",
            "m8n256",
            "m32n256",
            "m64n256",
            "m256n256",
        ),
        default="auto",
        help="force an NVFP4 workgroup tile for tuning",
    )
    args = parser.parse_args()
    if os.environ.get("BATCHES", "8") != "8":
        parser.error("Muse-Spark serving workload requires BATCHES=8")
    os.environ["SGLANG_NVFP4_WOQ_FORCE_TILE"] = args.force_tile

    if not torch.xpu.is_available():
        raise RuntimeError("XPU is unavailable")
    ext = load_extension()
    print(f"device={torch.xpu.get_device_name()}")
    if args.verify_cache_modes:
        verify_modes = tuple(args.verify_modes.split(","))
        invalid_modes = set(verify_modes) - {"register", "l1", "slm"}
        if invalid_modes:
            raise ValueError(f"invalid verification modes: {sorted(invalid_modes)}")
        verify_cache_modes(
            ext,
            args.groups,
            args.prefill_m,
            args.n,
            args.k,
            verify_modes,
        )
        print()
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
        if index or args.verify_cache_modes:
            cool_down_before_shape()
        run_case(ext, args, case, m, n, k)


if __name__ == "__main__":
    main()
