"""Device-event comparison of fused post norm/residual/pre norm and main PyTorch."""

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import statistics
import time

import sgl_kernel  # noqa: F401
import torch


def inputs(rows: int, hidden: int):
    torch.manual_seed(1000 + rows + hidden)
    branch = torch.randn(rows, hidden, device="xpu", dtype=torch.bfloat16)
    stream = torch.randn(rows, hidden, device="xpu", dtype=torch.float32)
    post_weight = torch.randn(hidden, device="xpu", dtype=torch.bfloat16)
    pre_weight = torch.randn(hidden, device="xpu", dtype=torch.bfloat16)
    alpha = torch.rand(hidden, device="xpu")
    beta = torch.rand(hidden, device="xpu")
    output = torch.empty_like(branch)
    new_stream = torch.empty_like(stream)
    post_identity = torch.ones_like(post_weight)
    post_result = torch.empty_like(branch)
    return branch, stream, post_weight, pre_weight, alpha, beta, output, new_stream, post_identity, post_result


def parse_shape(value: str):
    try:
        rows, hidden = (int(part) for part in value.split(":"))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("shape must be ROWS:HIDDEN") from exc
    if rows <= 0 or hidden <= 0 or hidden % 8:
        raise argparse.ArgumentTypeError("ROWS must be positive and HIDDEN a positive multiple of 8")
    return rows, hidden


def fused(args, tensors):
    branch, stream, post_weight, pre_weight, alpha, beta, output, new_stream = tensors[:8]
    torch.ops.sgl_kernel.fused_post_norm_residual_pre_norm(
        branch,
        stream,
        post_weight if args.post_mode != "none" else post_weight[:0],
        pre_weight,
        alpha,
        beta,
        output,
        new_stream,
        1e-8,
        1e-5,
        args.post_mode == "input_scale",
        args.round_pre_norm_input,
        None,
    )
    return output, new_stream


def main_torch(args, tensors, return_post=False):
    branch, stream, post_weight, pre_weight, alpha, beta, _, _ = tensors[:8]
    post_input = branch.float()
    if args.post_mode == "input_scale":
        post_input = post_input * post_weight.float()
    post_rstd = torch.rsqrt(post_input.square().mean(dim=-1, keepdim=True) + 1e-8)
    normed = post_input * post_rstd
    if args.post_mode == "weight":
        normed = normed * post_weight.float()
    rounded_post = normed.bfloat16()
    new_stream = alpha * stream + beta * rounded_post.float()
    pre_input = (
        new_stream.bfloat16().float()
        if args.round_pre_norm_input
        else new_stream
    )
    pre_rstd = torch.rsqrt(pre_input.square().mean(dim=-1, keepdim=True) + 1e-5)
    output = (pre_input * pre_rstd * pre_weight.float()).bfloat16()
    if return_post:
        return output, new_stream, rounded_post
    return output, new_stream


def main_sycl(args, tensors):
    if args.post_mode == "input_scale" or not args.round_pre_norm_input:
        raise ValueError(
            "main BF16 rmsnorm composition requires post input scaling off "
            "and BF16 rounding before pre norm"
        )
    branch, stream, post_weight, pre_weight, alpha, beta, output, _ = tensors[:8]
    post_identity = tensors[8]
    post_result = tensors[9]
    torch.ops.sgl_kernel.rmsnorm(
        post_result,
        branch,
        post_weight if args.post_mode == "weight" else post_identity,
        1e-8,
    )
    new_stream = alpha * stream + beta * post_result.float()
    pre_input = new_stream.bfloat16()
    torch.ops.sgl_kernel.rmsnorm(output, pre_input, pre_weight, 1e-5)
    return output, new_stream


@lru_cache(maxsize=1)
def framework_triton_components():
    import triton
    from sglang_meta_common.ops.fused_norm_residual import (
        _fused_norm_residual_config,
        _fused_norm_residual_triton_kernel,
    )
    return triton, _fused_norm_residual_config, _fused_norm_residual_triton_kernel


def framework_triton(args, tensors, warp_override=None):
    if args.post_mode == "input_scale":
        raise ValueError("framework Triton input-scale path has an extra BF16 pre-normalization round")
    triton, _fused_norm_residual_config, _fused_norm_residual_triton_kernel = framework_triton_components()

    branch, stream, post_weight, pre_weight, alpha, beta, output, new_stream = tensors[:8]
    rows, hidden = branch.shape
    m_bucket = 0 if rows <= 256 else (1 if rows <= 2048 else 2)
    num_warps, num_stages, extra = _fused_norm_residual_config(
        hidden, m_bucket, rows, False
    )
    if warp_override is not None:
        num_warps = warp_override
    _fused_norm_residual_triton_kernel[(rows,)](
        branch,
        stream,
        output,
        new_stream,
        new_stream,
        post_weight if args.post_mode == "weight" else post_weight[:0],
        pre_weight,
        alpha,
        beta,
        1e-8,
        1e-5,
        hidden,
        triton.next_power_of_2(hidden),
        False,
        True,
        True,
        args.post_mode == "weight",
        True,
        False,
        args.round_pre_norm_input,
        m_bucket,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra,
    )
    return output, new_stream


def framework_triton_w4(args, tensors):
    return framework_triton(args, tensors, warp_override=4)


def framework_triton_w16(args, tensors):
    return framework_triton(args, tensors, warp_override=16)


def framework_triton_w32(args, tensors):
    return framework_triton(args, tensors, warp_override=32)


def measure(providers, args, tensors):
    for name, op in providers.items():
        if name.startswith("wg"):
            os.environ["SGL_KERNEL_REVIEW_WG"] = name[2:]
        else:
            os.environ.pop("SGL_KERNEL_REVIEW_WG", None)
        for _ in range(args.warmup):
            op(args, tensors)
    torch.xpu.synchronize()
    samples = {name: [] for name in providers}
    for repeat in range(args.repeats):
        order = list(providers)
        if repeat % 2:
            order.reverse()
        for name in order:
            if name.startswith("wg"):
                os.environ["SGL_KERNEL_REVIEW_WG"] = name[2:]
            else:
                os.environ.pop("SGL_KERNEL_REVIEW_WG", None)
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                providers[name](args, tensors)
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / args.iterations)
    return samples


def run(args):
    loaded_fused_libraries = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libsgl-ops-sycl-FusedNormResidual.so" in line
        }
    )
    print(
        json.dumps(
            {
                "device": torch.xpu.get_device_name(0),
                "gpu_index": 0,
                "torch": torch.__version__,
                "package": sgl_kernel.__file__,
                "loaded_fused_libraries": loaded_fused_libraries,
                "branch_dtype": "bfloat16",
                "stream_dtype": "float32",
                "weight_dtype": "bfloat16",
                "factor_dtype": "float32",
                "layout": "row-major contiguous",
                "timing": "torch.xpu.Event, microseconds per invocation",
                "order": "alternated per repeat",
                "initial_warmup": args.initial_warmup,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "iterations": args.iterations,
                "serving": "Muse Spark B8/ISL4096/OSL16/TP1/EP1",
            }
        ),
        flush=True,
    )
    available = {
        "fused": fused,
        "main_torch": main_torch,
        "main_sycl": main_sycl,
        "framework_triton": framework_triton,
        "framework_triton_w4": framework_triton_w4,
        "framework_triton_w16": framework_triton_w16,
        "framework_triton_w32": framework_triton_w32,
        "wg256": fused,
        "wg512": fused,
        "wg1024": fused,
    }
    providers = {name: available[name] for name in args.providers}
    initial = inputs(8, 8192)
    for name, op in providers.items():
        if name.startswith("wg"):
            os.environ["SGL_KERNEL_REVIEW_WG"] = name[2:]
        else:
            os.environ.pop("SGL_KERNEL_REVIEW_WG", None)
        for _ in range(args.initial_warmup):
            op(args, initial)
    torch.xpu.synchronize()
    time.sleep(2)
    shapes = args.shape or [(rows, hidden) for rows in args.rows for hidden in args.hidden]
    for rows, hidden in shapes:
        tensors = inputs(rows, hidden)
        measured = measure(providers, args, tensors)
        for provider, samples in measured.items():
            result = {
                "provider": provider,
                "rows": rows,
                "hidden": hidden,
                "post_mode": args.post_mode,
                "round_pre_norm_input": args.round_pre_norm_input,
                "samples_us": samples,
                "p50_us": statistics.median(samples),
                "range_us": [min(samples), max(samples)],
            }
            print(json.dumps(result), flush=True)
        torch.xpu.synchronize()
        time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 32768])
    parser.add_argument("--hidden", type=int, nargs="+", default=[8192])
    parser.add_argument("--shape", type=parse_shape, action="append")
    parser.add_argument(
        "--post-mode", choices=["none", "weight", "input_scale"], default="none"
    )
    parser.add_argument("--round-pre-norm-input", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--initial-warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--providers",
        choices=[
            "fused", "main_torch", "main_sycl", "framework_triton",
            "framework_triton_w4", "framework_triton_w16", "framework_triton_w32",
            "wg256", "wg512", "wg1024",
        ],
        nargs="+",
        default=["fused", "main_torch"],
    )
    run(parser.parse_args())
