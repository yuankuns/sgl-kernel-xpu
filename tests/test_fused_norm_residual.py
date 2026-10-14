import argparse
import json
import os
import statistics
import tempfile
from pathlib import Path

import pytest
import sgl_kernel  # noqa: F401
import torch

POST_EPS = 1e-8
PRE_EPS = 1e-5


def reference(
    branch: torch.Tensor,
    stream: torch.Tensor,
    post_weight: torch.Tensor,
    pre_weight: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    post_norm_input_scale: bool,
    round_pre_norm_input: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    post_input = branch.float()
    if post_weight.numel() and post_norm_input_scale:
        post_input = post_input * post_weight.float()
    post_rstd = torch.rsqrt(post_input.square().mean(dim=-1, keepdim=True) + POST_EPS)
    normed_branch = post_input * post_rstd
    if post_weight.numel() and not post_norm_input_scale:
        normed_branch = normed_branch * post_weight.float()
    normed_branch = normed_branch.bfloat16()
    rounded_post_norm = normed_branch.float()
    new_stream = alpha * stream + beta * rounded_post_norm
    pre_norm_input = (
        new_stream.bfloat16().float() if round_pre_norm_input else new_stream
    )
    pre_rstd = torch.rsqrt(pre_norm_input.square().mean(dim=-1, keepdim=True) + PRE_EPS)
    output = (pre_norm_input * pre_rstd * pre_weight.float()).bfloat16()
    return output, new_stream, normed_branch


def run_op(
    branch: torch.Tensor,
    stream: torch.Tensor,
    post_weight: torch.Tensor,
    pre_weight: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    post_norm_input_scale: bool = False,
    round_pre_norm_input: bool = False,
    output_fp32: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(branch)
    new_stream = torch.empty_like(stream)
    torch.ops.sgl_kernel.fused_post_norm_residual_pre_norm(
        branch,
        stream,
        post_weight,
        pre_weight,
        alpha,
        beta,
        output,
        new_stream,
        POST_EPS,
        PRE_EPS,
        post_norm_input_scale,
        round_pre_norm_input,
        output_fp32,
    )
    return output, new_stream


@pytest.mark.parametrize("rows", [1, 37, 257])
@pytest.mark.parametrize("hidden_size", [4096, 8192, 16384])
@pytest.mark.parametrize("post_mode", ["none", "weight", "input_scale"])
@pytest.mark.parametrize("round_pre_norm_input", [False, True])
def test_fused_norm_residual(
    rows: int, hidden_size: int, post_mode: str, round_pre_norm_input: bool
) -> None:
    torch.manual_seed(1234 + rows)
    device = torch.device("xpu")
    branch = torch.randn(rows, hidden_size, device=device, dtype=torch.bfloat16)
    stream = torch.randn(rows, hidden_size, device=device, dtype=torch.float32)
    post_weight = (
        torch.empty(0, device=device, dtype=torch.bfloat16)
        if post_mode == "none"
        else torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
    )
    pre_weight = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
    alpha = torch.rand(hidden_size, device=device, dtype=torch.float32)
    beta = torch.rand(hidden_size, device=device, dtype=torch.float32)

    # The XPU elementwise reference can produce incorrect values on CRI for
    # these shapes. Compute the independent reference from the same inputs on
    # CPU and compare there so the test checks the fused kernel itself.
    expected_output, expected_stream, rounded_post_norm = reference(
        branch.cpu(),
        stream.cpu(),
        post_weight.cpu(),
        pre_weight.cpu(),
        alpha.cpu(),
        beta.cpu(),
        post_mode == "input_scale",
        round_pre_norm_input,
    )
    actual_output, actual_stream = run_op(
        branch,
        stream,
        post_weight,
        pre_weight,
        alpha,
        beta,
        post_mode == "input_scale",
        round_pre_norm_input,
    )

    torch.testing.assert_close(
        actual_output.cpu(), expected_output, rtol=2e-2, atol=2e-2
    )
    actual_stream_cpu = actual_stream.cpu()
    upper = torch.nextafter(
        rounded_post_norm, torch.full_like(rounded_post_norm, float("inf"))
    ).float()
    lower = torch.nextafter(
        rounded_post_norm, torch.full_like(rounded_post_norm, float("-inf"))
    ).float()
    post_norm_ulp = torch.maximum(
        upper - rounded_post_norm.float(), rounded_post_norm.float() - lower
    )
    # A tiny FP32 reduction change can flip a BF16 value at a rounding
    # midpoint. Its effect on the residual add is at most |beta| * one ULP.
    stream_error = (actual_stream_cpu - expected_stream).abs()
    stream_bound = 1e-5 + beta.cpu().abs() * post_norm_ulp
    violations = stream_error > stream_bound
    assert not violations.any(), (
        f"{violations.sum().item()} residual elements exceed the BF16 ULP bound; "
        f"max excess {(stream_error - stream_bound).max().item()}"
    )

    output_fp32 = torch.empty_like(stream)
    widened_output, widened_stream = run_op(
        branch,
        stream,
        post_weight,
        pre_weight,
        alpha,
        beta,
        post_mode == "input_scale",
        round_pre_norm_input,
        output_fp32,
    )
    torch.testing.assert_close(
        widened_output.cpu(), actual_output.cpu(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        widened_stream.cpu(), actual_stream.cpu(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        output_fp32.cpu(), actual_output.cpu().float(), rtol=0, atol=0
    )


def benchmark(
    rows: int,
    hidden_size: int,
    warmup: int,
    iterations: int,
    post_norm_input_scale: bool,
    round_pre_norm_input: bool,
    profiler_events: bool = False,
) -> None:
    device = torch.device("xpu")
    branch = torch.randn(rows, hidden_size, device=device, dtype=torch.bfloat16)
    stream = torch.randn(rows, hidden_size, device=device, dtype=torch.float32)
    post_weight = (
        torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
        if post_norm_input_scale
        else torch.empty(0, device=device, dtype=torch.bfloat16)
    )
    pre_weight = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
    alpha = torch.rand(hidden_size, device=device, dtype=torch.float32)
    beta = torch.rand(hidden_size, device=device, dtype=torch.float32)

    for _ in range(warmup):
        run_op(
            branch,
            stream,
            post_weight,
            pre_weight,
            alpha,
            beta,
            post_norm_input_scale,
            round_pre_norm_input,
        )
    torch.xpu.synchronize()

    if profiler_events:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        ) as profiler:
            for _ in range(iterations):
                run_op(
                    branch,
                    stream,
                    post_weight,
                    pre_weight,
                    alpha,
                    beta,
                    post_norm_input_scale,
                    round_pre_norm_input,
                )
            torch.xpu.synchronize()
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.json"
            profiler.export_chrome_trace(str(trace_path))
            events = json.loads(trace_path.read_text())["traceEvents"]
        kernels = [event for event in events if event.get("cat") == "kernel"]
        if len(kernels) != iterations:
            names = sorted({event.get("name", "") for event in kernels})
            raise RuntimeError(
                f"expected {iterations} XPU kernel events, got {len(kernels)}: {names}"
            )
        latency_us = statistics.median(event["dur"] for event in kernels)
        print(
            f"rows={rows} hidden={hidden_size} "
            f"post_norm_input_scale={post_norm_input_scale} "
            f"round_pre_norm_input={round_pre_norm_input} "
            f"kernel_event_p50={latency_us:.3f} us"
        )
        return

    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        run_op(
            branch,
            stream,
            post_weight,
            pre_weight,
            alpha,
            beta,
            post_norm_input_scale,
            round_pre_norm_input,
        )
    end.record()
    end.synchronize()
    elapsed_ms = start.elapsed_time(end) / iterations

    # Compulsory tensor traffic: branch + stream + output + new_stream.
    bytes_per_row = hidden_size * (2 + 4 + 2 + 4)
    bandwidth_gbs = bytes_per_row * rows / (elapsed_ms * 1e6)
    print(
        f"rows={rows} hidden={hidden_size} "
        f"post_norm_input_scale={post_norm_input_scale} "
        f"round_pre_norm_input={round_pre_norm_input} time_ms={elapsed_ms:.4f} "
        f"three_layers_ms={elapsed_ms * 3:.4f} bandwidth_GBps={bandwidth_gbs:.1f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serving-batch", type=int, default=8)
    parser.add_argument("--isl", type=int, default=4096)
    parser.add_argument("--osl", type=int, default=16)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--rows", type=int, choices=[8, 32768], default=32768)
    parser.add_argument(
        "--hidden-size", type=int, choices=[4096, 8192, 16384], default=8192
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--post-norm-input-scale", action="store_true")
    parser.add_argument("--round-pre-norm-input", action="store_true")
    args = parser.parse_args()
    if (args.serving_batch, args.isl, args.osl, args.tp, args.ep) != (
        8,
        4096,
        16,
        1,
        1,
    ) or os.environ.get("BATCHES", "8") != "8":
        parser.error("Muse-Spark serving scope requires B=8, ISL=4096, OSL=16, TP1/EP1")
    benchmark(
        args.rows,
        args.hidden_size,
        args.warmup,
        args.iterations,
        args.post_norm_input_scale,
        args.round_pre_norm_input,
        args.profiler,
    )
