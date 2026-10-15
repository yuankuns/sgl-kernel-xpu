import argparse
import json
import statistics
import tempfile
from pathlib import Path

import pytest
import sgl_kernel  # noqa: F401
import torch

POST_EPS = 1e-8
PRE_EPS = 1e-5


def round_bfloat16(values: torch.Tensor) -> torch.Tensor:
    """Round an independent FP64 oracle directly to BF16, without an FP32 double-round."""
    high = values.float()
    midpoint = ((high.view(torch.int32) & 0xFFFF) == 0x8000) & (values != high.double())
    if midpoint.any():
        direction = torch.where(
            values[midpoint] > high[midpoint].double(), float("inf"), float("-inf")
        ).float()
        high[midpoint] = torch.nextafter(high[midpoint], direction)
    return high.bfloat16()


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
    """Independent FP64 mathematics with the model's BF16 boundaries and FP32 stream store."""
    post_input = branch.double()
    if post_weight.numel() and post_norm_input_scale:
        post_input = round_bfloat16(post_input * post_weight.double()).double()
    post_eps = torch.tensor(POST_EPS, dtype=torch.float32, device="cpu").double()
    pre_eps = torch.tensor(PRE_EPS, dtype=torch.float32, device="cpu").double()
    post_rstd = torch.rsqrt(post_input.square().mean(dim=-1, keepdim=True) + post_eps)
    normed_branch = post_input * post_rstd
    if post_weight.numel() and not post_norm_input_scale:
        normed_branch = normed_branch * post_weight.double()
    normed_branch = round_bfloat16(normed_branch)
    rounded_post_norm = normed_branch.float()
    new_stream = (
        alpha.double() * stream.double() + beta.double() * rounded_post_norm.double()
    ).float()
    pre_norm_input = (
        new_stream.bfloat16().double() if round_pre_norm_input else new_stream.double()
    )
    pre_rstd = torch.rsqrt(pre_norm_input.square().mean(dim=-1, keepdim=True) + pre_eps)
    output = (pre_norm_input * pre_rstd * pre_weight.double()).float()
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


def test_round_bfloat16_reference_midpoints():
    midpoint = 1.30859375
    values = torch.tensor(
        [midpoint - 2**-40, midpoint, midpoint + 2**-40], dtype=torch.float64
    )
    expected = torch.tensor([1.3046875, 1.3125, 1.3125], dtype=torch.bfloat16)
    torch.testing.assert_close(round_bfloat16(values), expected, rtol=0, atol=0)
    torch.testing.assert_close(round_bfloat16(-values), -expected, rtol=0, atol=0)


def test_fused_norm_residual_fp32_fma_boundary():
    device = torch.device("xpu")
    branch = -torch.ones(1, 8, dtype=torch.bfloat16, device=device)
    stream = torch.full((1, 8), 1 - 2**-23, dtype=torch.float32, device=device)
    alpha = torch.full((8,), 1 + 2**-23, dtype=torch.float32, device=device)
    beta = torch.ones_like(alpha)
    post = torch.empty(0, dtype=torch.bfloat16, device=device)
    pre = torch.ones(8, dtype=torch.bfloat16, device=device)
    _, updated = run_op(branch, stream, post, pre, alpha, beta)
    torch.testing.assert_close(
        updated.cpu(), torch.full((1, 8), -(2**-46)), rtol=0, atol=0
    )


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
    branch_cpu, stream_cpu = branch.cpu(), stream.cpu()
    post_weight_cpu, pre_weight_cpu = post_weight.cpu(), pre_weight.cpu()
    alpha_cpu, beta_cpu = alpha.cpu(), beta.cpu()
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

    actual_output_cpu = actual_output.float().cpu()
    actual_stream_cpu = actual_stream.cpu()
    for start in range(0, rows, 512):
        expected_output, expected_stream, _ = reference(
            branch_cpu[start : start + 512],
            stream_cpu[start : start + 512],
            post_weight_cpu,
            pre_weight_cpu,
            alpha_cpu,
            beta_cpu,
            post_mode == "input_scale",
            round_pre_norm_input,
        )
        torch.testing.assert_close(
            actual_output_cpu[start : start + 512],
            expected_output,
            rtol=5e-3,
            atol=5e-3,
        )
        torch.testing.assert_close(
            actual_stream_cpu[start : start + 512],
            expected_stream,
            rtol=1e-5,
            atol=1e-5,
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


def test_fused_norm_residual_hidden_tail_rows_boundary() -> None:
    # H=8200 uses a partial final vector; rows=385 borders a workgroup change.
    test_fused_norm_residual(385, 8200, "none", True)


@pytest.mark.parametrize("rows", [8, 32768])
@pytest.mark.parametrize("post_mode", ["none", "weight", "input_scale"])
@pytest.mark.parametrize("round_pre_norm_input", [False, True])
def test_fused_norm_residual_production_shapes(rows, post_mode, round_pre_norm_input):
    # Muse-Spark B8: decode T8, ISL4096 prefill T32768, hidden8192, BF16/FP32 stream.
    test_fused_norm_residual(rows, 8192, post_mode, round_pre_norm_input)


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
        durations_us = [event["dur"] for event in kernels]
        latency_us = statistics.median(durations_us)
        print(
            f"rows={rows} hidden={hidden_size} "
            f"post_norm_input_scale={post_norm_input_scale} "
            f"round_pre_norm_input={round_pre_norm_input} "
            f"kernel_event_p50={latency_us:.3f} us "
            f"kernel_event_range_us=[{min(durations_us):.3f},{max(durations_us):.3f}] "
            f"kernel_event_samples_us={json.dumps(durations_us)}"
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
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--post-norm-input-scale", action="store_true")
    parser.add_argument("--round-pre-norm-input", action="store_true")
    args = parser.parse_args()
    if args.rows <= 0 or args.hidden_size <= 0 or args.hidden_size % 8:
        parser.error(
            "rows must be positive and hidden-size must be a positive multiple of 8"
        )
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be nonnegative and iterations must be positive")
    benchmark(
        args.rows,
        args.hidden_size,
        args.warmup,
        args.iterations,
        args.post_norm_input_scale,
        args.round_pre_norm_input,
        args.profiler,
    )
