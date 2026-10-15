# Test for fused QK normalization and RoPE
# Adapted from the CUDA implementation in sglang

import math
import sys

import pytest
import sgl_kernel
import torch
import utils

precision = {
    torch.bfloat16: 5e-3,
    torch.float16: 1e-3,
    torch.float32: 1e-5,
}
device = utils.get_device()


def llama_rms_norm(x, w, eps=1e-6):
    """PyTorch reference implementation of RMS normalization."""
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x


def apply_rotary_emb_native(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Native PyTorch rotary embedding implementation.
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, rotary_dim // 2]
        sin: [num_tokens, rotary_dim // 2]
        is_neox_style: Whether to use Neox-style or interleaved style
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)

    if is_neox_style:
        # Neox style: split in half along head dimension
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        # Interleaved style: even and odd indices
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def compute_inv_freq_yarn(
    head_dim: int,
    rotary_dim: int,
    base: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
):
    """Evaluate CUDA's negative-exponent frequency formula, rounded once to FP32."""
    inv_freq = (
        base
        ** (
            -torch.arange(0, rotary_dim, 2, dtype=torch.float64, device="cpu")
            / rotary_dim
        )
    ).float()

    if factor != 1.0:
        # YARN scaling
        dim_range = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cpu")

        # Compute linear interpolation factor
        if abs(low_freq_factor - high_freq_factor) <= 1e-6:
            high_freq_factor += 0.001
        linear_func = (dim_range - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        ramp_func = torch.clamp(linear_func, 0.0, 1.0)

        inv_freq_extrapolation = inv_freq
        inv_freq_interpolation = inv_freq / factor

        inv_freq = (
            inv_freq_interpolation * (1.0 - ramp_func)
            + inv_freq_extrapolation * ramp_func
        )

    return inv_freq


def fused_qk_norm_rope_reference(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    base: float,
    is_neox: bool,
    position_ids: torch.Tensor,
    factor: float = 1.0,
    low: float = 1.0,
    high: float = 1.0,
    attention_factor: float = 1.0,
    rotary_dim: int = None,
) -> torch.Tensor:
    """
    Reference implementation in PyTorch for testing.

    Args:
        qkv: [num_tokens, (num_heads_q + num_heads_k + num_heads_v) * head_dim]
        Other args match the kernel interface
    """
    if rotary_dim is None:
        rotary_dim = head_dim

    num_tokens = qkv.shape[0]
    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Reshape QKV to separate Q, K, V
    qkv_reshaped = qkv.view(num_tokens, total_heads, head_dim)

    q = qkv_reshaped[:, :num_heads_q, :]
    k = qkv_reshaped[:, num_heads_q : num_heads_q + num_heads_k, :]
    v = qkv_reshaped[:, num_heads_q + num_heads_k :, :]

    # Apply RMSNorm to Q and K
    q_normalized = llama_rms_norm(q, q_weight, eps)
    k_normalized = llama_rms_norm(k, k_weight, eps)

    # Compute RoPE frequencies
    inv_freq = compute_inv_freq_yarn(head_dim, rotary_dim, base, factor, low, high)

    # Compute cos and sin for each position. Ensure both tensors are on the
    # same device to avoid cross-device ops (tests sometimes pass CPU tensors
    # as reference while inv_freq is constructed on `device`).
    positions = position_ids.to(torch.float32)
    inv_freq = inv_freq.to(positions.device)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()

    # Apply attention factor
    cos = cos * attention_factor
    sin = sin * attention_factor

    # Apply RoPE to Q and K (only to rotary_dim portion)
    q_rot = q_normalized[..., :rotary_dim]
    q_pass = q_normalized[..., rotary_dim:]
    q_rot = apply_rotary_emb_native(q_rot, cos, sin, is_neox)
    q_final = torch.cat([q_rot, q_pass], dim=-1)

    k_rot = k_normalized[..., :rotary_dim]
    k_pass = k_normalized[..., rotary_dim:]
    k_rot = apply_rotary_emb_native(k_rot, cos, sin, is_neox)
    k_final = torch.cat([k_rot, k_pass], dim=-1)

    # Concatenate Q, K, V back together
    result = torch.cat([q_final, k_final, v], dim=1)
    result = result.view(num_tokens, total_heads * head_dim)

    return result


@pytest.mark.parametrize("num_tokens", [1, 7, 32, 128])
@pytest.mark.parametrize("num_heads_q", [8, 32, 64])
@pytest.mark.parametrize("num_heads_k", [4, 8])
@pytest.mark.parametrize("num_heads_v", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_qk_norm_rope_basic(
    num_tokens, num_heads_q, num_heads_k, num_heads_v, head_dim, is_neox, dtype
):
    """Test basic fused QK norm + RoPE without YARN."""
    torch.random.manual_seed(42)
    eps = 1e-6
    base = 10000.0
    factor = 1.0
    low = 1.0
    high = 1.0
    attention_factor = 1.0
    rotary_dim = head_dim

    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Create input tensors
    qkv = torch.randn(num_tokens, total_heads * head_dim, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # Create a copy for reference
    qkv_ref = qkv.clone().float().to("cpu")
    q_weight_ref = q_weight.clone().float().to("cpu")
    k_weight_ref = k_weight.clone().float().to("cpu")
    position_ids_ref = position_ids.clone().to("cpu")

    # Compute reference output
    output_ref = fused_qk_norm_rope_reference(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight_ref,
        k_weight_ref,
        base,
        is_neox,
        position_ids_ref,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    # Run kernel (in-place operation)
    sgl_kernel.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    # Compare results
    torch.testing.assert_close(
        qkv.float().cpu(), output_ref, rtol=precision[dtype], atol=precision[dtype]
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 64, 129, 1024, 2048, 4096])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_qk_norm_rope_production_shapes(num_tokens, is_neox, dtype):
    """Cover production head counts with generic weights, lengths and rotary styles."""
    # Muse-Spark Q128/K16/V16, D64; actual B8 prefill has T32768, not T4096.
    torch.random.manual_seed(42)
    num_heads_q = 128
    num_heads_k = 16
    num_heads_v = 16
    head_dim = 64
    eps = 1e-6
    base = 500000.0
    factor = 1.0
    low = 1.0
    high = 1.0
    attention_factor = 1.0
    rotary_dim = head_dim
    total_heads = num_heads_q + num_heads_k + num_heads_v

    qkv = torch.randn(num_tokens, total_heads * head_dim, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    output_ref = fused_qk_norm_rope_reference(
        qkv.float().cpu(),
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight.float().cpu(),
        k_weight.float().cpu(),
        base,
        is_neox,
        position_ids.cpu(),
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    sgl_kernel.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    torch.testing.assert_close(
        qkv.float().cpu(), output_ref, rtol=precision[dtype], atol=precision[dtype]
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("execution", ["cold_graph", "streams"])
def test_fused_qk_norm_rope_frequency_table_lifetime(dtype, execution):
    torch.manual_seed(127)
    head_dim, heads_q, heads_k, tail, tokens = 64, 8, 4, 4, 7
    base = (130003.0 if execution == "cold_graph" else 140009.0) + (
        16 if dtype == torch.float16 else 0
    )
    original = torch.randn(tokens, 16 * head_dim, device=device, dtype=dtype)
    q_weight = torch.randn(head_dim, device=device, dtype=dtype)
    k_weight = torch.randn(head_dim, device=device, dtype=dtype)
    positions = torch.tensor(
        [0, 1, 255, 1023, 2047, 4094, 4095], device=device, dtype=torch.int32
    )
    expected = fused_qk_norm_rope_reference(
        original.float().cpu(),
        heads_q,
        heads_k,
        tail,
        head_dim,
        1e-6,
        q_weight.float().cpu(),
        k_weight.float().cpu(),
        base,
        False,
        positions.cpu(),
    )

    def invoke(buffer):
        sgl_kernel.fused_qk_norm_rope(
            buffer,
            heads_q,
            heads_k,
            tail,
            head_dim,
            1e-6,
            q_weight,
            k_weight,
            base,
            False,
            positions,
            rotary_dim=head_dim,
        )

    buffers = [original.clone(), original.clone()]
    if execution == "cold_graph":
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            invoke(buffers[0])
        graph.replay()
        invoke(buffers[1])
    else:
        streams = [torch.xpu.Stream(), torch.xpu.Stream()]
        for stream, buffer in zip(streams, buffers):
            stream.wait_stream(torch.xpu.current_stream())
            with torch.xpu.stream(stream):
                invoke(buffer)
    torch.xpu.synchronize()
    for buffer in buffers:
        torch.testing.assert_close(
            buffer.float().cpu(), expected, rtol=precision[dtype], atol=precision[dtype]
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [8, 32768, 32769])
def test_fused_qk_norm_rope_full_prefill(num_tokens, dtype):
    # Muse-Spark B8: decode T8, ISL4096 prefill T32768; Q128/K16/D64 plus gated tail144.
    # T32769 is a generic launch tail, not another serving batch configuration.
    torch.manual_seed(42)
    num_heads_q, num_heads_k, tail_heads, head_dim = 128, 16, 144, 64
    qkv = torch.randn(num_tokens, 288 * head_dim, dtype=dtype, device=device)
    initial = qkv.cpu()
    q_weight = torch.full(
        (head_dim,), 43.7840518911 / math.sqrt(head_dim), dtype=dtype, device=device
    )
    k_weight = torch.ones(head_dim, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int32, device=device) % 4096
    sgl_kernel.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        tail_heads,
        head_dim,
        1e-5,
        q_weight,
        k_weight,
        500000.0,
        False,
        positions,
        rotary_dim=head_dim,
    )
    actual = qkv.cpu()
    for start in range(0, num_tokens, 512):
        expected = fused_qk_norm_rope_reference(
            initial[start : start + 512].float(),
            num_heads_q,
            num_heads_k,
            tail_heads,
            head_dim,
            1e-5,
            q_weight.float().cpu(),
            k_weight.float().cpu(),
            500000.0,
            False,
            positions[start : start + 512].cpu(),
            rotary_dim=head_dim,
        )
        torch.testing.assert_close(
            actual[start : start + 512].float(),
            expected,
            rtol=precision[dtype],
            atol=precision[dtype],
        )
    assert torch.equal(actual[:, 144 * head_dim :], initial[:, 144 * head_dim :])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_qk_norm_rope_gated_stride(dtype):
    """Cover the DRAM-sized, two-chunk interleaved launch with a large untouched tail."""
    torch.random.manual_seed(42)
    num_tokens = 4096
    num_heads_q = 128
    num_heads_k = 16
    num_heads_v = 144
    head_dim = 64
    eps = 1e-6
    base = 500000.0
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    qkv = torch.randn(
        num_tokens,
        (num_heads_q + num_heads_k + num_heads_v) * head_dim,
        dtype=dtype,
        device=device,
    )

    output_ref = fused_qk_norm_rope_reference(
        qkv.float().cpu(),
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight.float().cpu(),
        k_weight.float().cpu(),
        base,
        False,
        position_ids.cpu(),
        rotary_dim=head_dim,
    )

    sgl_kernel.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        False,
        position_ids,
        rotary_dim=head_dim,
    )

    torch.testing.assert_close(
        qkv.float().cpu(), output_ref, rtol=precision[dtype], atol=precision[dtype]
    )


@pytest.mark.parametrize("num_tokens", [32, 128])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_qk_norm_rope_yarn(num_tokens, head_dim, is_neox, dtype):
    """Test fused QK norm + RoPE with YARN scaling."""
    torch.random.manual_seed(42)
    num_heads_q = 32
    num_heads_k = 8
    num_heads_v = 8
    eps = 1e-6
    base = 10000.0
    factor = 2.0  # YARN factor
    low = 8.0
    high = 1024.0
    attention_factor = 0.707  # sqrt(0.5)
    rotary_dim = head_dim

    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Create input tensors
    qkv = torch.randn(num_tokens, total_heads * head_dim, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # Create a copy for reference
    qkv_ref = qkv.clone().float().to("cpu")
    q_weight_ref = q_weight.clone().float().to("cpu")
    k_weight_ref = k_weight.clone().float().to("cpu")
    position_ids_ref = position_ids.clone().to("cpu")

    # Compute reference output
    output_ref = fused_qk_norm_rope_reference(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight_ref,
        k_weight_ref,
        base,
        is_neox,
        position_ids_ref,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    # Run kernel (in-place operation)
    sgl_kernel.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    # Compare results - use slightly relaxed tolerance for YARN
    torch.testing.assert_close(
        qkv.float().cpu(), output_ref, rtol=precision[dtype], atol=precision[dtype]
    )


@pytest.mark.parametrize("num_tokens", [64])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("rotary_dim", [32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_qk_norm_rope_partial_rotary(num_tokens, head_dim, rotary_dim, dtype):
    """Test with partial rotary dimensions (rotary_dim < head_dim)."""
    torch.random.manual_seed(42)
    num_heads_q = 16
    num_heads_k = 4
    num_heads_v = 4
    eps = 1e-6
    base = 10000.0
    is_neox = True
    factor = 1.0
    low = 1.0
    high = 1.0
    attention_factor = 1.0

    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Create input tensors
    qkv = torch.randn(num_tokens, total_heads * head_dim, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # Create a copy for reference
    qkv_ref = qkv.clone().float().to("cpu")
    q_weight_ref = q_weight.clone().float().to("cpu")
    k_weight_ref = k_weight.clone().float().to("cpu")
    position_ids_ref = position_ids.clone().to("cpu")

    # Compute reference output
    output_ref = fused_qk_norm_rope_reference(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight_ref,
        k_weight_ref,
        base,
        is_neox,
        position_ids_ref,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    # Run kernel (in-place operation)
    sgl_kernel.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    # Compare results
    torch.testing.assert_close(
        qkv.float().cpu(), output_ref, rtol=precision[dtype], atol=precision[dtype]
    )


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "num_heads_q,num_heads_k,num_heads_v",
    [
        (2, 2, 2),
        (1, 1, 1),
        (4, 2, 2),
        (8, 4, 2),
    ],
)
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("is_neox", [True, False])
def test_fused_qk_norm_rope_fp8_e4m3(
    num_tokens, num_heads_q, num_heads_k, num_heads_v, head_dim, is_neox
):
    """Test fused QK norm + RoPE with FP8_E4M3 dtype."""
    torch.random.manual_seed(42)
    dtype = torch.float8_e4m3fn
    eps = 1e-6
    base = 10000.0
    factor = 1.0
    low = 1.0
    high = 1.0
    attention_factor = 1.0
    rotary_dim = head_dim

    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Create input tensors in float32 first, then convert to FP8
    qkv_f32 = torch.randn(
        num_tokens, total_heads * head_dim, dtype=torch.float32, device=device
    )
    # Clamp to FP8 representable range to avoid infinities/NaNs on conversion
    qkv_f32 = qkv_f32.clamp(-448.0, 448.0)
    qkv = qkv_f32.to(dtype)

    q_weight_f32 = torch.randn(head_dim, dtype=torch.float32, device=device)
    q_weight_f32 = q_weight_f32.clamp(-448.0, 448.0)
    q_weight = q_weight_f32.to(dtype)

    k_weight_f32 = torch.randn(head_dim, dtype=torch.float32, device=device)
    k_weight_f32 = k_weight_f32.clamp(-448.0, 448.0)
    k_weight = k_weight_f32.to(dtype)

    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # Create a copy for reference from FP8-dequantized values
    qkv_ref = qkv.to(torch.float32).clone().cpu()
    q_weight_ref = q_weight.to(torch.float32).clone().cpu()
    k_weight_ref = k_weight.to(torch.float32).clone().cpu()
    position_ids_ref = position_ids.clone().cpu()

    # Compute reference output on CPU
    output_ref = fused_qk_norm_rope_reference(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight_ref,
        k_weight_ref,
        base,
        is_neox,
        position_ids_ref,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    # Run kernel (in-place operation)
    sgl_kernel.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    # Compare results - use relaxed tolerance for FP8
    # FP8 has limited precision, so we need higher tolerance
    torch.testing.assert_close(qkv.float().cpu(), output_ref, rtol=5e-2, atol=5e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
