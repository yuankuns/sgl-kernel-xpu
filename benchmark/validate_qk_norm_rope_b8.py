"""Validate the full Muse B8 prefill launch at batch and tail boundaries."""

import os

import torch
from sgl_kernel import fused_qk_norm_rope


def main():
    torch.manual_seed(42)
    tokens = int(os.environ.get("QK_VALIDATE_TOKENS", "32768"))
    heads_q = int(os.environ.get("QK_Q_HEADS", "128"))
    heads_k = int(os.environ.get("QK_K_HEADS", "16"))
    heads_v = int(os.environ.get("QK_TAIL_HEADS", "144"))
    dim = int(os.environ.get("QK_HEAD_DIM", "64"))
    eps, base = 1e-5, 500000.0
    scale = 43.7840518911 / (dim**0.5)
    qkv = torch.randn(
        tokens,
        (heads_q + heads_k + heads_v) * dim,
        device="xpu:0",
        dtype=torch.bfloat16,
    )
    original = qkv.clone()
    q_weight = torch.full((dim,), scale, device="xpu:0", dtype=torch.bfloat16)
    k_weight = torch.ones(dim, device="xpu:0", dtype=torch.bfloat16)
    positions = torch.arange(tokens, device="xpu:0", dtype=torch.int32) % 4096
    boundary_rows = [
        0,
        1,
        4095,
        4096,
        8191,
        8192,
        16383,
        16384,
        tokens // 2 - 1,
        tokens // 2,
        tokens - 2,
        tokens - 1,
    ]
    selected = torch.tensor(
        sorted({row for row in boundary_rows if 0 <= row < tokens}), device="xpu:0"
    )

    rows = original[selected].reshape(-1, heads_q + heads_k + heads_v, dim).float()
    q = rows[:, :heads_q]
    k = rows[:, heads_q : heads_q + heads_k]
    q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + eps) * q_weight.float()
    k = k * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + eps) * k_weight.float()
    inv_freq = base ** (
        -torch.arange(0, dim, 2, device="xpu:0", dtype=torch.float32) / dim
    )
    theta = positions[selected].float()[:, None] * inv_freq[None, :]
    cos, sin = theta.cos()[:, None, :], theta.sin()[:, None, :]

    def rotate(x):
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, odd * cos + even * sin), dim=-1
        ).flatten(-2)

    expected = torch.cat((rotate(q), rotate(k)), dim=1).to(torch.bfloat16)
    fused_qk_norm_rope(
        qkv,
        heads_q,
        heads_k,
        heads_v,
        dim,
        eps,
        q_weight,
        k_weight,
        base,
        False,
        positions,
        rotary_dim=dim,
    )
    torch.xpu.synchronize()
    actual = qkv[selected, : (heads_q + heads_k) * dim].reshape_as(expected)
    difference = (actual.float() - expected.float()).abs()
    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=4e-3)
    assert torch.equal(
        qkv[:, (heads_q + heads_k) * dim :], original[:, (heads_q + heads_k) * dim :]
    )
    print(
        f"PASS tokens={tokens} bf16 heads={heads_q},{heads_k},{heads_v} "
        f"dim={dim} rows={selected.tolist()} max_abs={difference.max().item():.6f}"
    )


if __name__ == "__main__":
    main()
