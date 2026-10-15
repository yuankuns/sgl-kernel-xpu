"""Validate the full Muse B8 prefill launch at batch and tail boundaries."""

import argparse

import torch
from sgl_kernel import fused_qk_norm_rope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--q-heads", type=int, default=128)
    parser.add_argument("--k-heads", type=int, default=16)
    parser.add_argument("--tail-heads", type=int, default=144)
    parser.add_argument("--head-dim", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()
    if min(args.tokens, args.q_heads, args.k_heads) <= 0 or args.tail_heads < 0:
        parser.error(
            "tokens and Q/K heads must be positive; tail-heads must be nonnegative"
        )
    torch.manual_seed(42)
    tokens = args.tokens
    heads_q, heads_k, heads_v, dim = (
        args.q_heads, args.k_heads, args.tail_heads, args.head_dim
    )
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    eps, base = 1e-5, 500000.0
    scale = 43.7840518911 / (dim**0.5)
    qkv = torch.randn(
        tokens,
        (heads_q + heads_k + heads_v) * dim,
        device="xpu:0",
        dtype=dtype,
    )
    original = qkv.clone()
    q_weight = torch.full((dim,), scale, device="xpu:0", dtype=dtype)
    k_weight = torch.ones(dim, device="xpu:0", dtype=dtype)
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

    expected = torch.cat((rotate(q), rotate(k)), dim=1).to(dtype)
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
        f"PASS tokens={tokens} {args.dtype} heads={heads_q},{heads_k},{heads_v} "
        f"dim={dim} rows={selected.tolist()} max_abs={difference.max().item():.6f}"
    )


if __name__ == "__main__":
    main()
