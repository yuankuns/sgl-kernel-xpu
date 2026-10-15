#!/usr/bin/env python3
"""Validate the production Muse Spark decode FMHA shapes against FP32."""

import sgl_kernel  # noqa: F401
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

# Muse-Spark: B8, ISL4096, OSL16, Q/KV heads128/16, D64, BF16, TP1/EP1.
BATCH = 8
SEQLEN_K = 4097
PAGE_SIZE = 64
HEADS_Q = 128
HEADS_KV = 16
HEAD_DIM = 64
GQA_GROUP = HEADS_Q // HEADS_KV


def reference(q, k_cache, v_cache, page_table, local):
    pages_per_seq = (SEQLEN_K + PAGE_SIZE - 1) // PAGE_SIZE
    k = (
        k_cache.cpu()[page_table.cpu()]
        .view(BATCH, pages_per_seq * PAGE_SIZE, HEADS_KV, HEAD_DIM)[:, :SEQLEN_K]
        .float()
    )
    v = (
        v_cache.cpu()[page_table.cpu()]
        .view(BATCH, pages_per_seq * PAGE_SIZE, HEADS_KV, HEAD_DIM)[:, :SEQLEN_K]
        .float()
    )
    q_grouped = q.cpu().float().view(BATCH, HEADS_KV, GQA_GROUP, HEAD_DIM)
    scores = torch.einsum("bkgd,bskd->bkgs", q_grouped, k) * HEAD_DIM**-0.5
    if local:
        first_key = SEQLEN_K - 1 - 2047
        scores[..., :first_key] = float("-inf")
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bkgs,bskd->bkgd", probs, v).reshape(BATCH, HEADS_Q, HEAD_DIM)


def main():
    pages_per_seq = (SEQLEN_K + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = BATCH * pages_per_seq
    torch.manual_seed(0)
    q = torch.randn(BATCH, HEADS_Q, HEAD_DIM, device="xpu", dtype=torch.bfloat16)
    k_cache = torch.randn(
        num_pages,
        PAGE_SIZE,
        HEADS_KV,
        HEAD_DIM,
        device="xpu",
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.randperm(num_pages, device="xpu", dtype=torch.int32).view(
        BATCH, pages_per_seq
    )
    cache_seqlens = torch.full((BATCH,), SEQLEN_K, device="xpu", dtype=torch.int32)
    cu_seqlens_q = torch.arange(BATCH + 1, device="xpu", dtype=torch.int32)

    for local in (False, True):
        out = flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            max_seqlen_k=SEQLEN_K,
            softmax_scale=HEAD_DIM**-0.5,
            causal=True,
            window_size=(2047, 0) if local else (-1, -1),
            rotary_interleaved=False,
        )
        torch.xpu.synchronize()
        ref = reference(q, k_cache, v_cache, page_table, local)
        error = out.float().cpu() - ref
        max_abs = error.abs().max().item()
        mean_abs = error.abs().mean().item()
        torch.testing.assert_close(out.float().cpu(), ref, rtol=5e-3, atol=5e-3)
        print(
            f"{'swa' if local else 'global'}: max_abs={max_abs:.6f} "
            f"mean_abs={mean_abs:.6f}"
        )
        torch.xpu.synchronize()


if __name__ == "__main__":
    main()
