# SPDX-License-Identifier: Apache-2.0
"""Decode split-KV GQA FMHA coverage for the Xe3 (CRI) paged path.

Every case below takes the `dispatch_xe3_decode_splitkv` route in
`flash_attentionXe35_common.hpp`: one query token per request, GQA ratio at
most 8, paged BF16 cache.  Cases cover ragged cache lengths, page-tail
handling, sliding windows that start inside a tile, GQA ratios 1/4/8, page
sizes 64 and 128, and split counts that leave some splits empty.  The default
heuristic targets about 64 workgroups, so the production B=8 shape runs with a
single split and the small-slot cases exercise the multi-split merge.

"""

import time

import pytest
import sgl_kernel  # noqa: F401 - registers torch.ops.sgl_kernel.* custom ops
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache


def is_xpu_available():
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_cri_device():
    if not is_xpu_available():
        return False
    return torch.xpu.get_device_properties(0).device_id == 0x674F


def reference(q, k_cache, v_cache, page_table, cache_seqlens, window_left):
    """FP32 CPU reference for one-token paged decode with GQA."""
    batch, heads_q, head_dim = q.shape
    _, page_size, heads_kv, _ = k_cache.shape
    group = heads_q // heads_kv
    k_cache = k_cache.cpu().float()
    v_cache = v_cache.cpu().float()
    page_table = page_table.cpu()
    out = torch.zeros(batch, heads_q, head_dim, dtype=torch.float32)
    for b in range(batch):
        length = int(cache_seqlens[b])
        if length == 0:
            continue
        pages = page_table[b, : (length + page_size - 1) // page_size]
        k = k_cache[pages].reshape(-1, heads_kv, head_dim)[:length]
        v = v_cache[pages].reshape(-1, heads_kv, head_dim)[:length]
        qb = q[b].cpu().float().view(heads_kv, group, head_dim)
        scores = torch.einsum("kgd,skd->kgs", qb, k) * head_dim**-0.5
        if window_left >= 0:
            first = length - 1 - window_left
            if first > 0:
                scores[..., :first] = float("-inf")
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("kgs,skd->kgd", probs, v).reshape(heads_q, head_dim)
    return out


def run_case(
    lengths, heads_q, heads_kv, head_dim, page_size, window, seed=0, num_splits=0
):
    device = torch.device("xpu")
    dtype = torch.bfloat16
    torch.manual_seed(seed)
    batch = len(lengths)
    max_len = max(lengths)
    pages_per_seq = max(1, (max_len + page_size - 1) // page_size)
    num_pages = batch * pages_per_seq + 3
    q = torch.randn(batch, heads_q, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(
        num_pages, page_size, heads_kv, head_dim, device=device, dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.randperm(num_pages, dtype=torch.int32, device=device)[
        : batch * pages_per_seq
    ].view(batch, pages_per_seq)
    cache_seqlens = torch.tensor(lengths, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=device)

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        max_seqlen_k=max_len,
        softmax_scale=head_dim**-0.5,
        causal=True,
        window_size=window,
        rotary_interleaved=False,
        num_splits=num_splits,
    )
    torch.xpu.synchronize()
    ref = reference(q, k_cache, v_cache, page_table, lengths, window[0])
    return out.float().cpu(), ref


@pytest.fixture(autouse=True)
def cool_between_cases():
    yield
    if is_xpu_available():
        torch.xpu.synchronize()
        time.sleep(2)


CASES = [
    # Muse Spark production decode shape: B=8, 4096 prefill + 1 token.
    pytest.param([4097] * 8, 128, 16, 64, 64, (-1, -1), id="muse-b8-global"),
    pytest.param([4097] * 8, 128, 16, 64, 64, (2047, 0), id="muse-b8-swa2047"),
    # Ragged lengths, tile and page tails, windows that start mid tile.
    pytest.param([1, 63, 64, 65, 129], 128, 16, 64, 64, (-1, -1), id="ragged-global"),
    pytest.param([1, 63, 64, 65, 129], 128, 16, 64, 64, (63, 0), id="ragged-swa63"),
    pytest.param(
        [200, 4097, 1, 2048, 2049, 2112],
        128,
        16,
        64,
        64,
        (2047, 0),
        id="ragged-swa2047",
    ),
    pytest.param(
        [200, 4097, 1, 2048, 2049, 2112],
        128,
        16,
        64,
        64,
        (-1, -1),
        id="ragged-long-global",
    ),
    # Other GQA ratios and tiny batch (many splits per slot).
    pytest.param([777, 4097], 32, 8, 64, 64, (-1, -1), id="gqa4-global"),
    pytest.param([777, 4097], 16, 16, 64, 64, (300, 0), id="gqa1-swa300"),
    pytest.param([5000], 8, 1, 64, 64, (-1, -1), id="gqa8-single-slot"),
    # The small KV-head grid uses multiple splits; short requests leave some
    # splits empty, and the 2048-token request ends on a tile boundary.
    pytest.param(
        [1, 63, 4097, 2048], 8, 1, 64, 64, (-1, -1), id="gqa8-ragged-many-splits"
    ),
    pytest.param(
        [1, 63, 4097, 2048], 8, 1, 64, 64, (2047, 0), id="gqa8-ragged-many-splits-swa"
    ),
    # head_dim 128 goes through the same dispatch (decode_128 target).
    pytest.param([4097, 130, 1], 128, 16, 128, 64, (-1, -1), id="hd128-global"),
    pytest.param([4097, 130, 1], 128, 16, 128, 64, (2047, 0), id="hd128-swa2047"),
    # Page size 128 uses the generic tile-in-page addressing.
    pytest.param([4097, 130, 1], 128, 16, 64, 128, (-1, -1), id="page128-global"),
    pytest.param([4097, 130, 1], 128, 16, 64, 128, (2047, 0), id="page128-swa2047"),
]


@pytest.mark.skipif(not is_xpu_available(), reason="Intel XPU not available")
@pytest.mark.skipif(not is_cri_device(), reason="Requires a CRI (Xe3P) device")
@pytest.mark.parametrize("lengths,heads_q,heads_kv,head_dim,page_size,window", CASES)
def test_decode_splitkv(lengths, heads_q, heads_kv, head_dim, page_size, window):
    out, ref = run_case(lengths, heads_q, heads_kv, head_dim, page_size, window)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not is_cri_device(), reason="Requires a CRI (Xe3P) device")
@pytest.mark.parametrize("num_splits", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("window", [(-1, -1), (2047, 0)])
@pytest.mark.parametrize("heads_q", [128, 144])
def test_decode_explicit_split_count(num_splits, window, heads_q):
    out, ref = run_case(
        [4097, 2049, 2112],
        heads_q,
        16,
        64,
        64,
        window,
        num_splits=num_splits,
    )
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not is_xpu_available(), reason="Intel XPU not available")
@pytest.mark.skipif(not is_cri_device(), reason="Requires a CRI (Xe3P) device")
def test_decode_splitkv_repeat_is_deterministic():
    """The last-arriving split merges in fixed order: repeated launches agree bit for bit.

    One request with 16 KV heads gives 16 slots, which the default heuristic
    (about 64 workgroups) cuts into 4 splits, so the merge path is exercised.
    """
    outs = []
    for _ in range(3):
        out, _ = run_case([4097], 128, 16, 64, 64, (-1, -1), seed=1)
        outs.append(out)
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(outs[0], outs[2])


@pytest.mark.skipif(not is_xpu_available(), reason="Intel XPU not available")
@pytest.mark.skipif(not is_cri_device(), reason="Requires a CRI (Xe3P) device")
def test_decode_splitkv_concurrent_torch_streams_b8():
    """Two B8 split-KV requests must not share counters or partial results."""
    batch, length, heads_q, heads_kv, head_dim, page_size = 8, 4097, 8, 1, 64, 64
    pages_per_seq = (length + page_size - 1) // page_size
    num_pages = batch * pages_per_seq
    page_table = torch.arange(num_pages, dtype=torch.int32, device="xpu").view(
        batch, pages_per_seq
    )
    cache_seqlens = torch.full((batch,), length, dtype=torch.int32, device="xpu")
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device="xpu")

    inputs = []
    for seed in (11, 23):
        torch.manual_seed(seed)
        q = torch.randn(batch, heads_q, head_dim, dtype=torch.bfloat16, device="xpu")
        k = torch.randn(
            num_pages, page_size, heads_kv, head_dim, dtype=torch.bfloat16, device="xpu"
        )
        v = torch.randn_like(k)
        ref = reference(q, k, v, page_table, [length] * batch, -1)
        inputs.append((q, k, v, ref))

    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    caller_stream = torch.xpu.current_stream()
    for stream in streams:
        stream.wait_stream(caller_stream)

    for _ in range(3):
        outputs = []
        for stream, (q, k, v, ref) in zip(streams, inputs):
            with torch.xpu.stream(stream):
                out = flash_attn_with_kvcache(
                    q,
                    k,
                    v,
                    cache_seqlens=cache_seqlens,
                    page_table=page_table,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=1,
                    max_seqlen_k=length,
                    softmax_scale=head_dim**-0.5,
                    causal=True,
                    window_size=(-1, -1),
                    rotary_interleaved=False,
                )
                outputs.append((out, ref))
        torch.xpu.synchronize()
        for out, ref in outputs:
            torch.testing.assert_close(out.float().cpu(), ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
