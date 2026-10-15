"""Validate sampled production paged prefill against independent FP32 math."""

import argparse
import json

import sgl_kernel
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-batch", type=int, default=8)
    args = parser.parse_args()
    if args.serving_batch != 8:
        parser.error("Muse-Spark serving workload requires B=8")
    if not sgl_kernel.is_xe3_arch():
        parser.error("requires Xe3.5 attention capability")
    # Muse-Spark: B8, ISL4096, OSL16, Q/KV heads128/16, D64, BF16, TP1/EP1.
    batch, length, heads_q, heads_kv, dim, page = 8, 4096, 128, 16, 64, 64
    torch.manual_seed(42)
    pages_per_request = length // page
    pages = batch * pages_per_request
    queries = torch.randn(
        batch, length, heads_q, dim, device="xpu", dtype=torch.bfloat16
    )
    keys = torch.randn(pages, page, heads_kv, dim, device="xpu", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    table = (
        torch.randperm(pages, device="xpu")
        .to(torch.int32)
        .view(batch, pages_per_request)
    )
    lengths = torch.full((batch,), length, dtype=torch.int32, device="xpu")
    offsets = torch.arange(batch + 1, dtype=torch.int32, device="xpu") * length
    selected = [0, 63, 127, 2047, 2048, 4095]
    query_sample = queries[:, selected].cpu().float()
    keys_cpu, values_cpu, table_cpu = (
        keys.cpu().float(),
        values.cpu().float(),
        table.cpu().long(),
    )
    query_positions = torch.tensor(selected).view(-1, 1)
    key_positions = torch.arange(length).view(1, -1)
    for window_left in [-1, 2047]:
        actual = (
            flash_attn_with_kvcache(
                queries.flatten(0, 1),
                keys,
                values,
                cache_seqlens=lengths,
                page_table=table,
                cu_seqlens_q=offsets,
                max_seqlen_q=length,
                softmax_scale=dim**-0.5,
                causal=True,
                window_size=(window_left, 0),
            )
            .view_as(queries)[:, selected]
            .float()
            .cpu()
        )
        expected = torch.empty_like(query_sample)
        masked = key_positions > query_positions
        if window_left >= 0:
            masked = masked | (key_positions < query_positions - window_left)
        for request in range(batch):
            keys_request = keys_cpu[table_cpu[request]].reshape(length, heads_kv, dim)
            values_request = values_cpu[table_cpu[request]].reshape(
                length, heads_kv, dim
            )
            query_request = query_sample[request].reshape(
                len(selected), heads_kv, heads_q // heads_kv, dim
            )
            scores = (
                torch.einsum("qhgd,khd->hgqk", query_request, keys_request) * dim**-0.5
            )
            probabilities = scores.masked_fill(masked, -float("inf")).softmax(dim=-1)
            expected[request] = torch.einsum(
                "hgqk,khd->qhgd", probabilities, values_request
            ).reshape(len(selected), heads_q, dim)
        torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
        print(
            json.dumps(
                {
                    "accuracy": "PASS",
                    "batch": batch,
                    "length": length,
                    "window_left": window_left,
                    "query_indices": selected,
                    "max_abs": (actual - expected).abs().max().item(),
                    "atol": 5e-3,
                    "rtol": 5e-3,
                    "reference": "FP32 CPU math on actual BF16 inputs",
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
