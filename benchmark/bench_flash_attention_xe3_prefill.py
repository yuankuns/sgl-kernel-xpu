"""Validate and time production BF16 causal prefill on the B8 1+7 split."""

import argparse
import json

import sgl_kernel  # noqa: F401 - registers the production attention op
import torch
import triton.testing
from sgl_kernel.flash_attn import flash_attn_with_kvcache


@torch.inference_mode()
def check_tail_shapes():
    """Kernel accuracy only: paged GQA and partial query/key tiles.

    The production Xe3.5 Muse-Spark path uses BF16.
    """
    for dtype in (torch.bfloat16,):
        for length in (33, 129, 257):
            torch.xpu.synchronize()
            torch.manual_seed(42)
            batch = 8
            page, hq, hkv, dim = 64, 128, 16, 64
            pages_per_seq = (length + page - 1) // page
            pages = batch * pages_per_seq
            q = torch.randn(batch * length, hq, dim, device="xpu", dtype=dtype)
            k = torch.randn(pages, page, hkv, dim, device="xpu", dtype=dtype)
            v = torch.randn_like(k)
            table = (
                torch.randperm(pages, device="xpu")
                .to(torch.int32)
                .view(batch, pages_per_seq)
            )
            cu = torch.arange(batch + 1, device="xpu", dtype=torch.int32) * length
            lengths = torch.full((batch,), length, dtype=torch.int32, device="xpu")
            # Independent FP32 CPU reference, using the actual shuffled pages.
            order = table.cpu().long()
            keys = k.cpu()[order].reshape(batch, -1, hkv, dim)[:, :length].float()
            values = v.cpu()[order].reshape(batch, -1, hkv, dim)[:, :length].float()
            keys = keys.repeat_interleave(hq // hkv, dim=2).transpose(1, 2)
            values = values.repeat_interleave(hq // hkv, dim=2).transpose(1, 2)
            queries = q.cpu().float().view(batch, length, hq, dim).transpose(1, 2)
            scores = queries @ keys.transpose(2, 3)
            scores *= dim**-0.5
            scores.masked_fill_(
                torch.ones(length, length, dtype=torch.bool).triu(1), -float("inf")
            )
            expected = (scores.softmax(dim=-1) @ values).transpose(1, 2)
            expected = expected.reshape(batch * length, hq, dim).to(dtype)
            actual = flash_attn_with_kvcache(
                q=q,
                k_cache=k,
                v_cache=v,
                page_table=table,
                cache_seqlens=lengths,
                cu_seqlens_q=cu,
                max_seqlen_q=length,
                softmax_scale=dim**-0.5,
                causal=True,
                window_size=(-1, -1),
            )
            torch.testing.assert_close(actual.cpu(), expected, atol=0.01, rtol=0.01)
            print(
                json.dumps(
                    {
                        "accuracy": "PASS",
                        "length": length,
                        "dtype": str(dtype),
                        "reference": "FP32 CPU causal GQA",
                        "path": "flash_attn_with_kvcache",
                    }
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-batch", type=int, default=8)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=500)
    parser.add_argument("--check-tails", action="store_true")
    args = parser.parse_args()
    if args.serving_batch != 8:
        parser.error("Muse-Spark serving workload requires B=8")
    if args.warmup_ms < 1 or args.rep_ms < 1:
        parser.error("warmup and repetition time must be positive")
    if torch.xpu.get_device_properties(0).device_id != 0x674F:
        raise RuntimeError("requires CRI 0x674f")
    if args.check_tails:
        check_tail_shapes()
    for requests in (1, 7):
        torch.xpu.synchronize()
        torch.manual_seed(42)
        length, page, hq, hkv, dim = 4096, 64, 128, 16, 64
        pages = requests * length // page
        q = torch.randn(requests * length, hq, dim, device="xpu", dtype=torch.bfloat16)
        k = torch.randn(pages, page, hkv, dim, device="xpu", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        table = torch.randperm(pages, device="xpu").to(torch.int32).view(requests, -1)
        cu = torch.arange(requests + 1, device="xpu", dtype=torch.int32) * length
        lengths = torch.full((requests,), length, device="xpu", dtype=torch.int32)

        def launch():
            return flash_attn_with_kvcache(
                q=q,
                k_cache=k,
                v_cache=v,
                page_table=table,
                cache_seqlens=lengths,
                cu_seqlens_q=cu,
                max_seqlen_q=length,
                softmax_scale=dim**-0.5,
                causal=True,
                window_size=(-1, -1),
            )

        median_ms, p20_ms, p80_ms = triton.testing.do_bench(
            launch, warmup=args.warmup_ms, rep=args.rep_ms, quantiles=[0.5, 0.2, 0.8]
        )
        print(
            json.dumps(
                {
                    "serving_batch": 8,
                    "step_requests": requests,
                    "tokens": requests * length,
                    "median_ms": median_ms,
                    "p20_ms": p20_ms,
                    "p80_ms": p80_ms,
                    "warmup_ms": args.warmup_ms,
                    "rep_ms": args.rep_ms,
                    "precision": "BF16 input/output, FP32 accumulation",
                    "path": "flash_attn_with_kvcache",
                }
            ),
            flush=True,
        )
        del q, k, v, table, cu, lengths
        torch.xpu.synchronize()


if __name__ == "__main__":
    main()
