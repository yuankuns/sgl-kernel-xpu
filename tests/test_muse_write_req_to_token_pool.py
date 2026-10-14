import pytest
import sgl_kernel  # noqa: F401
import torch


def test_muse_write_req_to_token_pool_prefix_and_extension():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    device = "xpu"
    pool = torch.full((5, 8), -1, device=device, dtype=torch.int32)
    req_indices = torch.tensor([3, 1, 4], device=device, dtype=torch.int64)
    prefixes = [
        torch.tensor([20, 21], device=device, dtype=torch.int64),
        torch.empty(0, device=device, dtype=torch.int64),
        torch.tensor([30, 31, 32], device=device, dtype=torch.int64),
    ]
    pointers = torch.tensor(
        [p.data_ptr() for p in prefixes], device=device, dtype=torch.uint64
    )
    prefix_lens = torch.tensor([2, 0, 3], device=device, dtype=torch.int64)
    seq_lens = torch.tensor([4, 2, 4], device=device, dtype=torch.int64)
    extend_lens = torch.tensor([2, 2, 1], device=device, dtype=torch.int64)
    out_cache_loc = torch.tensor([10, 11, 12, 13, 14], device=device, dtype=torch.int64)
    torch.ops.sgl_kernel.muse_write_req_to_token_pool(
        pool,
        req_indices,
        pointers,
        prefix_lens,
        seq_lens,
        extend_lens,
        out_cache_loc,
    )
    torch.testing.assert_close(
        pool[3],
        torch.tensor(
            [20, 21, 10, 11, -1, -1, -1, -1], device=device, dtype=torch.int32
        ),
    )
    torch.testing.assert_close(
        pool[1],
        torch.tensor(
            [12, 13, -1, -1, -1, -1, -1, -1], device=device, dtype=torch.int32
        ),
    )
    torch.testing.assert_close(
        pool[4],
        torch.tensor(
            [30, 31, 32, 14, -1, -1, -1, -1], device=device, dtype=torch.int32
        ),
    )


def test_muse_write_req_to_token_pool_multigroup_irregular_prefix():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    pool = torch.full((3, 5200), -1, device="xpu", dtype=torch.int32)
    prefix = torch.arange(3001, device="xpu", dtype=torch.int64) + 20
    empty_prefix = torch.empty(0, device="xpu", dtype=torch.int64)
    pointers = torch.tensor(
        [prefix.data_ptr(), empty_prefix.data_ptr()], device="xpu", dtype=torch.uint64
    )
    req_indices = torch.tensor([2, 0], device="xpu", dtype=torch.int64)
    prefix_lens = torch.tensor([3001, 0], device="xpu", dtype=torch.int64)
    seq_lens = torch.tensor([5050, 1], device="xpu", dtype=torch.int64)
    extend_lens = torch.tensor([2049, 1], device="xpu", dtype=torch.int64)
    loc = torch.arange(2050, device="xpu", dtype=torch.int64) + 10000
    torch.ops.sgl_kernel.muse_write_req_to_token_pool(
        pool, req_indices, pointers, prefix_lens, seq_lens, extend_lens, loc
    )
    torch.testing.assert_close(pool[2, :5050], torch.cat((prefix, loc[:2049])).int())
    torch.testing.assert_close(
        pool[2, 5050:], torch.full((150,), -1, device="xpu", dtype=torch.int32)
    )
    torch.testing.assert_close(pool[0, 0], loc[-1].int())
    torch.testing.assert_close(
        pool[0, 1:], torch.full((5199,), -1, device="xpu", dtype=torch.int32)
    )
