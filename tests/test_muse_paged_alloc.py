import os
import time

import pytest
import torch

if library := os.environ.get("MUSE_MEMORY_TEST_LIBRARY"):
    torch.ops.load_library(library)
else:
    import sgl_kernel  # noqa: F401


@pytest.fixture(autouse=True)
def cool_between_device_shapes():
    time.sleep(2)


@pytest.mark.parametrize("last_dtype", [torch.int32, torch.int64])
def test_muse_alloc_extend_crosses_page_boundaries(last_dtype):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    device = "xpu"
    pre = torch.tensor([2, 4, 7, 1], device=device, dtype=torch.int64)
    seq = torch.tensor([5, 6, 10, 3], device=device, dtype=torch.int64)
    last = torch.tensor([101, -1, 202, 400], device=device, dtype=last_dtype)
    free_pages = torch.tensor([10, 20, 30], device=device, dtype=torch.int64)
    out = torch.empty(10, device=device, dtype=torch.int64)
    torch.ops.sgl_kernel.muse_alloc_extend(pre, seq, last, free_pages, out, 4)
    expected = torch.tensor(
        [102, 103, 40, 80, 81, 203, 120, 121, 401, 402],
        device=device,
        dtype=torch.int64,
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("last_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("page_size", [4, 32, 64, 128])
def test_muse_alloc_decode_reuses_or_opens_page(last_dtype, page_size):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    device = "xpu"
    seq = torch.tensor(
        [page_size, page_size + 1, 2 * page_size, 2 * page_size + 1],
        device=device,
        dtype=torch.int64,
    )
    last = torch.tensor([23, 31, 39, 47], device=device, dtype=last_dtype)
    free_pages = torch.tensor([11, 12], device=device, dtype=torch.int64)
    out = torch.empty_like(seq)
    torch.ops.sgl_kernel.muse_alloc_decode(seq, last, free_pages, out, page_size)
    expected = torch.tensor(
        [24, 11 * page_size, 40, 12 * page_size], device=device, dtype=torch.int64
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("last_dtype", [torch.int32, torch.int64])
def test_muse_alloc_extend_page64_partial_and_new_pages(last_dtype):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    pre = torch.tensor([62, 64, 127], device="xpu", dtype=torch.int64)
    seq = torch.tensor([66, 130, 130], device="xpu", dtype=torch.int64)
    last = torch.tensor([300, -1, 511], device="xpu", dtype=last_dtype)
    free_pages = torch.tensor([10, 20, 30, 40], device="xpu", dtype=torch.int64)
    out = torch.empty(73, device="xpu", dtype=torch.int64)
    torch.ops.sgl_kernel.muse_alloc_extend(pre, seq, last, free_pages, out, 64)
    expected = torch.tensor(
        [301, 302, 640, 641] + list(range(1280, 1344)) + [1920, 1921, 512, 2560, 2561],
        device="xpu",
        dtype=torch.int64,
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("page_size", [7, 32, 64, 128])
@pytest.mark.parametrize("last_dtype", [torch.int32, torch.int64])
def test_muse_alloc_extend_and_write_matches_split_with_prefix(page_size, last_dtype):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    prefix_lens = torch.tensor(
        [2, page_size, page_size - 1], device="xpu", dtype=torch.int64
    )
    seq_lens = torch.tensor(
        [page_size + 2, page_size + 3, page_size + 2], device="xpu", dtype=torch.int64
    )
    extend_lens = seq_lens - prefix_lens
    last = torch.tensor([101, 200, 301], device="xpu", dtype=last_dtype)
    free_pages = torch.arange(10, 16, device="xpu", dtype=torch.int64)
    total_extend = int(extend_lens.sum().item())
    split_out = torch.empty(total_extend, device="xpu", dtype=torch.int64)
    fused_out = torch.empty_like(split_out)
    pool_width = int(seq_lens.max().item())
    split_pool = torch.full((3, pool_width), -1, device="xpu", dtype=torch.int32)
    fused_pool = split_pool.clone()
    req_indices = torch.tensor([2, 0, 1], device="xpu", dtype=torch.int64)
    prefixes = [
        torch.arange(int(length), device="xpu", dtype=torch.int64) + 1000 * (index + 1)
        for index, length in enumerate(prefix_lens.tolist())
    ]
    pointers = torch.tensor(
        [tensor.data_ptr() for tensor in prefixes], device="xpu", dtype=torch.uint64
    )
    torch.ops.sgl_kernel.muse_alloc_extend(
        prefix_lens, seq_lens, last, free_pages, split_out, page_size
    )
    torch.ops.sgl_kernel.muse_write_req_to_token_pool(
        split_pool, req_indices, pointers, prefix_lens, seq_lens, extend_lens, split_out
    )
    torch.ops.sgl_kernel.muse_alloc_extend_and_write(
        prefix_lens,
        seq_lens,
        last,
        free_pages,
        fused_out,
        fused_pool,
        req_indices,
        pointers,
        page_size,
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(fused_out.cpu(), split_out.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(fused_pool.cpu(), split_pool.cpu(), rtol=0, atol=0)
