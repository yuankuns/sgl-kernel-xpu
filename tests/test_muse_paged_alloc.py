import pytest
import sgl_kernel  # noqa: F401
import torch


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
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("last_dtype", [torch.int32, torch.int64])
def test_muse_alloc_decode_reuses_or_opens_page(last_dtype):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    device = "xpu"
    seq = torch.tensor([4, 5, 8, 9], device=device, dtype=torch.int64)
    last = torch.tensor([23, 31, 39, 47], device=device, dtype=last_dtype)
    free_pages = torch.tensor([11, 12], device=device, dtype=torch.int64)
    out = torch.empty_like(seq)
    torch.ops.sgl_kernel.muse_alloc_decode(seq, last, free_pages, out, 4)
    expected = torch.tensor([24, 44, 40, 48], device=device, dtype=torch.int64)
    torch.testing.assert_close(out, expected)


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
    torch.testing.assert_close(out, expected)
