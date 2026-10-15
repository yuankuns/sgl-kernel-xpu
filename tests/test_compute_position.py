import pytest
import sgl_kernel  # noqa: F401
import torch


def test_compute_position_with_prefix():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    seq = torch.tensor([2, 0, 3], device="xpu", dtype=torch.int32)
    prefix = torch.tensor([4, 2, 7], device="xpu", dtype=torch.int32)
    positions = torch.empty(5, device="xpu", dtype=torch.int64)
    starts = torch.empty(3, device="xpu", dtype=torch.int32)
    torch.ops.sgl_kernel.muse_compute_position(positions, starts, prefix, seq)
    torch.testing.assert_close(
        positions, torch.tensor([4, 5, 7, 8, 9], device="xpu", dtype=torch.int64)
    )
    torch.testing.assert_close(
        starts, torch.tensor([0, 2, 2], device="xpu", dtype=torch.int32)
    )


def test_compute_position_multiple_workgroups_irregular():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    lengths = torch.tensor([1, 4097, 33], device="xpu", dtype=torch.int64)
    prefix = torch.tensor([2, 4, 6], device="xpu", dtype=torch.int64)
    positions = torch.empty(4131, device="xpu", dtype=torch.int64)
    starts = torch.empty(3, device="xpu", dtype=torch.int32)
    torch.ops.sgl_kernel.muse_compute_position(positions, starts, prefix, lengths)
    expected = torch.cat(
        [
            torch.arange(int(length), device="xpu") + int(base)
            for length, base in zip(lengths.cpu(), prefix.cpu())
        ]
    )
    torch.testing.assert_close(positions, expected)
    torch.testing.assert_close(
        starts, torch.tensor([0, 1, 4098], device="xpu", dtype=torch.int32)
    )


def test_compute_position_multiple_workgroups_no_prefix():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    lengths = torch.tensor([2049, 1], device="xpu", dtype=torch.int32)
    prefix = torch.empty(0, device="xpu", dtype=torch.int32)
    positions = torch.empty(2050, device="xpu", dtype=torch.int64)
    starts = torch.empty(2, device="xpu", dtype=torch.int32)
    torch.ops.sgl_kernel.muse_compute_position(positions, starts, prefix, lengths)
    torch.testing.assert_close(
        positions,
        torch.cat([torch.arange(2049, device="xpu"), torch.arange(1, device="xpu")]),
    )
    torch.testing.assert_close(
        starts, torch.tensor([0, 2049], device="xpu", dtype=torch.int32)
    )


@pytest.mark.parametrize("long_request", [4096, 8193])
def test_compute_position_with_host_max_hint_skew_b8(long_request):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    lengths = torch.tensor([long_request] + [0] * 7, device="xpu", dtype=torch.int32)
    prefix = torch.arange(8, device="xpu", dtype=torch.int32) * 3
    positions = torch.empty(long_request, device="xpu", dtype=torch.int64)
    starts = torch.empty(8, device="xpu", dtype=torch.int32)
    torch.ops.sgl_kernel.muse_compute_position_with_hint(
        positions, starts, prefix, lengths, long_request
    )
    torch.testing.assert_close(positions, torch.arange(long_request, device="xpu"))
    torch.testing.assert_close(
        starts,
        torch.tensor(
            [0] + [long_request] * 7,
            device="xpu",
            dtype=torch.int32,
        ),
    )
    # A deliberately low hint must remain correct; it only affects occupancy.
    torch.ops.sgl_kernel.muse_compute_position_with_hint(
        positions, starts, prefix, lengths, 1
    )
    torch.testing.assert_close(positions, torch.arange(long_request, device="xpu"))
    with pytest.raises(RuntimeError, match="nonnegative"):
        torch.ops.sgl_kernel.muse_compute_position_with_hint(
            positions, starts, prefix, lengths, -1
        )
