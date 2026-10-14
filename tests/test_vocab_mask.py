import pytest
import sgl_kernel  # noqa: F401
import torch


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_vocab_mask_boundaries(dtype):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    org_start, org_end, padding = 100, 151, 13
    added_start, added_end = 200, 207
    values = [
        -1,
        0,
        org_start - 1,
        org_start,
        org_end - 1,
        org_end,
        added_start - 1,
        added_start,
        added_end - 1,
        added_end,
        100000,
    ]
    input_ids = torch.tensor(values * 3, dtype=dtype, device="xpu")
    mapped, invalid = torch.ops.sgl_kernel.vocab_mask(
        input_ids, org_start, org_end, padding, added_start, added_end
    )
    org_valid = (input_ids >= org_start) & (input_ids < org_end)
    added_valid = (input_ids >= added_start) & (input_ids < added_end)
    offset = (
        org_start * org_valid
        + (added_start - (org_end - org_start) - padding) * added_valid
    )
    valid = org_valid | added_valid
    torch.testing.assert_close(mapped, (input_ids - offset) * valid)
    torch.testing.assert_close(invalid, ~valid)


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_vocab_mask_tp2_shards(tp_rank, dtype):
    """Check both TP2 shard index mappings on one XPU without multi-GPU collectives."""
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")

    device = torch.device("xpu:0")
    ids = torch.tensor(
        [0, 511, 512, 1009, 1010, 1025, 1026, -1],
        device=device,
        dtype=dtype,
    )
    if tp_rank == 0:
        # 1010 original tokens are padded to 1024, then 16 added tokens.
        args = (0, 512, 0, 1010, 1026)
        mapped = [0, 511, 0, 0, 512, 527, 0, 0]
        invalid = [False, False, True, True, False, False, True, True]
    else:
        args = (512, 1010, 14, 1026, 1026)
        mapped = [0, 0, 0, 497, 0, 0, 0, 0]
        invalid = [True, True, False, False, True, True, True, True]

    actual_mapped, actual_invalid = torch.ops.sgl_kernel.vocab_mask(ids, *args)
    torch.testing.assert_close(
        actual_mapped, torch.tensor(mapped, device=device, dtype=torch.int64)
    )
    torch.testing.assert_close(
        actual_invalid, torch.tensor(invalid, device=device, dtype=torch.bool)
    )
    assert actual_mapped.device == device
    assert actual_invalid.device == device

    # Also exercise the framework dispatch, which must route XPU inputs here.
    from sglang.srt.layers.vocab_parallel_embedding import get_masked_input_and_mask

    mapped_from_layer, invalid_from_layer = get_masked_input_and_mask(ids, *args)
    torch.testing.assert_close(mapped_from_layer, actual_mapped)
    torch.testing.assert_close(invalid_from_layer, actual_invalid)


def test_vocab_mask_subgroup_size_boundary():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    ids = torch.tensor(
        [0, 511, 512, 1009, 1010, 1025, 1026, -1] * 4, device="xpu", dtype=torch.int64
    )
    mapped, invalid = torch.ops.sgl_kernel.vocab_mask(ids, 0, 512, 0, 1010, 1026)
    expected_mapped = torch.tensor(
        [0, 511, 0, 0, 512, 527, 0, 0] * 4, device="xpu", dtype=torch.int64
    )
    expected_invalid = torch.tensor(
        [False, False, True, True, False, False, True, True] * 4, device="xpu"
    )
    torch.testing.assert_close(mapped, expected_mapped)
    torch.testing.assert_close(invalid, expected_invalid)
