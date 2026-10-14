import pytest
import sgl_kernel  # noqa: F401
import torch


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("experts,alignment", [(128, 4), (256, 4), (512, 16)])
def test_muse_mxfp4_small_permutation(index_dtype, experts, alignment):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")

    # B=8, topk=8. Repeated experts exercise stable ordering and padding.
    tokens, topk = 8, 8
    ids_cpu = (
        torch.arange(tokens * topk, dtype=torch.int64).reshape(tokens, topk) * 7
    ) % 23
    ids = ids_cpu.to(device="xpu", dtype=index_dtype)
    pairs = tokens * topk
    upper_rows = pairs + (alignment - 1) * pairs
    counts = torch.empty(experts, device="xpu", dtype=torch.int32)
    perm_src = torch.empty(upper_rows, device="xpu", dtype=torch.int32)
    expanded_to_permuted = torch.empty(pairs, device="xpu", dtype=torch.int32)
    scale_map = torch.empty(upper_rows, 2, device="xpu", dtype=torch.int32)

    torch.ops.sgl_kernel.mxfp4_small_permutation(
        ids, counts, perm_src, expanded_to_permuted, scale_map, alignment
    )

    flat_ids = ids_cpu.flatten().tolist()
    raw_counts = [flat_ids.count(expert) for expert in range(experts)]
    expected_counts = [
        (count + alignment - 1) // alignment * alignment for count in raw_counts
    ]
    expected_perm_src = [0] * upper_rows
    expected_expanded_to_permuted = [0] * pairs
    expected_scale_map = [[row, 0] for row in range(upper_rows)]
    start = 0
    for expert, padded in enumerate(expected_counts):
        indices = [index for index, value in enumerate(flat_ids) if value == expert]
        for rank, index in enumerate(indices):
            row = start + rank
            expected_perm_src[row] = index // topk
            expected_expanded_to_permuted[index] = row
            expected_scale_map[row] = [start, padded]
        for row in range(start + len(indices), start + padded):
            expected_scale_map[row] = [start, -padded]
        start += padded

    assert counts.cpu().tolist() == expected_counts
    assert perm_src.cpu().tolist() == expected_perm_src
    assert expanded_to_permuted.cpu().tolist() == expected_expanded_to_permuted
    assert scale_map.cpu().tolist() == expected_scale_map
