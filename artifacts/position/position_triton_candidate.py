import torch
import triton
import triton.language as tl


def compute_position_triton(
    extend_prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_seq_lens_sum: int,
    max_seq_len_hint: int = -1,
):
    """Compute positions. It is a fused version of `compute_position_torch`."""
    batch_size = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == batch_size

    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    extend_start_loc = torch.empty(
        batch_size, dtype=torch.int32, device=extend_seq_lens.device
    )

    if batch_size == 0:
        return positions, extend_start_loc

    # Spread a long request across programs while keeping one program for
    # short requests. A stale hint changes only occupancy, not coverage.
    average_groups = triton.cdiv(extend_seq_lens_sum, batch_size * 512)
    hinted_groups = triton.cdiv(max_seq_len_hint, 512) if max_seq_len_hint > 0 else 1
    useful_groups = max(1, triton.cdiv(extend_seq_lens_sum, 512))
    groups_per_request = min(
        useful_groups, max(1, average_groups, hinted_groups)
    )
    compute_position_kernel[(batch_size, groups_per_request)](
        positions,
        extend_start_loc,
        extend_prefix_lens,
        extend_seq_lens,
        has_prefix,
        groups_per_request,
    )

    return positions, extend_start_loc


@triton.jit
def compute_position_kernel(
    positions,
    extend_start_loc,
    extend_prefix_lens,
    extend_seq_lens,
    has_prefix: tl.constexpr,
    groups_per_request: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)

    prefix_len = tl.load(extend_prefix_lens + pid) if has_prefix else 0
    seq_len = tl.load(extend_seq_lens + pid)

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_seq_lens + i)

    for i in range(tile * BLOCK_SIZE, seq_len, groups_per_request * BLOCK_SIZE):
        offset = tl.arange(0, BLOCK_SIZE) + i
        tl.store(
            positions + cumsum_start + offset,
            prefix_len + offset,
            mask=offset < seq_len,
        )
    if tile == 0:
        tl.store(extend_start_loc + pid, cumsum_start)
