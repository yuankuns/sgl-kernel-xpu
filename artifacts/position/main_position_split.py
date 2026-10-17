"""Experimental split-grid variant of the upstream Triton position kernel."""

import triton
import triton.language as tl


@triton.jit
def compute_position_kernel_split(
    positions,
    extend_start_loc,
    extend_prefix_lens,
    extend_seq_lens,
    has_prefix: tl.constexpr,
    groups_per_request: tl.constexpr,
):
    block_size: tl.constexpr = 512
    pid = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    prefix_len = tl.load(extend_prefix_lens + pid) if has_prefix else 0
    seq_len = tl.load(extend_seq_lens + pid)
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_seq_lens + i)
    for i in range(tile * block_size, seq_len, groups_per_request * block_size):
        offset = tl.arange(0, block_size) + i
        tl.store(positions + cumsum_start + offset, prefix_len + offset, mask=offset < seq_len)
    if tile == 0:
        tl.store(extend_start_loc + pid, cumsum_start)
