"""Compare the current SYCL position kernel to a split-grid upstream Triton variant."""

import importlib
import json
import os
from statistics import median

import torch

from bench_position_occupancy import measure, spread
from main_position_split import compute_position_kernel_split

importlib.import_module(os.environ["POSITION_EXTENSION_NAME"])

for length in (1, 128, 512, 1024, 1536, 2048, 4096, 8193):
    batch = 8
    seq = torch.full((batch,), length, dtype=torch.int32, device="xpu")
    prefix = torch.zeros(batch, dtype=torch.int32, device="xpu")
    positions_t = torch.empty(batch * length, dtype=torch.int64, device="xpu")
    positions_s = torch.empty_like(positions_t)
    starts_t = torch.empty(batch, dtype=torch.int32, device="xpu")
    starts_s = torch.empty_like(starts_t)
    groups = max(1, (length + 511) // 512)

    def split_triton():
        compute_position_kernel_split[(batch, groups)](
            positions_t, starts_t, prefix, seq, True, groups
        )

    def sycl():
        torch.ops.sgl_kernel.muse_compute_position(positions_s, starts_s, prefix, seq)

    split_triton()
    sycl()
    torch.xpu.synchronize()
    torch.testing.assert_close(positions_s, positions_t, rtol=0, atol=0)
    torch.testing.assert_close(starts_s, starts_t, rtol=0, atol=0)
    t, s = measure(split_triton, sycl, 31, False)
    print(
        json.dumps(
            {
                "length": length,
                "groups": groups,
                "split_us": t,
                "sycl_us": s,
                "split_p50_us": median(t),
                "sycl_p50_us": median(s),
                "split_p20_p80_us": spread(t),
                "sycl_p20_p80_us": spread(s),
            }
        ),
        flush=True,
    )
    torch.xpu.synchronize()
