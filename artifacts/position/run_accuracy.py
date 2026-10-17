import os
import subprocess
import sys

cases = [
    "test_compute_position_with_prefix",
    "test_compute_position_multiple_workgroups_irregular",
    "test_compute_position_multiple_workgroups_no_prefix",
    "test_compute_position_with_host_max_hint_skew_b8[4096]",
    "test_compute_position_with_host_max_hint_skew_b8[8193]",
]
for case in cases:
    command = [
        sys.executable,
        "-c",
        "import torch,importlib,os,pytest,sys; importlib.import_module(os.environ['POSITION_EXTENSION_NAME']); sys.exit(pytest.main(['-q',sys.argv[1]]))",
        "tests/test_compute_position.py::" + case,
    ]
    print("command:", command, flush=True)
    result = subprocess.run(command, env=os.environ.copy(), check=False)
    print("result:", case, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)

import torch

from main_position import compute_position_kernel
from main_position_split import compute_position_kernel_split

for lengths, prefix_values in [
    ([1, 4097, 33], [2, 4, 6]),
    ([0, 17, 8193, 1], []),
]:
    batch = len(lengths)
    seq = torch.tensor(lengths, dtype=torch.int32, device="xpu")
    prefix = torch.tensor(prefix_values, dtype=torch.int32, device="xpu")
    positions = torch.empty(sum(lengths), dtype=torch.int64, device="xpu")
    split_positions = torch.empty_like(positions)
    starts = torch.empty(batch, dtype=torch.int32, device="xpu")
    split_starts = torch.empty_like(starts)
    has_prefix = len(prefix_values) == batch
    groups = max(1, (max(lengths) + 511) // 512)
    compute_position_kernel[(batch,)](
        positions, starts, prefix, seq, has_prefix
    )
    compute_position_kernel_split[(batch, groups)](
        split_positions, split_starts, prefix, seq, has_prefix, groups
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(split_positions, positions, rtol=0, atol=0)
    torch.testing.assert_close(split_starts, starts, rtol=0, atol=0)
    print("split accuracy:", lengths, prefix_values, "PASS", flush=True)
