"""Compare old and patched SGLang position wrappers with the AOT SYCL op."""

import importlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from statistics import median

import torch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spread(samples):
    values = sorted(samples)
    return values[6], values[24]


old = load_module("position_old_main", "/workspace/artifacts/position/main_position.py")
patched = load_module(
    "position_patched_main",
    "/root/worktrees/sglang/muse-position-triton-split-main/python/sglang/kernels/ops/attention/position.py",
)
importlib.import_module("position_aot_after")
assert torch.xpu.is_available()

order = os.environ.get("POSITION_ORDER", "old-first")
assert order in ("old-first", "patched-first")
print(
    json.dumps(
        {
            "meta": {
                "order": order,
                "device": torch.xpu.get_device_name(0),
                "dtype": "int32 lengths and prefix, int64 positions, int32 starts",
                "layout": "contiguous",
                "timing": "torch.profiler XPU kernel event duration, microseconds",
                "samples": 31,
            }
        }
    ),
    flush=True,
)

cases = [
    ([1] * 8, 0),
    ([512] * 8, 0),
    ([1024] * 8, 0),
    ([1536] * 8, 0),
    ([1792] * 8, 0),
    ([2048] * 8, 0),
    ([4096] * 7, 0),
    ([4096] * 8, 4096),
    ([8193] * 8, 8193),
    ([4096] + [0] * 7, 4096),
    ([8193] + [0] * 7, 8193),
    ([1, 4097, 33], 4097),
    ([0, 17, 8193, 1], 8193),
]

for lengths, hint in cases:
    batch, total = len(lengths), sum(lengths)
    seq = torch.tensor(lengths, dtype=torch.int32, device="xpu")
    prefix = torch.zeros(batch, dtype=torch.int32, device="xpu")
    if batch == 3:
        prefix = torch.tensor([2, 4, 6], dtype=torch.int32, device="xpu")
    if batch == 4:
        prefix = torch.empty(0, dtype=torch.int32, device="xpu")
    positions_sycl = torch.empty(total, dtype=torch.int64, device="xpu")
    starts_sycl = torch.empty(batch, dtype=torch.int32, device="xpu")

    def old_call():
        return old.compute_position_triton(prefix, seq, total)

    def patched_call():
        return patched.compute_position_triton(prefix, seq, total, hint)

    def sycl_call():
        if hint > 0:
            torch.ops.sgl_kernel.muse_compute_position_with_hint(
                positions_sycl, starts_sycl, prefix, seq, hint
            )
        else:
            torch.ops.sgl_kernel.muse_compute_position(
                positions_sycl, starts_sycl, prefix, seq
            )
        return positions_sycl, starts_sycl

    expected = torch.cat(
        [
            torch.arange(size, dtype=torch.int64, device="xpu")
            + (prefix[i] if prefix.numel() else 0)
            for i, size in enumerate(lengths)
        ]
    )
    expected_starts = torch.tensor(
        [sum(lengths[:i]) for i in range(batch)], dtype=torch.int32, device="xpu"
    )
    for fn in (old_call, patched_call, sycl_call):
        positions, starts = fn()
        torch.xpu.synchronize()
        torch.testing.assert_close(positions, expected, rtol=0, atol=0)
        torch.testing.assert_close(starts, expected_starts, rtol=0, atol=0)
    print(json.dumps({"accuracy": "PASS", "lengths": lengths, "hint": hint}), flush=True)

    calls = (
        (old_call, patched_call, sycl_call)
        if order == "old-first"
        else (patched_call, old_call, sycl_call)
    )
    for _ in range(5):
        for fn in calls:
            fn()
    torch.xpu.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU]
    ) as profile:
        for _ in range(31):
            for fn in calls:
                fn()
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.json"
        profile.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
    kernels = [
        (float(event["ts"]), event["name"], float(event["dur"]))
        for event in events
        if event.get("cat") == "kernel"
        and (
            "compute_position_kernel" in event.get("name", "")
            or "MuseComputePositionKernel" in event.get("name", "")
        )
    ]
    kernels.sort()
    assert len(kernels) == 93, (lengths, len(kernels), kernels[:6])
    raw = {"old": [], "patched": [], "sycl": []}
    names = (
        ("old", "patched", "sycl")
        if order == "old-first"
        else ("patched", "old", "sycl")
    )
    for start in range(0, len(kernels), 3):
        group = kernels[start : start + 3]
        assert all("compute_position_kernel" in group[i][1] for i in (0, 1))
        assert "MuseComputePositionKernel" in group[2][1]
        for i, name in enumerate(names):
            raw[name].append(group[i][2])
    print(
        json.dumps(
            {
                "lengths": lengths,
                "hint": hint,
                "raw_us": raw,
                "p50_us": {name: median(values) for name, values in raw.items()},
                "p20_p80_us": {name: spread(values) for name, values in raw.items()},
            }
        ),
        flush=True,
    )
    torch.xpu.synchronize()
