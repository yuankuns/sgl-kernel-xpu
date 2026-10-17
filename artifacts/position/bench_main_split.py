import json
import os
import tempfile
from pathlib import Path
from statistics import median

import torch

from main_position import compute_position_kernel
from main_position_split import compute_position_kernel_split

order = os.environ.get("POSITION_SPLIT_ORDER", "split-first")
assert order in ("split-first", "main-first")
print(json.dumps({"meta": {"length_dtype": "int32", "order": order}}), flush=True)
for length in (16, 128, 512, 1024, 1536, 2048, 3072, 4096):
    batch = 8
    seq = torch.full((batch,), length, device='xpu', dtype=torch.int32)
    prefix = torch.zeros(batch, device='xpu', dtype=torch.int32)
    pos_old = torch.empty(batch * length, device='xpu', dtype=torch.int64)
    pos_split = torch.empty_like(pos_old)
    start_old = torch.empty(batch, device='xpu', dtype=torch.int32)
    start_split = torch.empty_like(start_old)
    groups = max(1, (length + 511) // 512)
    def old(): compute_position_kernel[(batch,)](pos_old, start_old, prefix, seq, True)
    def split(): compute_position_kernel_split[(batch, groups)](pos_split, start_split, prefix, seq, True, groups)
    for _ in range(5):
        if order == "split-first":
            split(); old()
        else:
            old(); split()
    torch.xpu.synchronize()
    torch.testing.assert_close(pos_old, pos_split, atol=0, rtol=0)
    torch.testing.assert_close(start_old, start_split, atol=0, rtol=0)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU]) as profile:
        for _ in range(31):
            if order == "split-first":
                split(); old()
            else:
                old(); split()
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'trace.json'
        profile.export_chrome_trace(str(path))
        events = json.loads(path.read_text())['traceEvents']
    def samples(name): return [float(e['dur']) for e in events if e.get('cat') == 'kernel' and name == e.get('name','')]
    o, s = samples('compute_position_kernel'), samples('compute_position_kernel_split')
    assert len(o) == len(s) == 31, (len(o), len(s), [e['name'] for e in events if e.get('cat') == 'kernel'][:5])
    print(json.dumps({'length':length, 'groups':groups, 'main_us':o, 'split_us':s, 'main_p50':median(o), 'split_p50':median(s), 'main_span':max(o)-min(o), 'split_span':max(s)-min(s)}), flush=True)
    torch.xpu.synchronize()
