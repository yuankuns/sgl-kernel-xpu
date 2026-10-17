"""Summarize paired profiler logs without discarding raw samples."""

import json
from pathlib import Path


def read_runs(path):
    meta = None
    data = {}
    for line in Path(path).read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "meta" in row and "batch" in row["meta"]:
            meta = row["meta"]
        elif "main_p50_us" in row:
            key = (
                meta["batch"],
                meta["reverse_order"],
                meta["skew"],
                meta["max_seq_len_hint"],
                row["length"],
            )
            data[key] = row
    return data


before = read_runs("artifacts/position/local-int32-before.log")
after = read_runs("artifacts/position/local-int32-after.log")
print("batch reverse skew hint length main_before main_after sycl_before sycl_after sycl_delta_pct")
for key in sorted(before.keys() & after.keys()):
    b, a = before[key], after[key]
    delta = 100 * (a["branch_p50_us"] / b["branch_p50_us"] - 1)
    print(*key, b["main_p50_us"], a["main_p50_us"], b["branch_p50_us"], a["branch_p50_us"], round(delta, 2))
