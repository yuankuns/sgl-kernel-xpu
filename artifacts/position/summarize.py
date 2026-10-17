import json
import sys
from statistics import median

for filename in sys.argv[1:]:
    print(f"\n{filename}")
    print("batch length main_p50 sycl_p50 delta_us main_p20-p80 sycl_p20-p80")
    batch = None
    for line in open(filename):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "meta" in row:
            batch = row["meta"]["batch"]
        if "length" not in row or "branch_sycl_us" not in row:
            continue
        main = sorted(row["main_triton_us"])
        sycl = sorted(row["branch_sycl_us"])
        n = len(main)
        assert n == len(sycl)
        p20 = round(0.2 * (n - 1))
        p80 = round(0.8 * (n - 1))
        print(
            batch,
            row["length"],
            median(main),
            median(sycl),
            round(median(main) - median(sycl), 3),
            f"{main[p20]}-{main[p80]}",
            f"{sycl[p20]}-{sycl[p80]}",
        )
