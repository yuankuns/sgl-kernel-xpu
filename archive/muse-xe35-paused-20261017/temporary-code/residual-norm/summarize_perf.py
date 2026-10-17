import argparse
import json
import statistics
from pathlib import Path


def read_cases_xe35(path):
    cases = {}
    for line in path.read_text().splitlines():
        if line.startswith('{"provider"'):
            case = json.loads(line)
            key = (case["rows"], case["post_mode"], case["round_pre_norm_input"], case["output_fp32"])
            cases[key] = case
    return cases


def main_xe35():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    runs = {name: read_cases_xe35(root / f"{args.prefix}perf_{name}.log") for name in ["B1", "A1", "A2", "B2"]}
    assert all(len(cases) == 24 for cases in runs.values())
    results = []
    table = ["| Rows | Post mode | Round pre | FP32 widen | Source µs | Port µs | Change |",
             "|---:|---|---|---|---:|---:|---:|"]
    for key in runs["B1"]:
        assert all(key in cases for cases in runs.values())
        source = statistics.median(runs["A1"][key]["samples_us"] + runs["A2"][key]["samples_us"])
        port = statistics.median(runs["B1"][key]["samples_us"] + runs["B2"][key]["samples_us"])
        delta = (port / source - 1) * 100
        result = {"case": key, "source_us": source, "port_us": port, "delta_pct": delta,
                  "retest_required": delta > 5,
                  "individual_p50_us": {name: cases[key]["p50_us"] for name, cases in runs.items()}}
        results.append(result)
        table.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {source:.3f} | {port:.3f} | {delta:+.2f}% |")
    (root / f"{args.prefix}perf_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    (root / f"{args.prefix}perf_table.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print("retest_required", [result["case"] for result in results if result["retest_required"]])


if __name__ == "__main__":
    main_xe35()
