import argparse
import json
from pathlib import Path


def drift(case, first, second):
    medians = case["individual_p50_us"]
    return (medians[second] / medians[first] - 1) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    initial = json.loads((root / f"{args.prefix}perf_summary.json").read_text())
    retest_path = root / f"{args.prefix}retest_perf_summary.json"
    retest = {tuple(case["case"]): case for case in json.loads(retest_path.read_text())} if retest_path.exists() else {}
    records = []
    for case in initial:
        key = tuple(case["case"])
        repeat = retest.get(key)
        if case["retest_required"]:
            assert repeat is not None, f">5% regression without matched retest: {key}"
            status = "persistent >5% regression; explanation required" if repeat["retest_required"] else "initial >5% regression not reproduced; drift review required"
        else:
            status = "initial within 5%"
            if repeat is not None and repeat["retest_required"]:
                status = "retest >5% regression; explanation required"
        records.append({
            "case": case["case"], "initial_delta_pct": case["delta_pct"],
            "initial_source_AA_drift_pct": drift(case, "A1", "A2"),
            "initial_port_BB_drift_pct": drift(case, "B1", "B2"),
            "retest_delta_pct": None if repeat is None else repeat["delta_pct"],
            "retest_source_AA_drift_pct": None if repeat is None else drift(repeat, "A1", "A2"),
            "retest_port_BB_drift_pct": None if repeat is None else drift(repeat, "B1", "B2"),
            "status": status,
            "manual_explanation_required": status != "initial within 5%",
        })
    (root / f"{args.prefix}perf_review.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"cases": len(records), "diagnostic_only": args.prefix == "diagnostic_",
                      "manual_explanation_required": [case["case"] for case in records if case["manual_explanation_required"]]}, indent=2))


if __name__ == "__main__":
    main()
