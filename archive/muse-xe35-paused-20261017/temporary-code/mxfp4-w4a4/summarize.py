import argparse
import json
import pathlib
import re
import statistics

import torch


def main_xe35():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, choices=(0, 4), default=0)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--precision-only", action="store_true")
    args = parser.parse_args()
    canonical = bool(re.fullmatch(r"canonical-(?:v[1-9][0-9]*(?:-icpx)?-)?(?:precision|performance)-", args.prefix))
    if not canonical and args.prefix not in ("", "diagnostic-", "fresh-diagnostic-", "compat-diagnostic-"):
        parser.error("Invalid evidence prefix")
    if canonical and args.precision_only != args.prefix.endswith("precision-"):
        parser.error("Canonical prefix and precision/performance mode must match")
    artifact = pathlib.Path(__file__).resolve().parent
    reports = {}
    for relative_round, implementation in ((0, "source"), (1, "port"), (2, "port"), (3, "source")):
        round_id = args.offset + relative_round
        if (artifact / f"{args.prefix}BAAB-{round_id}-{implementation}.exit").read_text().strip() != "0":
            raise RuntimeError(f"BAAB round {round_id} did not pass")
        for line in (artifact / f"{args.prefix}BAAB-{round_id}-{implementation}.log").read_text().splitlines():
            if not line.startswith("{"):
                continue
            data = json.loads(line)
            if "shape" in data:
                reports.setdefault(data["shape"], {})[relative_round] = data
    expected_shapes = {"256-8192-4096", "256-4096-4096", "257-4096-4096", "262144-4096-4096", "262144-8192-4096"}
    if set(reports) != expected_shapes:
        raise RuntimeError(f"incomplete production shape coverage: {set(reports)}")
    result = {"scope": "B8/ISL4096/OSL16/TP1/EP1", "acceptance_kind": "native-kernel diagnostic, not official full-package pass" if args.prefix and not args.prefix.startswith("canonical-") else "official full-package runtime", "shapes": {}}
    result["independent_reference_scope"] = "Original CPU FP64 oracle: max_rows=4 per expert, every output column of sampled rows; not an independent every-element production oracle"
    result["full_output_equivalence_scope"] = "When reported, all valid output rows and columns are compared using complete raw SHA256 across four rounds; this is source/port bitwise equivalence, not a full independent FP64 oracle"
    regression = False
    for shape, rounds in reports.items():
        if set(rounds) != {0, 1, 2, 3}:
            raise RuntimeError(f"incomplete rounds: {shape}")
        if args.precision_only and any(not report.get("precision_only") or report["samples_ms"] for report in rounds.values()):
            raise RuntimeError(f"Precision-only round contains performance work: {shape}")
        if not args.precision_only and any(not report["samples_ms"] for report in rounds.values()):
            raise RuntimeError(f"missing device timing samples: {shape}")
        if args.prefix in ("fresh-diagnostic-", "compat-diagnostic-") or canonical:
            inputs = rounds[0]["input_hashes"]
            if any(report["input_hashes"] != inputs for report in rounds.values()):
                raise RuntimeError(f"input/scale SHA256 mismatch: {shape}")
        if args.prefix == "compat-diagnostic-" or canonical:
            output_hash = rounds[0]["valid_output_hash"]
            if any(report["valid_output_hash"] != output_hash for report in rounds.values()):
                raise RuntimeError(f"Full valid-output bitwise SHA256 mismatch: {shape}")
        baseline = torch.load(rounds[0]["output_sample"], weights_only=True)
        for round_id in (1, 2, 3):
            actual = torch.load(rounds[round_id]["output_sample"], weights_only=True)
            torch.testing.assert_close(actual, baseline, rtol=0.005, atol=0.005)
        if args.precision_only:
            result["shapes"][shape] = {"source_port_accuracy": "passed at original 0.005/0.005", "valid_output_hash": output_hash, "valid_output_rows": rounds[0]["valid_output_rows"], "unwritten_slack_rows": rounds[0]["unwritten_slack_rows"], "full_valid_output_bitwise": "all four rounds SHA256 identical", "performance": "not run"}
            continue
        source_ms = statistics.median(rounds[0]["samples_ms"] + rounds[3]["samples_ms"])
        port_ms = statistics.median(rounds[1]["samples_ms"] + rounds[2]["samples_ms"])
        change = (port_ms / source_ms - 1) * 100
        result["shapes"][shape] = {"source_ms": source_ms, "port_ms": port_ms, "change_pct": change, "source_port_accuracy": "passed at original 0.005/0.005", "needs_regression_retest": change > 5}
        if args.prefix == "compat-diagnostic-" or canonical:
            result["shapes"][shape].update({"valid_output_hash": output_hash, "valid_output_rows": rounds[0]["valid_output_rows"], "unwritten_slack_rows": rounds[0]["unwritten_slack_rows"], "full_valid_output_bitwise": "all four rounds SHA256 identical"})
        regression |= change > 5
    result["performance_acceptance"] = "not run; GPU8 is precision-only" if args.precision_only else ("requires regression retest" if regression else "passed")
    path = artifact / (args.prefix + ("precision-summary.json" if args.precision_only else ("BAAB-summary.json" if args.offset == 0 else "BAAB-retest-summary.json")))
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if regression:
        raise SystemExit(5)


if __name__ == "__main__":
    main_xe35()
