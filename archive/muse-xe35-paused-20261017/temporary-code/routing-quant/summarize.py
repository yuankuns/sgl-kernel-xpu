import argparse
import json
import pathlib
import re
import statistics

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, choices=(0, 4), default=0)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--precision-only", action="store_true")
    args = parser.parse_args()
    canonical = bool(re.fullmatch(r"canonical-(?:v[1-9][0-9]*(?:-icpx)?-)?(?:precision|performance)-", args.prefix))
    if not canonical and args.prefix not in ("", "diagnostic-"):
        parser.error("Invalid evidence prefix")
    if canonical and args.precision_only != args.prefix.endswith("precision-"):
        parser.error("Canonical prefix and precision/performance mode must match")
    artifact = pathlib.Path(__file__).resolve().parent
    reports = {}
    for relative_round, implementation in ((0, "source"), (1, "port"), (2, "port"), (3, "source")):
        round_id = args.offset + relative_round
        if (artifact / f"{args.prefix}BAAB-{round_id}-{implementation}.exit").read_text().strip() != "0":
            raise RuntimeError(f"BAAB {round_id} did not pass")
        for line in (artifact / f"{args.prefix}BAAB-{round_id}-{implementation}.log").read_text().splitlines():
            if not line.startswith("{"):
                continue
            data = json.loads(line)
            if "shape" in data:
                reports.setdefault(data["shape"], {})[relative_round] = data
    expected_shapes = {
        "route-8-256-8-fp32", "route-32768-256-8-fp32",
        "gather-quant-map-256-4096-bf16", "gather-quant-map-262912-4096-bf16",
        "swiglu-quant-map-256-4096-bf16", "swiglu-quant-map-262912-4096-bf16",
    }
    if set(reports) != expected_shapes:
        raise RuntimeError(f"incomplete production shape coverage: {set(reports)}")
    result = {"scope": "B8/ISL4096/OSL16/TP1/EP1", "acceptance_kind": "native-kernel diagnostic, not official full-package pass" if args.prefix == "diagnostic-" else "official full-package runtime", "shapes": {}}
    regression = False
    for shape, rounds in reports.items():
        if set(rounds) != {0, 1, 2, 3}:
            raise RuntimeError(f"incomplete shape rounds: {shape}")
        if args.precision_only and any(not report.get("precision_only") or report["samples_ms"] for report in rounds.values()):
            raise RuntimeError(f"Precision-only round contains performance work: {shape}")
        if not args.precision_only and any(not report["samples_ms"] for report in rounds.values()):
            raise RuntimeError(f"missing device timing samples: {shape}")
        baseline = rounds[0]["accuracy"]
        for round_id in (1, 2, 3):
            actual = rounds[round_id]["accuracy"]
            if "output" in baseline:
                expected_tensors = torch.load(baseline["output"], weights_only=True)
                actual_tensors = torch.load(actual["output"], weights_only=True)
                torch.testing.assert_close(actual_tensors["ids"], expected_tensors["ids"], rtol=0, atol=0)
                torch.testing.assert_close(actual_tensors["weights"], expected_tensors["weights"], rtol=0, atol=2.5e-4)
            else:
                if actual["q_sha256"] != baseline["q_sha256"] or actual["scale_sha256"] != baseline["scale_sha256"]:
                    raise AssertionError(f"strict production FP4 codes/scales mismatch: {shape}")
        if args.precision_only:
            result["shapes"][shape] = {"source_port_accuracy": "passed original strict precision gates", "performance": "not run"}
            continue
        source_ms = statistics.median(rounds[0]["samples_ms"] + rounds[3]["samples_ms"])
        port_ms = statistics.median(rounds[1]["samples_ms"] + rounds[2]["samples_ms"])
        change = (port_ms / source_ms - 1) * 100
        result["shapes"][shape] = {"source_ms": source_ms, "port_ms": port_ms, "change_pct": change, "source_port_accuracy": "passed original strict precision gates", "needs_regression_retest": change > 5}
        regression |= change > 5
    result["performance_acceptance"] = "not run; GPU8 precision-only" if args.precision_only else ("requires regression retest/diagnosis" if regression else "passed")
    path = artifact / (args.prefix + ("precision-summary.json" if args.precision_only else ("BAAB-summary.json" if args.offset == 0 else "BAAB-retest-summary.json")))
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if regression:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
