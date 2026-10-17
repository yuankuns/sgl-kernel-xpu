import json
import math
from pathlib import Path
import statistics


def read_cases(path):
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{"provider')]
    assert len(cases) == 4, (path, len(cases))
    return {(case["post_mode"], case["round_pre_norm_input"], case["output_fp32"]): case for case in cases}


def correlation(first, second):
    first_mean, second_mean = statistics.mean(first), statistics.mean(second)
    centered_first = [value - first_mean for value in first]
    centered_second = [value - second_mean for value in second]
    denominator = math.sqrt(sum(value * value for value in centered_first) *
                            sum(value * value for value in centered_second))
    return None if denominator == 0 else sum(first_value * second_value for first_value, second_value in
                                           zip(centered_first, centered_second)) / denominator


def trace_summary(cases):
    trace = [sample for case in cases for sample in case["cpu_enqueue_samples"]]
    event = [sample["device_event_interval_us"] for sample in trace]
    enqueue = [sample["cpu_enqueue_span_us"] for sample in trace]
    fields = ("start_record_us", "start_to_replay_cpu_gap_us", "graph_submit_us",
              "replay_to_end_cpu_gap_us", "end_record_us", "cpu_enqueue_span_us")
    return {"samples": len(trace), "p50_us": {name: statistics.median([sample[name] for sample in trace]) for name in fields},
            "event_interval_vs_cpu_enqueue_correlation": correlation(event, enqueue),
            "cpu_environment": cases[0]["cpu_environment"]}


def main():
    root = Path(__file__).resolve().parent
    aggregate = []
    individual = []
    table = ["# GPU7 long-graph diagnostic: event interval, not exclusive kernel time", "",
             "Source/port normal original sgl_kernel ABI; pure native forwarding diagnostic only. CPU spans overlap GPU work: do not subtract them as exclusive kernel time.", "",
             "| Environment | Calls | Post | Round | Widen | Source event µs/call | Port event µs/call | Change | Source CPU enqueue µs/graph | Port CPU enqueue µs/graph |",
             "|---|---:|---|---|---|---:|---:|---:|---:|---:|"]
    for environment in ("baseline", "reference_default", "reference_kmp0"):
        for calls in (128, 256):
            runs = {}
            for round_number in (1, 2):
                for name in ("B1", "A1", "A2", "B2"):
                    stem = f"long_graph_{environment}_{calls}_round{round_number}_perf_{name}"
                    assert (root / f"{stem}.exit").read_text().strip() == "0"
                    runs[(round_number, name)] = read_cases(root / f"{stem}.log")
            for key in runs[(1, "B1")]:
                source_cases = [cases[key] for (round_number, name), cases in runs.items() if name.startswith("A")]
                port_cases = [cases[key] for (round_number, name), cases in runs.items() if name.startswith("B")]
                for case in source_cases + port_cases:
                    assert case["rows"] == 8 and case["hidden_size"] == 8192
                    assert case["scope"] == "B8/ISL4096/OSL16/TP1/EP1"
                    assert case["iterations"] == calls
                    assert len(case["samples_us"]) == len(case["cpu_enqueue_samples"]) == 31
                    assert case["cpu_environment"]["reference_before_timing"] == (environment != "baseline")
                    assert case["cpu_environment"]["KMP_BLOCKTIME"] == ("0" if environment == "reference_kmp0" else None)
                source = statistics.median([sample for case in source_cases for sample in case["samples_us"]])
                port = statistics.median([sample for case in port_cases for sample in case["samples_us"]])
                source_trace, port_trace = trace_summary(source_cases), trace_summary(port_cases)
                delta = (port / source - 1) * 100
                aggregate.append({"environment": environment, "calls": calls, "case": key,
                                  "source_event_interval_us_per_call": source,
                                  "port_event_interval_us_per_call": port,
                                  "delta_pct": delta, "above_5pct": delta > 5,
                                  "source_cpu": source_trace, "port_cpu": port_trace,
                                  "exclusive_kernel_time": False})
                table.append(f"| {environment} | {calls} | {key[0]} | {key[1]} | {key[2]} | {source:.4f} | {port:.4f} | {delta:+.2f}% | {source_trace['p50_us']['cpu_enqueue_span_us']:.3f} | {port_trace['p50_us']['cpu_enqueue_span_us']:.3f} |")
                for round_number in (1, 2):
                    medians = {name: runs[(round_number, name)][key]["p50_us"] for name in ("B1", "A1", "A2", "B2")}
                    source_round = statistics.median(runs[(round_number, "A1")][key]["samples_us"] + runs[(round_number, "A2")][key]["samples_us"])
                    port_round = statistics.median(runs[(round_number, "B1")][key]["samples_us"] + runs[(round_number, "B2")][key]["samples_us"])
                    individual.append({"environment": environment, "calls": calls, "round": round_number, "case": key,
                                       "delta_pct": (port_round / source_round - 1) * 100, "p50_us": medians,
                                       "source_AA_drift_pct": (medians["A2"] / medians["A1"] - 1) * 100,
                                       "port_BB_drift_pct": (medians["B2"] / medians["B1"] - 1) * 100})
    record = {"diagnostic_only": True, "gpu": 7, "cases_per_provider": 4,
              "full_24_case_accuracy_evidence": "diagnostic_accuracy_summary.json (unchanged)",
              "timing": "device event graph interval includes potential host submission gaps; no exclusive-kernel inference or CPU-span subtraction",
              "aggregate": aggregate, "per_BAAB_round": individual}
    (root / "long_graph_summary.json").write_text(json.dumps(record, indent=2) + "\n")
    (root / "LONG_GRAPH_RESULTS.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print("aggregate_above_5pct", [(case["environment"], case["calls"], case["case"]) for case in aggregate if case["above_5pct"]])


if __name__ == "__main__":
    main()
