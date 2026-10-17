import argparse
import hashlib
import itertools
import json
import pathlib

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', choices=('diagnostic-', 'fresh-diagnostic-', 'compat-diagnostic-'), default='diagnostic-')
    args = parser.parse_args()
    torch.set_num_threads(16)
    artifact = pathlib.Path(__file__).resolve().parent
    reports = {}
    for round_id, implementation in ((0, "source"), (1, "port"), (2, "port"), (3, "source")):
        path = artifact / f"{args.prefix}BAAB-{round_id}-{implementation}.log"
        for line in path.read_text().splitlines():
            if line.startswith("{"):
                report = json.loads(line)
                if "shape" in report:
                    reports.setdefault(report["shape"], {})[round_id] = report
    result = {"scope": "CPU-only forensic audit; no GPU initialization", "original_tolerance": {"rtol": 0.005, "atol": 0.005}, "shapes": {}}
    if args.prefix == 'compat-diagnostic-':
        expected_shapes = {'256-8192-4096', '256-4096-4096', '257-4096-4096', '262144-4096-4096', '262144-8192-4096'}
        if set(reports) != expected_shapes or any(set(rounds) != {0, 1, 2, 3} for rounds in reports.values()):
            raise RuntimeError('Post-adaptation forensic requires all five shapes and four rounds')
    for shape, rounds in reports.items():
        tensors = {}
        entry = {"artifacts": {}, "comparisons": {}, "bitwise_scope": "all saved original independent-reference sample rows; full valid-output hashes required in post-adaptation run"}
        if args.prefix in ('fresh-diagnostic-', 'compat-diagnostic-'):
            inputs = {round_id: report['input_hashes'] for round_id, report in rounds.items()}
            entry['input_fingerprints_all_four_equal'] = len(inputs) == 4 and all(record == inputs[0] for record in inputs.values())
            entry['input_fingerprints'] = inputs
            entry['fp64_difference_points'] = {round_id: report['fp64_difference_points'] for round_id, report in rounds.items()}
        if args.prefix == 'compat-diagnostic-':
            hashes = {round_id: report['valid_output_hash'] for round_id, report in rounds.items()}
            entry['full_valid_output_hashes'] = hashes
            entry['full_valid_output_all_four_bitwise_equal'] = all(record == hashes[0] for record in hashes.values())
            entry['valid_output_rows'] = rounds[0]['valid_output_rows']
            entry['unwritten_slack_rows'] = rounds[0]['unwritten_slack_rows']
            points = [point for report in rounds.values() for point in report['fp64_difference_points'].values()]
            entry['recorded_midpoint_count'] = len(points)
            entry['all_recorded_points_exact_midpoint_and_source_semantics'] = len(points) == 32 and all(point['expected_minus_midpoint'] == 0 and point['current_output'] == point['old_source'] for point in points)
        for round_id, report in rounds.items():
            path = pathlib.Path(report["output_sample"])
            tensors[round_id] = torch.load(path, map_location="cpu", weights_only=True)
            entry["artifacts"][round_id] = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "reference": report["accuracy"]}
        for first, second in itertools.combinations(sorted(tensors), 2):
            baseline = tensors[first]
            actual = tensors[second]
            baseline_double = baseline.double()
            actual_double = actual.double()
            difference = (actual_double - baseline_double).abs()
            mismatches = difference > (0.005 + 0.005 * baseline_double.abs())
            different = actual != baseline
            baseline_bits = baseline.view(torch.int16).int()
            actual_bits = actual.view(torch.int16).int()
            ulp_distance = (actual_bits - baseline_bits).abs()
            examples = torch.nonzero(different)[:8].tolist()
            original_failure = None
            try:
                torch.testing.assert_close(actual, baseline, rtol=0.005, atol=0.005)
            except AssertionError as error:
                original_failure = str(error)
            entry["comparisons"][f"{first}-{second}"] = {
                "bitwise_equal": torch.equal(actual, baseline),
                "total": baseline.numel(),
                "different": int(torch.count_nonzero(actual != baseline)),
                "original_dtype_assertion": "failed" if original_failure else "passed",
                "original_failure": original_failure,
                "double_evaluated_same_tolerance_mismatches": int(mismatches.sum()),
                "max_abs": float(difference.max()),
                "max_bf16_encoding_distance": int(ulp_distance.max()),
                "baseline_even_at_different": int(((baseline_bits & 1 == 0) & different).sum()),
                "actual_even_at_different": int(((actual_bits & 1 == 0) & different).sum()),
                "examples": [{"index": index, "baseline": float(baseline[tuple(index)]), "actual": float(actual[tuple(index)])} for index in examples],
            }
        result["shapes"][shape] = entry
    result['precision_gate'] = 'failed' if any(comparison['original_dtype_assertion'] == 'failed' for shape in result['shapes'].values() for comparison in shape['comparisons'].values()) else 'passed sampled original gate'
    if args.prefix == 'compat-diagnostic-':
        complete = all(shape['input_fingerprints_all_four_equal'] and shape['full_valid_output_all_four_bitwise_equal'] and shape['all_recorded_points_exact_midpoint_and_source_semantics'] for shape in result['shapes'].values())
        result['precision_gate'] = 'passed original 0.005 gate, all valid outputs bitwise, and 160 exact midpoint source-semantic points' if complete and result['precision_gate'] != 'failed' else 'failed'
    output = artifact / (args.prefix + "equivalence-forensics.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
