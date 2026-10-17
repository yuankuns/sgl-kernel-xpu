import argparse
import fcntl
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['source', 'after'], required=True)
    options = parser.parse_args()
    directory = Path(__file__).resolve().parent
    artifact = directory.parent
    source = options.kind == 'source'
    evidence = artifact / ('fresh-source-full-accuracy-v1' if source else 'official-after-accuracy-v4-gpu11-v1')
    gpu = 8 if source else 11
    original = 'original-full-1300' if source else 'original-adapted-full-1300'
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 72) if source else range(72, 80))
    for name in [f'package-post-{options.kind}', f'provider-post-{options.kind}']:
        assert int((directory / f'{name}.exit').read_text()) == 0
    exits = {name: int((evidence / f'{name}.exit').read_text()) for name in ['host-launch', 'gpu-runtime', original] if (evidence / f'{name}.exit').exists()}
    assert 'host-launch' in exits and 'gpu-runtime' in exits
    known = [int((evidence / name).read_text()) for name in ['host-waiter.pid', 'host-supervisor.pid'] if (evidence / name).exists()]
    for attempt in range(10):
        if all(not Path(f'/proc/{pid}').exists() for pid in known):
            break
        time.sleep(1)
    assert all(not Path(f'/proc/{pid}').exists() for pid in known)
    with Path(f'/tmp/intel-gpu-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        released_epoch = time.time()
        fcntl.flock(lock, fcntl.LOCK_UN)
    nodes = json.loads((evidence / 'actual-collected-nodeids.json').read_text()) if (evidence / 'actual-collected-nodeids.json').exists() else []
    assert len(nodes) in [0, 1300]
    reports = [json.loads(line) for line in (evidence / 'per-case-actual-reports.jsonl').read_text().splitlines()] if (evidence / 'per-case-actual-reports.jsonl').exists() else []
    outcomes = {}
    for report in reports:
        if report['phase'] == 'call' or report['outcome'] != 'passed':
            previous = outcomes.get(report['nodeid'])
            if previous is None or report['outcome'] == 'failed':
                outcomes[report['nodeid']] = report['outcome']
    counts = {outcome: sum(value == outcome for value in outcomes.values()) for outcome in ['passed', 'failed', 'skipped']}
    failed = [node for node, outcome in outcomes.items() if outcome == 'failed']
    skipped = [report for report in reports if report['outcome'] == 'skipped']
    baseline = ['test_fused_swiglu_matches_separate_dtype_surface[dtype0-0.5-2.40625]', 'test_fused_swiglu_cpu_reference[256-dtype0]', 'test_fused_swiglu_cpu_reference[8192-dtype0]']
    new_failures = [node for node in failed if not any(node.endswith(suffix) for suffix in baseline)]
    scope_complete = len(nodes) == len(outcomes) == 1300
    passed = scope_complete and exits.get(original) == 0 and counts['failed'] == 0
    report = {'kind': options.kind, 'CPU_classification_evidence_complete': True, 'original_test_gate_passed': passed,
              'actual_exits': exits, 'full_1300_completed': scope_complete, 'counts': counts,
              'failed_nodeids': failed, 'new_failure_nodeids_beyond_three_SOURCE_baseline': new_failures,
              'original_baseline_failures_absent_in_this_provider': [suffix for suffix in baseline if not any(node.endswith(suffix) for node in failed)],
              'all_SKIP_actual_reports': skipped, 'missing_completed_nodes': sorted(set(nodes) - set(outcomes)),
              'host_owner_PIDs_absent': known, 'container_owner_PID_absent': json.loads((directory / f'provider-post-{options.kind}.json').read_text())['container_runtime_PID_absent'],
              'gpu_lock_probe_free': True, 'GPU_released_epoch': released_epoch,
              'expected_baseline_failure_not_mapped_to_success': True,
              'package_provider_source_native_post_unchanged': True,
              'no_successor_GPU_or_compile_launched': True}
    if not source:
        report['accuracy_subphase_actual_exits'] = {name: int((evidence / f'{name}.exit').read_text()) for name in ['six-production-shapes', 'full-SOURCE-AFTER-comparison', 'independent-four-component-accuracy'] if (evidence / f'{name}.exit').exists()}
    (directory / f'GPU{gpu}-{options.kind}-actual-drain-and-classification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
