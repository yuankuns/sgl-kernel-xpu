import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from canonical_check import ARTIFACT, READY, TARGET, inventory, local_path, verify_main_ready


def main():
    for name in ('BATCHES', 'BATCH_SIZE', 'BATCH', 'SERVING_BATCH', 'SERVING_BATCH_SIZE', 'SGLANG_BATCH_SIZE'):
        assert os.environ.get(name, '8') == '8', f'B8 only: {name}'
    for name, value in {'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}.items():
        assert os.environ.get(name, value) == value
    authorization = json.loads((ARTIFACT / 'preopt-authorization.json').read_text())
    assert authorization['authorized'] and authorization['gpu'] == 6
    assert os.environ['ZE_AFFINITY_MASK'] == '6'
    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    official = Path(os.environ['QK_OFFICIAL_ROOT'])
    formal = json.loads((official / 'formal-checks.json').read_text())
    assert formal['accuracy_verified'] and formal['all_runtime_mapped_dso_verified']
    provenance = json.loads((official / 'provenance.json').read_text())
    frozen, ready_path = verify_main_ready(provenance['canonical_version'], local_path(provenance['ready_manifest']))
    assert authorization['source_digest'] == provenance['original_qk_source_digest'] == json.loads(READY.read_text())['source_digest']
    assert authorization['canonical_version'] == provenance['canonical_version'] == 'canonical-v4-icpx'
    assert provenance['ready_source_sha256'] == frozen['source_digest']
    assert provenance['compiler_policy_proof']['verified'] and provenance['source_readonly_postverified']
    assert all(inventory.file_record(TARGET / name) == record for name, record in frozen['files'].items())
    evidence = ARTIFACT.parents[2] / 'sgl-kernel-xpu/review-validation-artifacts/qk-before-after-20261017'
    baseline = json.loads((evidence / 'provenance.json').read_text())['before']
    provenance.update({'preopt_commit': baseline['commit'], 'preopt_site': str(evidence / 'before/site'), 'preopt_manifest': str(evidence / 'before/site-manifest.json'), 'preopt_manifest_sha256': baseline['site_manifest_sha256'], 'preopt_qk_library_sha256': baseline['qk_library_sha256'], 'comparison': 'original preopt ab3a26c -> canonical PORT; independent from optimized SOURCE -> PORT'})
    from fresh_native_check import verify_provider

    provenance['historical_preopt_qk_library_sha256'] = provenance['preopt_qk_library_sha256']
    provenance.update(verify_provider('preopt'))
    root = official / 'preopt-to-port'
    assert not root.exists(), 'Do not overwrite independent preopt results'
    inventory.write_json(root / 'provenance.json', provenance)
    commands = []
    accuracy = {}
    timings = {}

    def run(label, name, arguments, numerical_failure_allowed=False):
        directory = root / label
        directory.mkdir(exist_ok=True)
        log = directory / f'{name}.log'
        audit_path = directory / f'{name}.runtime-dso.json'
        environment = {**os.environ, 'PYTHONPATH': str(local_path(provenance[f'{label}_site'])), 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'}
        command = [sys.executable, str(ARTIFACT / 'runtime_check.py'), '--provenance', str(root / 'provenance.json'), '--label', label, '--output', str(audit_path), '--', *arguments]
        started = time.monotonic()
        with log.open('x') as stream:
            result = subprocess.run(command, cwd=TARGET, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        log.with_suffix('.exit').write_text(f'{result.returncode}\n')
        record = {'label': label, 'name': name, 'command': command, 'exit': result.returncode, 'seconds': time.monotonic() - started, 'log': str(log), 'log_sha256': inventory.digest_file(log)}
        commands.append(record)
        inventory.write_json(root / 'commands.json', commands)
        print(log.read_text()[-5000:], flush=True)
        assert result.returncode == 0 or numerical_failure_allowed and result.returncode == 1
        audit = json.loads(audit_path.read_text())
        assert audit['normal_package_import_verified'] and audit['all_runtime_mapped_dso_verified']
        record['runtime_dso_audit'] = str(audit_path)
        record['runtime_dso_audit_sha256'] = inventory.digest_file(audit_path)
        inventory.write_json(root / 'commands.json', commands)
        time.sleep(2)
        return log, result.returncode

    for tokens in (8, 32768):
        for label in ('preopt', 'port'):
            log, status = run(label, f'accuracy-t{tokens}', [str(TARGET / 'benchmark/validate_qk_norm_rope_b8.py'), '--tokens', str(tokens)], numerical_failure_allowed=label == 'preopt')
            result = next(json.loads(line) for line in log.read_text().splitlines() if line.startswith('{') and 'violations' in line)
            assert result['atol'] == result['rtol'] == 5e-3 and result['tail_equal']
            assert result['dtype'] == 'bf16' and result['heads'] == [128, 16, 144] and result['head_dim'] == 64
            assert result['tokens'] == tokens and result['elements'] == tokens * 144 * 64
            assert status == (1 if result['violations'] else 0)
            if label == 'port':
                assert result['violations'] == 0 and status == 0
            accuracy[f'{label}-t{tokens}'] = {'actual_exit': status, **result}
            inventory.write_json(root / 'accuracy.json', accuracy)
        assert accuracy[f'preopt-t{tokens}']['input_sha256'] == accuracy[f'port-t{tokens}']['input_sha256']

    def graph_round(round_id, shapes):
        for tokens in shapes:
            for slot, label in enumerate(('preopt', 'port', 'port', 'preopt'), 1):
                name = f'graph-r{round_id}-t{tokens}-s{slot}'
                output = root / label / f'{name}.json'
                run(label, name, [str(TARGET / 'benchmark/bench_muse_qk_norm_rope.py'), '--tokens', str(tokens), '--output', str(output)])
                result = json.loads(output.read_text())
                assert result['repeats'] == 32 and len(result['raw_us']) == 21
                timings.setdefault(f'{label}-t{tokens}', []).append(result)

    for round_id in (1, 2):
        graph_round(round_id, (8, 32768))
    summary = {}
    for tokens in (8, 32768):
        before = statistics.median(result['p50_us'] for result in timings[f'preopt-t{tokens}'])
        after = statistics.median(result['p50_us'] for result in timings[f'port-t{tokens}'])
        eligible = accuracy[f'preopt-t{tokens}']['violations'] == 0
        summary[str(tokens)] = {'preopt_us': before, 'port_us': after, 'preopt_violations': accuracy[f'preopt-t{tokens}']['violations'], 'preopt_actual_exit': accuracy[f'preopt-t{tokens}']['actual_exit'], 'port_violations': 0, 'correctness_matched': eligible, 'timing_role': 'correctness-matched comparison' if eligible else 'diagnostic timing only; failed precision baseline; no accepted speedup'}
        if eligible:
            summary[str(tokens)]['change_pct'] = (after / before - 1) * 100
            summary[str(tokens)]['speedup'] = before / after
        if after / before > 1.05:
            for round_id in (3, 4):
                graph_round(round_id, (tokens,))
            summary[str(tokens)]['retest_process_p50_us'] = {label: [result['p50_us'] for result in timings[f'{label}-t{tokens}'][4:]] for label in ('preopt', 'port')}
    inventory.write_json(root / 'summary.json', summary)
    inventory.write_json(root / 'completion.json', {'port_strict_zero_violations': True, 'preopt_actual_failure_exit_preserved': True, 'failed_baseline_speedup_claimed': False, 'optimized_source_to_port_independent_results': str(official), 'gpu': 6, 'all_tasks_require_drain_before_main_w4_grant': True})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
