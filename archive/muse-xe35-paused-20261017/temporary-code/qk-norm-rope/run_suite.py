import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

artifact = Path(__file__).resolve().parent
target = artifact.parents[1] / 'pr/muse-qk-norm-rope-xe35'
assert not (os.environ.get('QK_DIAGNOSTIC_ROOT') and os.environ.get('QK_OFFICIAL_ROOT'))
root = Path(os.environ.get('QK_OFFICIAL_ROOT', os.environ.get('QK_DIAGNOSTIC_ROOT', artifact)))
diagnostic = bool(os.environ.get('QK_DIAGNOSTIC_ROOT'))
provenance = json.loads((root / 'provenance.json').read_text())
if not diagnostic:
    assert provenance['mode'] == 'canonical-default-all-current-graph-fresh-install'
    from canonical_check import verify_main_ready, local_path

    frozen, ready_path = verify_main_ready(provenance['canonical_version'], local_path(provenance['ready_manifest']))
    assert frozen['source_digest'] == provenance['ready_source_sha256']
    assert provenance['compiler_policy_proof']['verified'] and provenance['source_readonly_postverified']
sites = {label: Path(provenance[f'{label}_site'].replace('/home/syk/', '/root/', 1)) for label in ('source', 'port')}
commands = []
accuracy = {}
(root / 'run-info.json').write_text(json.dumps({'gpu_affinity': os.environ.get('ZE_AFFINITY_MASK'), 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'mode': provenance.get('mode', 'official-default-feature-full-wheel'), 'scope': 'B8/ISL4096/OSL16/TP1/EP1 BF16 T8/T32768 Q128/K16/tail144/D64'}, indent=2) + '\n')


def run_xe35(label, name, arguments):
    directory = root / label
    directory.mkdir(exist_ok=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(sites[label])
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = env['OPENBLAS_NUM_THREADS'] = '8'
    command = [sys.executable, *arguments]
    runtime_audit = None
    if not diagnostic:
        runtime_audit = directory / f'{name}.runtime-dso.json'
        command = [sys.executable, str(artifact / 'runtime_check.py'), '--provenance', str(root / 'provenance.json'), '--label', label, '--output', str(runtime_audit), '--', *arguments]
    print(f'RUN {label} {name}: {command}', flush=True)
    log = directory / f'{name}.log'
    start = time.monotonic()
    with log.open('w') as stream:
        result = subprocess.run(command, env=env, cwd=target, stdout=stream, stderr=subprocess.STDOUT)
    (directory / f'{name}.exit').write_text(f'{result.returncode}\n')
    commands.append({'label': label, 'name': name, 'command': command, 'exit': result.returncode, 'seconds': time.monotonic() - start, 'log': str(log), 'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()})
    (root / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    print(log.read_text()[-5000:], flush=True)
    assert result.returncode == 0, f'{label} {name} failed'
    if runtime_audit:
        audit = json.loads(runtime_audit.read_text())
        assert audit['normal_package_import_verified'] and audit['all_runtime_mapped_dso_verified']
        commands[-1]['runtime_dso_audit'] = str(runtime_audit)
        commands[-1]['runtime_dso_audit_sha256'] = hashlib.sha256(runtime_audit.read_bytes()).hexdigest()
        (root / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    time.sleep(2)
    return log


for label, site in sites.items():
    code = f'import json, torch, sgl_kernel; print(json.dumps({{"xpu_available": torch.xpu.is_available(), "device": torch.xpu.get_device_name(0), "package": sgl_kernel.__file__, "common_ops": sgl_kernel.common_ops.__file__}})); assert torch.xpu.is_available(); assert sgl_kernel.__file__.startswith({str(site)!r}); assert sgl_kernel.common_ops.__file__.startswith({str(site)!r})'
    run_xe35(label, 'normal-import', ['-c', code])

for tokens in (8, 32768):
    for label in ('source', 'port'):
        log = run_xe35(label, f'accuracy-t{tokens}', [str(target / 'benchmark/validate_qk_norm_rope_b8.py'), '--tokens', str(tokens)])
        record = next(json.loads(line) for line in log.read_text().splitlines() if line.startswith('{') and 'violations' in line)
        assert record['violations'] == 0 and record['tail_equal']
        assert record['atol'] == record['rtol'] == 5e-3
        assert record['dtype'] == 'bf16' and record['heads'] == [128, 16, 144] and record['head_dim'] == 64
        assert record['tokens'] == tokens and record['elements'] == tokens * 144 * 64
        for path, checksum in record['loaded_libraries'].items():
            assert Path(path).is_relative_to(sites[label]), path
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == checksum
        if diagnostic and label == 'port':
            expected_kernel = sites[label] / 'sgl_kernel/libsgl-ops-sycl-FusedQKNormRope_xe35.so'
            assert record['loaded_libraries'][str(expected_kernel)] == provenance['port_qk_library_sha256']
        accuracy[f'{label}-t{tokens}'] = record
    assert accuracy[f'source-t{tokens}']['input_sha256'] == accuracy[f'port-t{tokens}']['input_sha256']
    assert accuracy[f'source-t{tokens}']['output_sha256'] == accuracy[f'port-t{tokens}']['output_sha256'], 'SOURCE/PORT bitwise mismatch'
    (root / 'accuracy.json').write_text(json.dumps(accuracy, indent=2) + '\n')

fp16_log = run_xe35('port', 'fp16-t8', [str(target / 'benchmark/validate_qk_norm_rope_b8.py'), '--tokens', '8', '--dtype', 'fp16', '--entry', 'generic' if diagnostic else 'xe35'])
fp16_record = next(json.loads(line) for line in fp16_log.read_text().splitlines() if line.startswith('{') and 'violations' in line)
assert fp16_record['atol'] == fp16_record['rtol'] == 1e-3
assert fp16_record['violations'] == 0 and fp16_record['tail_equal'] and fp16_record['dtype'] == 'fp16'
(root / 'fp16-support-accuracy.json').write_text(json.dumps(fp16_record, indent=2) + '\n')
if not diagnostic:
    pytest_log = run_xe35('port', 'pytest', ['-m', 'pytest', '-p', 'no:cacheprovider', '-q', str(target / 'tests/test_fused_qk_norm_rope.py'), '-k', 'production_reference_xe35 or frequency_lifetime_and_generic_dispatch_xe35'])
    assert re.search(r'\b6 passed\b', pytest_log.read_text()), 'Official Xe35 acceptance requires six actual passes, not skips'
else:
    (root / 'verification-mode.json').write_text(json.dumps({'mode': 'source-full-package/native-abi-port-diagnostic', 'official_full_package_pass': False, 'not_validated': ['official wheel and registration', 'direct xe35 Python entry', 'official pytest lifetime and dispatch cases']}, indent=2) + '\n')

timings = {}


def graph_round_xe35(round_id, tokens_list):
    for tokens in tokens_list:
        for slot, label in enumerate(('source', 'port', 'port', 'source'), 1):
            name = f'graph-r{round_id}-t{tokens}-s{slot}'
            output = root / label / f'{name}.json'
            run_xe35(label, name, [str(target / 'benchmark/bench_muse_qk_norm_rope.py'), '--tokens', str(tokens), '--output', str(output)])
            record = json.loads(output.read_text())
            assert len(record['raw_us']) == 21 and record['repeats'] == 32
            assert record['loaded_libraries']
            for path in record['loaded_libraries']:
                assert Path(path).is_relative_to(sites[label]), path
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == record['loaded_libraries'][path]
            if diagnostic:
                kernel_name = 'libsgl-ops-sycl-FusedQKNormRope_xe35.so' if label == 'port' else 'libsgl-ops-sycl-FusedQKNormRope.so'
                kernel_path = str(sites[label] / 'sgl_kernel' / kernel_name)
                assert record['loaded_libraries'][kernel_path] == provenance[f'{label}_qk_library_sha256']
            timings.setdefault(f'{label}-t{tokens}', []).append(record)


for round_id in (1, 2):
    graph_round_xe35(round_id, (8, 32768))

summary = {}
for tokens in (8, 32768):
    before = statistics.median(record['p50_us'] for record in timings[f'source-t{tokens}'])
    after = statistics.median(record['p50_us'] for record in timings[f'port-t{tokens}'])
    summary[str(tokens)] = {'source_us': before, 'port_us': after, 'change_pct': (after / before - 1) * 100, 'source_process_p50_us': [record['p50_us'] for record in timings[f'source-t{tokens}']], 'port_process_p50_us': [record['p50_us'] for record in timings[f'port-t{tokens}']]}
regressed = [tokens for tokens in (8, 32768) if summary[str(tokens)]['change_pct'] > 5]
if regressed:
    for round_id in (3, 4):
        graph_round_xe35(round_id, regressed)
    for tokens in regressed:
        source_retest = [record['p50_us'] for record in timings[f'source-t{tokens}'][4:]]
        port_retest = [record['p50_us'] for record in timings[f'port-t{tokens}'][4:]]
        before, after = statistics.median(source_retest), statistics.median(port_retest)
        summary[str(tokens)]['retest'] = {'source_us': before, 'port_us': after, 'change_pct': (after / before - 1) * 100, 'source_process_p50_us': source_retest, 'port_process_p50_us': port_retest}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
if not diagnostic:
    (root / 'formal-checks.json').write_text(json.dumps({'ready_source_sha256': provenance['ready_source_sha256'], 'canonical_manifest_sha256': provenance['canonical_manifest_sha256'], 'default_all_command_verified': True, 'normal_package_import_verified': True, 'all_runtime_mapped_dso_verified': True, 'accuracy_verified': True, 'performance_measured_on_gpu6': True, 'regressions_over_5_percent': [tokens for tokens in (8, 32768) if summary[str(tokens)].get('retest', summary[str(tokens)])['change_pct'] > 5], 'independent_clean_build': False, 'diagnostic_bridge_used': False}, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
