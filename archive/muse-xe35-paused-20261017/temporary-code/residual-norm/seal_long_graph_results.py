import hashlib
import json
from pathlib import Path
import subprocess
import time

from stop_full_build import processes


root = Path(__file__).resolve().parent
for name in ['long_graph_gpu7', 'long_graph_supervisor', 'long_graph_analysis']:
    assert (root / f'{name}.exit').read_text().strip() == '0'
logs = sorted(root.glob('long_graph_*_perf_*.log'))
assert len(logs) == 48
total_cases = 0
total_samples = 0
reference_cases = 0
libraries = set()
for path in logs:
    assert path.with_suffix('.exit').read_text().strip() == '0'
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{"provider')]
    assert len(cases) == 4
    for case in cases:
        assert len(case['samples_us']) == len(case['cpu_enqueue_samples']) == 31
        assert case['iterations'] in (128, 256)
        assert case['rows'] == 8 and case['hidden_size'] == 8192
        assert case['scope'] == 'B8/ISL4096/OSL16/TP1/EP1'
        assert case['cpu_environment']['torch_num_threads'] == 8
        for sample, trace in zip(case['samples_us'], case['cpu_enqueue_samples']):
            assert abs(sample * case['iterations'] - trace['device_event_interval_us']) < 1e-7
        total_cases += 1
        total_samples += len(case['samples_us'])
        reference_cases += case['cpu_environment']['reference_before_timing']
    libraries.update(line for line in path.read_text().splitlines() if line.startswith('loaded_lib='))
assert total_cases == 192 and total_samples == 5952 and reference_cases == 128
subprocess.run(['sha256sum', '-c', str(root / 'long_graph_plan.sha256')], check=True)
subprocess.run(['sha256sum', '-c', str(root / 'diagnostic_evidence.sha256')], check=True)
assert subprocess.run(['flock', '-n', '/tmp/intel-gpu-7.lock', 'true']).returncode == 0
active = {pid: entry for pid, entry in processes().items() if 'Z' not in entry['state'] and
          (entry['args'].startswith(f'bash {root}/run_long_graph_gpu7_xe35.sh') or
           entry['args'].startswith(f'python3 {root}/wait_gpu_barrier.py --phase long-graph') or
           (entry['args'].startswith('/root/') and '--decode-focus --cpu-submit-trace' in entry['args']))}
assert not active, active
report = json.loads((root / 'long_graph_summary.json').read_text())
record = {'time_epoch': time.time(), 'handle': 96541, 'gpu': 7,
          'terminal_exit': 0, 'supervisor_exit': 0, 'analysis_exit': 0,
          'gpu7_lock_probe_exit': 0, 'active_owned_runtime': active,
          'next_owner_requires_main_grant': True, 'official_acceptance': False,
          'canonical_full_package_pending': True, 'kernel_changed': False,
          'providers': len(logs), 'timing_cases': total_cases, 'raw_samples_with_cpu_spans': total_samples,
          'full_decode_reference_cases': reference_cases,
          'aggregate_delta_range_pct': [min(case['delta_pct'] for case in report['aggregate']),
                                        max(case['delta_pct'] for case in report['aggregate'])],
          'per_round_delta_range_pct': [min(case['delta_pct'] for case in report['per_BAAB_round']),
                                        max(case['delta_pct'] for case in report['per_BAAB_round'])],
          'loaded_library_proofs': sorted(libraries),
          'attribution': 'long graph did not reproduce >5% regression; event intervals may include CPU submission gaps; no exclusive-kernel, DVFS, or KMP efficacy claim'}
(root / 'long_graph_terminal.json').write_text(json.dumps(record, indent=2) + '\n')
paths = logs + [path.with_suffix('.exit') for path in logs]
paths += [root / name for name in ['long_graph_gpu7.log', 'long_graph_gpu7.exit',
          'long_graph_supervisor.log', 'long_graph_supervisor.exit', 'long_graph_analysis.log',
          'long_graph_analysis.exit', 'long_graph_summary.json', 'LONG_GRAPH_RESULTS.md',
          'long_graph_terminal.json', 'gpu7-long-graph-barrier.json', 'long_graph_plan.sha256']]
(root / 'long_graph_evidence.sha256').write_text(''.join(
    f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n' for path in sorted(paths)))
print(json.dumps({key: value for key, value in record.items() if key != 'loaded_library_proofs'}, indent=2))
