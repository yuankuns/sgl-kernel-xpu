import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from build_pool_icpx_v2 import process_snapshot
from observe_maintenance_quiet_v2 import loaded_stack, PROCESS_NAMES


ROOT = Path(__file__).resolve().parent.parent
WRAPPER = '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh'
COMPONENTS = ('memory', 'position', 'logits-vocab')
DRIVERS = ('benchmark_memory_position_model_performance_v2.py', 'benchmark_logits_model_performance_v3.py',
           'benchmark_simple_canonical_verified_perf_v2.py', 'simple_verified_graph_v1.py',
           'simple_representative_performance_cases_v1.py', 'benchmark_simple.py', 'canonical_tools_loader_v1.py')


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save(path, record):
    with path.open('x') as output:
        output.write(json.dumps(record, indent=2) + '\n')


def container_path(path):
    return str(path).replace('/home/syk/', '/root/', 1)


def stage_path(component, label):
    if label == 'after':
        return ROOT / component / 'canonical-v4-icpx/site'
    comparator = json.loads((ROOT / component / 'canonical-validation-v4-icpx/source-comparator-manifest.json').read_text())
    return Path(comparator['stage'].replace('/root/', '/home/syk/', 1))


def maintenance():
    return {pid: record for pid, record in process_snapshot().items() if record['name'] in PROCESS_NAMES}


def container_phase(args):
    run = ROOT / args.run_name
    phase = run / f'round-{args.round}-slot-{args.slot}-{args.label}'
    assert set(os.sched_getaffinity(0)) == set(range(112, 128))
    assert os.environ['ZE_AFFINITY_MASK'] == '7'
    plan = json.loads((run / 'launch-plan.json').read_text())
    for name, expected in plan['drivers_sha256'].items():
        assert digest(ROOT / name) == expected
    for component in COMPONENTS:
        stage = stage_path(component, args.label)
        environment = dict(os.environ, PYTHONPATH=container_path(stage), PYTHONDONTWRITEBYTECODE='1',
                           OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', BATCHES='8', ISL='4096', OSL='16', TP='1', EP='1')
        environment['LD_LIBRARY_PATH'] = '/opt/gfx-deps/profiling_tools/lib:' + environment.get('LD_LIBRARY_PATH', '')
        for family in ('semantic', 'representative'):
            driver = ('benchmark_logits_model_performance_v3.py' if component == 'logits-vocab'
                      else 'benchmark_memory_position_model_performance_v2.py') if family == 'semantic' else 'benchmark_simple_canonical_verified_perf_v2.py'
            positional = [] if component == 'logits-vocab' and family == 'semantic' else [component]
            prefix = phase / f'{component}-{family}'
            command = [sys.executable, '-B', str(ROOT / driver), *positional, '--label', args.label,
                       '--phase', 'performance', '--grant', str(run / f'{component}-{family}-grant.json'),
                       '--output', str(prefix.with_suffix('.json'))]
            prefix.with_suffix('.command').write_text(shlex.join(command) + '\n')
            with prefix.with_suffix('.log').open('x') as output:
                status = subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT).returncode
            prefix.with_suffix('.exit').write_text(str(status) + '\n')
            if status:
                raise SystemExit(status)


def launch(args):
    assert args.run_name == 'simple-model-verified-performance-v2-gpu7-run1'
    assert subprocess.run(['flock', '-n', '/tmp/intel-gpu-7.lock', 'true']).returncode == 0
    assert not maintenance()
    quiet = json.loads((ROOT / 'MAINTENANCE_QUIET_AFTER_FM_v3.json').read_text())
    assert quiet['actual_exit'] == 0 and quiet['continuous_quiet_seconds'] >= 60
    stack = loaded_stack()
    assert stack == quiet['loaded_stack']
    assert (ROOT / 'simple-model-performance-v2-cpu-preparation/full-normal-CPU-gates.exit').read_text().strip() == '0'
    for component in COMPONENTS:
        for label in ('before', 'after'):
            report = json.loads((ROOT / 'simple-model-performance-v2-cpu-preparation' / f'{component}-{label}.json').read_text())
            assert report['actual_exit'] == 0 and report['GPU_initialized'] is False
            assert report['package_manifest_sha256'] == digest(ROOT / component / 'canonical-v4-icpx/source-binary-manifest.json')
    run = ROOT / args.run_name
    run.mkdir(exist_ok=False)
    hashes = {name: digest(ROOT / name) for name in DRIVERS}
    start = time.time() + 120
    provenance = json.loads((ROOT / 'SIMPLE_MODEL_CONFIG_PROVENANCE_v1.json').read_text())
    for component in COMPONENTS:
        official = ROOT / component / 'canonical-v4-icpx'
        comparator_path = ROOT / component / 'canonical-validation-v4-icpx/source-comparator-manifest.json'
        comparator = json.loads(comparator_path.read_text())
        for family in ('semantic', 'representative'):
            driver = ('benchmark_logits_model_performance_v3.py' if component == 'logits-vocab'
                      else 'benchmark_memory_position_model_performance_v2.py') if family == 'semantic' else 'benchmark_simple_canonical_verified_perf_v2.py'
            scope = ('logits-model-performance-v3' if component == 'logits-vocab'
                     else 'memory-position-model-performance-v2') if family == 'semantic' else 'simple-representative-verified-performance-v2'
            grant = {'issuer': 'main-coordinator', 'scope': scope, 'module': component, 'phase': 'performance',
                     'gpu': 7, 'coordinator_release': True, 'accuracy_authorized': False, 'performance_authorized': True,
                     'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1,
                     'start_epoch': start, 'driver_sha256': hashes[driver],
                     'graph_helper_sha256': hashes['simple_verified_graph_v1.py'],
                     'case_helper_sha256': hashes['simple_representative_performance_cases_v1.py'],
                     'package_manifest_sha256': digest(official / 'source-binary-manifest.json'),
                     'source_comparator_manifest_sha256': digest(comparator_path),
                     'source_native_manifest_sha256': comparator['source_native_manifest_sha256'],
                     'model_config_sha256': provenance['model_config_sha256'], 'CPU_affinity': list(range(112, 128))}
            save(run / f'{component}-{family}-grant.json', grant)
    save(run / 'launch-plan.json', {'issuer': 'main-coordinator', 'host_pid': os.getpid(), 'start_epoch': start,
          'drivers_sha256': hashes, 'loaded_stack': stack, 'rounds': 3, 'order': ['before', 'after', 'after', 'before'],
          'semantic_cases': 61, 'representative_cases': 14, 'samples': 101, 'calls': 128,
          'scope': 'Synthetic registered kernel contracts; not whole model, allocator replay, or eight-card proof'})
    time.sleep(max(0, start - time.time()))
    for round_index in range(3):
        for slot, label in enumerate(('before', 'after', 'after', 'before')):
            assert not maintenance() and loaded_stack() == stack
            assert all(digest(ROOT / name) == expected for name, expected in hashes.items())
            phase = run / f'round-{round_index}-slot-{slot}-{label}'
            phase.mkdir(exist_ok=False)
            command = [WRAPPER, '--gpu-lock', '--gpu', '7', '--workdir', container_path(ROOT), '--',
                       'taskset', '-c', '112-127', '/opt/gfx-deps/venv/bin/python', '-B',
                       container_path(Path(__file__)), '--container-phase', '--run-name', args.run_name,
                       '--round', str(round_index), '--slot', str(slot), '--label', label]
            phase.joinpath('wrapper.command').write_text(shlex.join(command) + '\n')
            contamination = []
            with (phase / 'wrapper.log').open('x') as output:
                worker = subprocess.Popen(command, env=dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015'), stdout=output, stderr=subprocess.STDOUT)
                save(phase / 'wrapper-owned.json', {'PID': worker.pid, 'epoch': time.time(), 'GPU': 7})
                while worker.poll() is None:
                    active = maintenance()
                    current = loaded_stack()
                    if active or current != stack:
                        contamination.append({'epoch': time.time(), 'active': active, 'loaded_stack': current})
                    time.sleep(2)
                status = worker.returncode
            phase.joinpath('wrapper.exit').write_text(str(status) + '\n')
            save(phase / 'maintenance-watch.json', {'contamination': contamination, 'external_processes_modified': False})
            if status or contamination:
                raise SystemExit(status or 2)
    save(run / 'complete.json', {'actual_exit': 0, 'phases': 12, 'precision_performance_raw_completed': True,
                                 'independent_comparison_analysis_pending': True, 'acceptance_complete': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch-main-authorized', action='store_true')
    parser.add_argument('--container-phase', action='store_true')
    parser.add_argument('--run-name', default='simple-model-verified-performance-v2-gpu7-run1')
    parser.add_argument('--round', type=int, choices=range(3))
    parser.add_argument('--slot', type=int, choices=range(4))
    parser.add_argument('--label', choices=('before', 'after'))
    args = parser.parse_args()
    if args.container_phase:
        container_phase(args)
    elif args.launch_main_authorized:
        launch(args)
    else:
        parser.error('Explicit main handoff and launch authorization required; preparation never starts GPU work')


if __name__ == '__main__':
    main()
