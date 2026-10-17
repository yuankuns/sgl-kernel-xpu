import argparse
import fcntl
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from providers import ARTIFACT, DIRECTORY, FRESH, PYTHONS, SITES, TORCH_ROOTS, digest, preflight

sys.path.insert(0, str(ARTIFACT / 'fresh-source-icpx-v2'))
import build_source
from process_tree import ProcessTree


SOURCE = Path('/root/worktrees/sgl-kernel-xpu/muse-spark-nvfp4-woq-main-20261014') if str(ARTIFACT).startswith('/root/') else Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-nvfp4-woq-main-20261014')
REVISION = 'd983e877d6bf97015027bd3b1379ff047009a5dd'
RECIPES = ('providers.py', 'accuracy_plugin.py', 'run_source_case.py', 'supervisor.py', 'historical-package-manifest.json')


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def interrupted(signum, frame):
    raise RuntimeError('Interrupted: ' + str(signum))


def run(command, name, environment, timeout=3600):
    stem = DIRECTORY / name
    stem.with_suffix('.command').write_text(shlex.join(command) + '\n')
    token = uuid.uuid4().hex
    environment = dict(environment, MUSE_SOURCE_BUILD_TOKEN=token)
    start = time.time()
    process = None
    tracker = None
    reason = None
    with stem.with_suffix('.log').open('x') as output, stem.with_suffix('.resources.jsonl').open('x') as resources:
        process = subprocess.Popen(command, env=environment, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        tracker = ProcessTree(process.pid, token)
        save(stem.with_suffix('.started.json'), {'PID': process.pid, 'startticks': build_source.sidecar.process_ticks(process.pid), 'token': token, 'epoch': start, 'command': command})
        try:
            while True:
                metrics = tracker.sample()
                resources.write(json.dumps(metrics) + '\n')
                resources.flush()
                save(stem.with_suffix('.owned.json'), {'token': token, 'identities': metrics['identities']})
                if time.time() - start > timeout:
                    reason = 'Accuracy command timeout'
                    tracker.drain()
                    break
                if process.poll() is not None:
                    if tracker.sample()['live']:
                        reason = 'Root exited with descendants still live'
                        tracker.drain()
                    break
                time.sleep(0.25)
        except BaseException:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            drain = tracker.drain()
            save(stem.with_suffix('.drain.json'), drain)
            process.wait(timeout=15)
            stem.with_suffix('.exit').write_text(str(process.returncode) + '\n')
            raise
    status = process.wait(timeout=15)
    final = tracker.sample()
    save(stem.with_suffix('.drain.json'), {'drained': not final['live'], 'live_after': final['live'], 'identities': final['identities'], 'PGIDs_signalled': sorted(tracker.groups_signalled)})
    stem.with_suffix('.exit').write_text(str(status) + '\n')
    save(stem.with_suffix('.terminal.json'), {'actual_exit': status, 'elapsed_seconds': time.time() - start, 'stop_reason': reason, 'all_descendants_drained': not final['live'], 'RSS_sampled': True, 'compiler_budget_applied': False, 'GPU_precision_only': name.startswith(('tests-', 'tails-', 'production-'))})
    if status or reason or final['live']:
        raise RuntimeError('Actual failure: ' + name + ' exit=' + str(status) + ' reason=' + str(reason))


def environment(label):
    result = dict(os.environ, PYTHONPATH=str(SITES[label]) + ':' + str(DIRECTORY), PYTHONDONTWRITEBYTECODE='1', NV_SOURCE_LABEL=label, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    result['LD_LIBRARY_PATH'] = str(TORCH_ROOTS[label] / 'lib') + ':/opt/gfx-deps/profiling_tools/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
    return result


def provenance():
    preflight()
    build_source.verify_original()
    require = lambda condition, message: condition or (_ for _ in ()).throw(RuntimeError(message))
    post = build_source.sidecar.verify_snapshot({'source_worktree': str(SOURCE), 'source_revision': REVISION}, FRESH / 'source')
    require(post['all_kernel_bodies_helpers_preserved'], 'Source helpers/math changed')
    for phase in ('torch-cpu-probe', 'configure', 'tested-native-build', 'supervisor'):
        require((FRESH / (phase + '.exit')).read_text().strip() == '0', 'Original native prerequisite failed')
        drain = json.loads((FRESH / (phase + '.drain.json')).read_text())
        require(drain['drained'] and not drain['live_after'], 'Original descendant drain failed')
    require((FRESH / 'normal-package-cpu-check.exit').read_text().strip() == '0', 'Original package check failed')
    recipes = {name: digest(DIRECTORY / name) for name in RECIPES}
    policy = json.loads((FRESH / 'compiler-policy-verified.json').read_text())
    require(policy['original_source_flags_preserved'] and policy['NV_original_GRF512_preserved'], 'Original flags/GRF512 missing')
    return {'source_revision': REVISION, 'cutlass_revision': build_source.CUTLASS_REVISION, 'source_body_helpers_verified': post, 'fresh_binary_manifest_sha256': digest(FRESH / 'source-binary-manifest.json'), 'compiler_policy_sha256': digest(FRESH / 'compiler-policy-verified.json'), 'fresh_CPU_package_check_sha256': digest(FRESH / 'normal-package-cpu-check.json'), 'historical_package_manifest_sha256': digest(DIRECTORY / 'historical-package-manifest.json'), 'scripts_sha256': recipes, 'private_process_tree_sha256': digest(ARTIFACT / 'fresh-source-icpx-v2/process_tree.py'), 'original_tests_sha256': digest(SOURCE / 'tests/test_nvfp4_woq_moe.py'), 'original_scale_reference_sha256': digest(SOURCE / 'tests/nvfp4_utils.py'), 'original_accuracy_reference_sha256': digest(SOURCE / 'benchmark/accuracy_reference.py')}


def cpu_proof():
    proof = provenance()
    build_source.verify_retry_inputs(FRESH)
    providers = {}
    for label in ('historical', 'fresh'):
        name = 'CPU-provider-v2-' + label
        run([str(PYTHONS[label]), '-B', str(DIRECTORY / 'providers.py'), label], name, environment(label), timeout=300)
        providers[label] = {'report': json.loads((DIRECTORY / (name + '.log')).read_text()), 'log_sha256': digest(DIRECTORY / (name + '.log')), 'exit_sha256': digest(DIRECTORY / (name + '.exit'))}
    save(DIRECTORY / 'CPU-proof.json', {'actual_exit': 0, 'provenance': proof, 'GPU_initialized': False, 'providers': providers, 'original_flags_and_includes_reverified': True})


def grant():
    proof = json.loads((DIRECTORY / 'CPU-proof.json').read_text())
    assert proof['actual_exit'] == 0 and proof['provenance'] == provenance()
    save(DIRECTORY / 'grant.json', {'issuer': 'explicit-user-main-coordinator-authorization', 'authorization': 'SOURCE-only GPU9 comparison after exact provenance/CPU-provider/drain checks; not official v4 wave', 'scope': 'NV-SOURCE-native-icpx-v2-accuracy-v1', 'provenance': proof['provenance'], 'CPU_proof_sha256': digest(DIRECTORY / 'CPU-proof.json'), 'start_epoch': time.time() + 45, 'gpu': 9, 'CPU_affinity': list(range(80, 96)), 'precision_only': True, 'timing_allowed': False, 'PORT_or_AFTER_allowed': False, 'serving': {'B': 8, 'ISL': 4096, 'OSL': 16, 'TP': 1, 'EP': 1}, 'original_tests_per_provider': 50, 'tail_records': 16, 'production_records': 16, 'full_valid_outputs_and_scales_bitwise': True, 'independent_reference': 'original CPU FP32 tolerance; production max3rows/active expert, tails allrows'})


def verify_grant():
    record = json.loads((DIRECTORY / 'grant.json').read_text())
    assert record['provenance'] == provenance()
    assert record['CPU_proof_sha256'] == digest(DIRECTORY / 'CPU-proof.json')
    assert record['gpu'] == 9 and record['precision_only'] and not record['timing_allowed'] and not record['PORT_or_AFTER_allowed']
    assert set(os.sched_getaffinity(0)) == set(record['CPU_affinity'])
    return record


def suite():
    record = verify_grant()
    assert os.environ.get('ZE_AFFINITY_MASK') == '9'
    assert time.time() >= record['start_epoch']
    save(DIRECTORY / 'GPU-suite-started.json', {'PID': os.getpid(), 'startticks': build_source.sidecar.process_ticks(os.getpid()), 'epoch': time.time(), 'grant_sha256': digest(DIRECTORY / 'grant.json'), 'gpu': 9, 'precision_only': True})
    for label in ('historical', 'fresh'):
        run([str(PYTHONS[label]), '-B', '-m', 'pytest', '-q', '--run-long-tests', '-p', 'no:cacheprovider', '-p', 'accuracy_plugin', str(SOURCE / 'tests/test_nvfp4_woq_moe.py'), '--junitxml=' + str(DIRECTORY / ('tests-' + label + '.xml'))], 'tests-' + label, environment(label))
        junit = ET.parse(DIRECTORY / ('tests-' + label + '.xml'))
        cases = junit.findall('.//testcase')
        assert len(cases) == 50 and not junit.findall('.//skipped') and not junit.findall('.//failure') and not junit.findall('.//error')
    for family in ('tails', 'production'):
        for phase, label in enumerate(('historical', 'fresh', 'fresh', 'historical')):
            command = [str(PYTHONS[label]), '-B', str(DIRECTORY / 'run_source_case.py'), label, '--run', family, '--round', '0', '--phase', str(phase)]
            if family == 'tails':
                command.append('--tail-only')
            run(command, family + '-p' + str(phase) + '-' + label, environment(label))
    families = {}
    for family in ('tails', 'production'):
        records = []
        for phase, label in enumerate(('historical', 'fresh', 'fresh', 'historical')):
            report = json.loads((DIRECTORY / 'results' / family / ('round0-phase' + str(phase) + '-' + label + '.json')).read_text())
            assert len(report['results']) == 4 and report['source_only'] and report['accuracy_only']
            records.extend(report['results'])
        assert len(records) == 16
        for result in records:
            assert result['samples_us'] == [] and result['output']['exact'] and result['output']['all_elements_checked']
            assert all(value['exact'] and value['all_elements_checked'] for value in result['scales'].values())
        families[family] = {'records': len(records), 'full_valid_outputs_and_scales_bitwise': True, 'max_abs': max(result['output']['max_abs'] for result in records), 'timing_samples': 0}
    save(DIRECTORY / 'summary.json', {'actual_suite_exit': 0, 'GPU': 9, 'source_only': True, 'original_tests': {'historical': 50, 'fresh': 50}, 'skips': 0, 'families': families, 'independent_reference': 'CPU FP32 original thresholds, production max3rows/active expert; tails full reference; not whole-production oracle', 'official_AFTER_acceptance': False, 'performance_acceptance': False, 'grant_sha256': digest(DIRECTORY / 'grant.json')})


def wrapper(mode, gpu=False):
    command = ['/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh']
    if gpu:
        command.extend(['--gpu-lock', '--gpu', '9'])
    script = str(DIRECTORY / 'supervisor.py').replace('/home/syk/', '/root/', 1)
    command.extend(['--workdir', '/root', '--', 'bash', '/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh', '5', 'env', 'PYTHONDONTWRITEBYTECODE=1', '/opt/gfx-deps/venv/bin/python', '-B', script, mode])
    return command


def stop_owned():
    results = {}
    for path in DIRECTORY.glob('*.owned.json'):
        if path.name.startswith(('tests-', 'tails-', 'production-', 'CPU-provider-')):
            record = json.loads(path.read_text())
            results[path.name] = ProcessTree(0, record['token'], known=record['identities']).drain()
    save(DIRECTORY / 'GPU-owned-drain.json', results)
    assert all(result['drained'] for result in results.values())


def launch():
    preflight()
    with (DIRECTORY / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = verify_grant()
        assert not (DIRECTORY / 'GPU-suite-started.json').exists()
        assert record['start_epoch'] > time.time()
        save(DIRECTORY / 'ready.json', {'PID': os.getpid(), 'startticks': build_source.sidecar.process_ticks(os.getpid()), 'epoch': time.time(), 'grant_sha256': digest(DIRECTORY / 'grant.json'), 'start_epoch': record['start_epoch'], 'GPU_lock_held': False, 'source_only': True})
        while time.time() < record['start_epoch']:
            time.sleep(min(1, record['start_epoch'] - time.time()))
        verify_grant()
        try:
            run(wrapper('--suite', gpu=True), 'GPU-wrapper', dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015'), timeout=10800)
        except BaseException:
            subprocess.run(wrapper('--stop-owned'), env=dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015'), timeout=90, check=False)
            raise
        save(DIRECTORY / 'GPU-release.json', {'actual_wrapper_exit': 0, 'all_owned_descendants_drained': json.loads((DIRECTORY / 'GPU-wrapper.drain.json').read_text())['drained'], 'actual_wrapper_lock_released': True, 'GPU': 9, 'GPU5_performance_granted': False, 'logical_GPU5_queue_released': False})


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('--cpu-proof', '--write-grant', '--launch', '--suite', '--stop-owned'))
    mode = sys.argv[1]
    if mode not in ('--cpu-proof', '--write-grant', '--launch', '--suite', '--stop-owned'):
        parser.error('Explicit mode required')
    {'--cpu-proof': cpu_proof, '--write-grant': grant, '--launch': launch, '--suite': suite, '--stop-owned': stop_owned}[mode]()
