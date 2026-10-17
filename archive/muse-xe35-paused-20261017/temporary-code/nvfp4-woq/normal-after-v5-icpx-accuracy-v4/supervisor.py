import argparse
import fcntl
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

from process_tree import ProcessTree
from providers import ARTIFACT, CANONICAL, DIRECTORY, FRESH, PYTHONS, SITES, SOURCE_ACCURACY, SOURCE_READY_SHA, TORCH_ROOT, canonical_result, digest, preflight, source_result


VERSION = 'canonical-v5-icpx'
RUN = 'nv-normal-v5-icpx-accuracy-v4'
SOURCE_DIGEST = 'dd2adcfa688cd0df1cc154fbbd2ffa9105c53309f154b7aa3a7275cd6edd97a2'
RECIPES = ('canonical_check_v5.py', 'providers.py', 'accuracy_plugin.py', 'run_accuracy_case.py', 'process_tree.py', 'supervisor.py')
TARGET = ARTIFACT.parents[1] / 'pr/muse-nvfp4-woq-xe35'
SOURCE = ARTIFACT.parents[2] / 'sgl-kernel-xpu/muse-spark-nvfp4-woq-main-20261014'


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def interrupted(signum, frame):
    raise RuntimeError('Interrupted: ' + str(signum))


def startticks(pid):
    return (Path('/proc') / str(pid) / 'stat').read_text().rpartition(')')[2].split()[19]


def run(command, name, environment, timeout=3600):
    stem = DIRECTORY / name
    stem.with_suffix('.command').write_text(shlex.join(command) + '\n')
    token = uuid.uuid4().hex
    environment = dict(environment, MUSE_SOURCE_BUILD_TOKEN=token)
    start = time.time()
    reason = None
    with stem.with_suffix('.log').open('x') as output, stem.with_suffix('.resources.jsonl').open('x') as resources:
        process = subprocess.Popen(command, env=environment, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        tracker = ProcessTree(process.pid, token)
        save(stem.with_suffix('.started.json'), {'PID': process.pid, 'startticks': startticks(process.pid), 'token': token, 'epoch': start, 'command': command})
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
    save(stem.with_suffix('.terminal.json'), {'actual_exit': status, 'elapsed_seconds': time.time() - start, 'stop_reason': reason, 'all_descendants_drained': not final['live'], 'RSS_sampled': True, 'compiler_budget_applied': False, 'accuracy_only': True})
    if status or reason or final['live']:
        raise RuntimeError('Actual failure: ' + name + ' exit=' + str(status) + ' reason=' + str(reason))


def environment(label):
    result = dict(os.environ, PYTHONPATH=str(SITES[label]) + ':' + str(DIRECTORY), PYTHONDONTWRITEBYTECODE='1', NV_ACCURACY_LABEL=label, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    result['LD_LIBRARY_PATH'] = str(TORCH_ROOT / 'lib') + ':/opt/gfx-deps/profiling_tools/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
    return result


def confirmation(path, result):
    data = json.loads(path.read_text())
    expected = {'component': 'nvfp4-woq', 'version': VERSION, 'actual_exit': 0, 'default_all_actual_exit': 0, 'install_actual_exit': 0, 'source_digest': SOURCE_DIGEST, 'package_manifest_sha256': result['manifest_sha256'], 'compiler_policy_sha256': result['compiler_policy_sha256'], 'installed_and_python_files_rehashed': 1675, 'native_binary_inventory_rehashed': 714, 'actual_CXX_SYCL_host_link': 'icpx', 'actual_C': 'icx', 'native_math_unchanged': True}
    assert all(data.get(name) == value for name, value in expected.items()), 'Main true0 and independent source/native/Python audit required'
    drain = json.loads((CANONICAL / 'COORDINATOR_COMPILER_DRAIN.json').read_text())
    assert drain['actual_audit_exit'] == 0 and drain['main_authorized_NV_build_handle'] == 25442
    assert drain['build_actual_exit'] == 0 and drain['install_actual_exit'] == 0 and drain['post_verified']
    assert drain['canonical_process_audit_drained'] and not drain['matching_live_compiler_processes']
    assert result['manifest_sha256'] == 'a4492cadb1020191435f184e2859bbe3ae47e7305a98b02e782562296a4e9dba'
    return data


def provenance():
    preflight()
    source_result()
    result = canonical_result()
    assert result['canonical_version'] == VERSION and result['source_digest'] == SOURCE_DIGEST
    sys.path.insert(0, str(ARTIFACT / 'fresh-source-icpx-v2'))
    import build_source

    build_source.verify_original()
    post = build_source.sidecar.verify_snapshot({'source_worktree': str(SOURCE), 'source_revision': 'd983e877d6bf97015027bd3b1379ff047009a5dd'}, FRESH / 'source')
    return {'canonical_version': VERSION, 'source_digest': SOURCE_DIGEST, 'canonical_manifest_sha256': result['manifest_sha256'], 'compiler_policy_sha256': result['compiler_policy_sha256'], 'main_compiler_drain_sha256': digest(CANONICAL / 'COORDINATOR_COMPILER_DRAIN.json'), 'main_source_freeze_sha256': digest(ARTIFACT / 'canonical-source-freeze-v5-icpx.json'), 'SOURCE_READY_sha256': SOURCE_READY_SHA, 'source_binary_manifest_sha256': digest(FRESH / 'source-binary-manifest.json'), 'source_driver_policy_sha256': digest(FRESH / 'compiler-policy-verified.json'), 'source_body_helpers_postverified': post, 'source_revision': 'd983e877d6bf97015027bd3b1379ff047009a5dd', 'source_cutlass_revision': 'e87870bb7aacf83bfb3be9da685f148316b8df23', 'AFTER_public_cutlass_revision': '7b56b936b87f799f80dde60c2fd8d11c1128cebb', 'scripts_sha256': {name: digest(DIRECTORY / name) for name in RECIPES}, 'original_SOURCE_tests_sha256': digest(SOURCE / 'tests/test_nvfp4_woq_moe.py'), 'normal_AFTER_tests_sha256': digest(TARGET / 'tests/test_nvfp4_woq_moe.py'), 'original_accuracy_reference_sha256': digest(SOURCE / 'benchmark/accuracy_reference.py'), 'original_scale_reference_sha256': digest(SOURCE / 'tests/nvfp4_utils.py')}


def cpu_proof(confirmation_path):
    result = canonical_result()
    confirmation(confirmation_path, result)
    record = provenance()
    providers = {}
    for label in ('source', 'after'):
        name = 'CPU-provider-' + label
        run([str(PYTHONS[label]), '-B', str(DIRECTORY / 'providers.py'), label, '--version', VERSION], name, environment(label), timeout=900)
        providers[label] = {'report': json.loads((DIRECTORY / (name + '.log')).read_text()), 'log_sha256': digest(DIRECTORY / (name + '.log')), 'exit_sha256': digest(DIRECTORY / (name + '.exit'))}
    assert providers['source']['report']['torch_C_sha256'] == providers['after']['report']['torch_C_sha256']
    assert providers['source']['report']['torch_xpu_sha256'] == providers['after']['report']['torch_xpu_sha256']
    save(DIRECTORY / 'CPU-proof.json', {'actual_exit': 0, 'provenance': record, 'main_confirmation_path': str(confirmation_path), 'main_confirmation_sha256': digest(confirmation_path), 'providers': providers, 'GPU_initialized': False})
    save(DIRECTORY / 'READY.json', {'CPU_READY': True, 'GPU_launched': False, 'canonical_version': VERSION, 'run': RUN, 'GPU': 9, 'CPU_affinity': list(range(80, 96)), 'CPU_proof_sha256': digest(DIRECTORY / 'CPU-proof.json'), 'provenance': record, 'grant_issued': False, 'performance_granted': False})


def verify_grant(path):
    preflight()
    data = json.loads(path.read_text())
    proof = json.loads((DIRECTORY / 'CPU-proof.json').read_text())
    current = provenance()
    assert proof['actual_exit'] == 0 and proof['provenance'] == current
    main_confirmation = Path(proof['main_confirmation_path'])
    if str(DIRECTORY).startswith('/root/'):
        main_confirmation = Path(str(main_confirmation).replace('/home/syk/', '/root/', 1))
    else:
        main_confirmation = Path(str(main_confirmation).replace('/root/', '/home/syk/', 1))
    assert digest(main_confirmation) == proof['main_confirmation_sha256']
    confirmation(main_confirmation, canonical_result())
    expected = {'issuer': 'main', 'component': 'nvfp4-woq', 'canonical_version': VERSION, 'run': RUN, 'phase': 'accuracy', 'gpu': 9, 'CPU_affinity': list(range(80, 96)), 'coordinator_release': True, 'accuracy_only': True, 'performance_allowed': False, 'default_all_actual_exit': 0, 'install_actual_exit': 0, 'post_actual_exit': 0, 'supervisor_actual_exit': 0, 'normal_import_actual_exit': 0, 'source_digest': SOURCE_DIGEST, 'canonical_manifest_sha256': current['canonical_manifest_sha256'], 'compiler_policy_sha256': current['compiler_policy_sha256'], 'SOURCE_READY_sha256': SOURCE_READY_SHA, 'source_binary_manifest_sha256': current['source_binary_manifest_sha256'], 'scripts_sha256': current['scripts_sha256'], 'CPU_proof_sha256': digest(DIRECTORY / 'CPU-proof.json'), 'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1}
    assert all(data.get(name) == value for name, value in expected.items()), 'Main fresh v5 accuracy grant mismatch'
    assert isinstance(data['start_epoch'], (int, float)) and data['start_epoch'] > 0
    assert set(os.sched_getaffinity(0)) == set(range(80, 96))
    return data


def suite(grant_path):
    record = verify_grant(grant_path)
    assert os.environ.get('ZE_AFFINITY_MASK') == '9' and time.time() >= record['start_epoch']
    os.environ['NV_V5_ACCURACY_GRANT'] = str(grant_path)
    save(DIRECTORY / 'GPU-suite-started.json', {'PID': os.getpid(), 'startticks': startticks(os.getpid()), 'epoch': time.time(), 'grant_sha256': digest(grant_path), 'gpu': 9, 'accuracy_only': True, 'canonical_version': VERSION})
    run([str(PYTHONS['after']), '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-p', 'accuracy_plugin', str(TARGET / 'tests/test_nvfp4_woq_moe.py'), '--junitxml=' + str(DIRECTORY / 'original-tests-after.xml')], 'original-tests-after', environment('after'))
    junit = ET.parse(DIRECTORY / 'original-tests-after.xml')
    assert len(junit.findall('.//testcase')) == 50 and not junit.findall('.//skipped') and not junit.findall('.//failure') and not junit.findall('.//error')
    for family in ('tails', 'production'):
        for phase, label in enumerate(('source', 'after', 'after', 'source')):
            command = [str(PYTHONS[label]), '-B', str(DIRECTORY / 'run_accuracy_case.py'), label, '--version', VERSION, '--run', family, '--round', '0', '--phase', str(phase)]
            if family == 'tails':
                command.append('--tail-only')
            run(command, family + '-p' + str(phase) + '-' + label, environment(label))
    families = {}
    for family in ('tails', 'production'):
        records = []
        for phase, label in enumerate(('source', 'after', 'after', 'source')):
            report = json.loads((DIRECTORY / 'results' / family / ('round0-phase' + str(phase) + '-' + label + '.json')).read_text())
            assert len(report['results']) == 4 and report['normal_AFTER'] and report['accuracy_only'] and report['canonical_version'] == VERSION
            records.extend(report['results'])
        assert len(records) == 16
        for result in records:
            assert result['samples_us'] == [] and result['output']['exact'] and result['output']['all_elements_checked']
            assert all(value['exact'] and value['all_elements_checked'] for value in result['scales'].values())
        families[family] = {'records': len(records), 'full_valid_outputs_and_scales_bitwise': True, 'max_abs': max(result['output']['max_abs'] for result in records), 'timing_samples': 0}
    save(DIRECTORY / 'summary.json', {'actual_suite_exit': 0, 'GPU': 9, 'canonical_version': VERSION, 'normal_AFTER_accuracy_verified': True, 'original_tests': 50, 'skips': 0, 'families': families, 'independent_reference': 'original CPU FP32 tolerances, production max3rows/active expert; tails all rows; whole output/scales bitwise is not full production independent oracle', 'performance_acceptance': False, 'full_task_acceptance': False, 'grant_sha256': digest(grant_path)})


def wrapper(mode, grant_path=None, gpu=False):
    command = ['/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh']
    if gpu:
        command.extend(['--gpu-lock', '--gpu', '9'])
    script = str(DIRECTORY / 'supervisor.py').replace('/home/syk/', '/root/', 1)
    command.extend(['--workdir', '/root', '--', 'bash', '/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh', '5', 'env', 'PYTHONDONTWRITEBYTECODE=1', 'GIT_CONFIG_COUNT=2', 'GIT_CONFIG_KEY_0=safe.directory', 'GIT_CONFIG_VALUE_0=/root/worktrees/sgl-kernel-xpu-up/pr/muse-nvfp4-woq-xe35', 'GIT_CONFIG_KEY_1=safe.directory', 'GIT_CONFIG_VALUE_1=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/deps/sycl-tla-7b56', '/opt/gfx-deps/venv/bin/python', '-B', script, '--mode', mode.removeprefix('--')])
    if grant_path is not None:
        command.extend(['--grant', str(grant_path).replace('/home/syk/', '/root/', 1)])
    return command


def stop_owned():
    results = {}
    for path in DIRECTORY.glob('*.owned.json'):
        if path.name.startswith(('original-tests-', 'tails-', 'production-', 'CPU-provider-')):
            record = json.loads(path.read_text())
            results[path.name] = ProcessTree(0, record['token'], known=record['identities']).drain()
    save(DIRECTORY / 'GPU-owned-drain.json', results)
    assert all(result['drained'] for result in results.values())


def launch(grant_path):
    preflight()
    with (DIRECTORY / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = verify_grant(grant_path)
        assert not (DIRECTORY / 'GPU-suite-started.json').exists()
        assert record['start_epoch'] > time.time()
        save(DIRECTORY / 'waiter-ready.json', {'PID': os.getpid(), 'startticks': startticks(os.getpid()), 'epoch': time.time(), 'grant_sha256': digest(grant_path), 'start_epoch': record['start_epoch'], 'GPU_lock_held': False, 'canonical_version': VERSION})
        while time.time() < record['start_epoch']:
            time.sleep(min(1, record['start_epoch'] - time.time()))
        verify_grant(grant_path)
        try:
            run(wrapper('--suite', grant_path, gpu=True), 'GPU-wrapper', dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015'), timeout=10800)
        except BaseException:
            subprocess.run(wrapper('--stop-owned'), env=dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015'), timeout=90, check=False)
            raise
        save(DIRECTORY / 'GPU-release.json', {'actual_wrapper_exit': 0, 'all_owned_descendants_drained': json.loads((DIRECTORY / 'GPU-wrapper.drain.json').read_text())['drained'], 'actual_wrapper_lock_released': True, 'GPU': 9, 'GPU5_performance_granted': False, 'logical_GPU5_queue_released': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=(VERSION,), default=VERSION)
    parser.add_argument('--mode', choices=('cpu-proof', 'launch', 'suite', 'stop-owned'), required=True)
    parser.add_argument('--confirmation', type=Path)
    parser.add_argument('--grant', type=Path)
    args = parser.parse_args()
    if args.mode == 'cpu-proof':
        if args.confirmation is None:
            parser.error('Main confirmation required')
        cpu_proof(args.confirmation)
    elif args.mode == 'launch':
        if args.grant is None:
            parser.error('Main futureepoch grant required')
        launch(args.grant)
    elif args.mode == 'suite':
        if args.grant is None:
            parser.error('Main futureepoch grant required')
        suite(args.grant)
    else:
        stop_owned()


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
