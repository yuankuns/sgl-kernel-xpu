import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

import common

sys.path.insert(0, str(common.OLD))
from process_tree import ProcessTree, budget_reason


def interrupted(signum, frame):
    raise RuntimeError('Interrupted: ' + str(signum))


def disk_check():
    size = sum(path.stat().st_size for path in common.DIRECTORY.rglob('*') if path.is_file())
    if size > common.DISK_CAP:
        return 'Artifact disk cap exceeded'
    if shutil.disk_usage(common.DIRECTORY).free < common.DISK_HEADROOM:
        return 'Artifact disk headroom below1GiB'
    return None


def run(command, name, environment):
    token = uuid.uuid4().hex
    environment = dict(environment, MUSE_SOURCE_BUILD_TOKEN=token)
    stem = common.DIRECTORY / name
    common.save(stem.with_suffix('.command.json'), command)
    with stem.with_suffix('.log').open('x') as log, stem.with_suffix('.resources.jsonl').open('x') as resources:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        tree = ProcessTree(process.pid, token)
        common.save(stem.with_suffix('.started.json'), {'PID': process.pid, 'startticks': tree.sample()['identities'], 'token': token, 'epoch': time.time()})
        started = time.monotonic()
        reason = None
        try:
            while True:
                metrics = tree.sample()
                resources.write(json.dumps(metrics) + '\n')
                resources.flush()
                reason = budget_reason(metrics) or disk_check()
                if time.monotonic() - started > 3600:
                    reason = 'Accuracy-only timeout'
                if reason:
                    tree.drain()
                    break
                if process.poll() is not None:
                    if tree.sample()['live']:
                        reason = 'Owned descendants remain after root exit'
                        tree.drain()
                    break
                time.sleep(0.25)
        except BaseException:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            common.save(stem.with_suffix('.exception-drain.json'), tree.drain())
            process.wait(timeout=20)
            stem.with_suffix('.exit').write_text(str(process.returncode) + '\n')
            raise
        status = process.wait(timeout=20)
        final = tree.sample()
        stem.with_suffix('.exit').write_text(str(status) + '\n')
        common.save(stem.with_suffix('.drain.json'), {'actual_exit': status, 'reason': reason, 'all_owned_drained': not final['live'],
                    'identities': final['identities'], 'live': final['live'], 'groups_signalled': sorted(tree.groups_signalled)})
        if reason or status or final['live']:
            raise RuntimeError('Actual failure stop, no retry: ' + name + ' ' + str(status) + ' ' + str(reason))


def suite(path):
    common.preflight()
    grant = common.grant(path)
    assert os.environ['ZE_AFFINITY_MASK'] == '11' and time.time() >= grant['start_epoch']
    fields = Path('/proc/self/stat').read_text().rpartition(')')[2].split()
    common.save(common.DIRECTORY / 'GPU-suite-started.json', {'PID': os.getpid(), 'epoch': time.time(),
                'startticks': int(fields[19]), 'token': os.environ['MUSE_SOURCE_BUILD_TOKEN'],
                'grant_sha256': common.digest(path), 'gpu': 11})
    for label in ('source', 'after'):
        environment = dict(os.environ, PYTHONPATH=str(common.SITES[label]) + ':' + str(common.DIRECTORY),
                           LD_LIBRARY_PATH='/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:/opt/gfx-deps/profiling_tools/lib:' + os.environ.get('LD_LIBRARY_PATH', ''),
                           PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
        run(['/opt/gfx-deps/venv/bin/python', '-B', str(common.DIRECTORY / 'runner.py'), label, '--grant', str(path)], label, environment)
    common.save(common.DIRECTORY / 'GPU_SUITE_RESULT.json', {'actual_exit': 0, 'cases': 4, 'phases': 2,
                'all_rows_CPU_FP32_reference': True, 'all_inputs_prep_output_SOURCE_bits': True,
                'performance_authorized': False, 'timing_samples': []})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', type=Path)
    parser.add_argument('--stop-owned', action='store_true')
    args = parser.parse_args()
    if args.stop_owned:
        common.preflight()
        results = []
        started_path = common.DIRECTORY / 'GPU-suite-started.json'
        if started_path.is_file():
            started = json.loads(started_path.read_text())
            known = [{'pid': started['PID'], 'start_ticks': started['startticks']}]
            tracker = ProcessTree(started['PID'], started['token'], known=known)
            results.append(tracker.drain())
        for label in ('source', 'after'):
            started_path = common.DIRECTORY / f'{label}.started.json'
            if started_path.is_file():
                started = json.loads(started_path.read_text())
                known = started['startticks']
                metrics_path = common.DIRECTORY / f'{label}.resources.jsonl'
                if metrics_path.is_file():
                    lines = metrics_path.read_text().splitlines()
                    for line in reversed(lines):
                        try:
                            known += json.loads(line)['identities']
                            break
                        except json.JSONDecodeError:
                            continue
                results.append(ProcessTree(started['PID'], started['token'], known=known).drain())
        common.save(common.DIRECTORY / 'explicit-stop-drain.json', {'results': results, 'all_owned_drained': all(item['drained'] for item in results)})
    elif args.suite:
        suite(args.suite)
    else:
        parser.error('suite or stop-owned required')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
