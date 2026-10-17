import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from checks import ROOT, ROUND, SCOPE, VERSION, digest, driver_hashes, grant_preflight, package_gate, write_json


def run(name, command):
    write_json(ROUND / (name + '.command.json'), command)
    with (ROUND / (name + '.log')).open('x') as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        write_json(ROUND / (name + '.started.json'), {'pid': child.pid, 'epoch': time.time()})
        status = child.wait()
    (ROUND / (name + '.exit')).write_text(str(status) + '\n')
    if name == 'GPU-round':
        write_json(ROUND / 'GPU-drained.json', {'gpu': 4, 'epoch': time.time(), 'actual_exit': status,
                                             'wrapper_returned': True, 'GPU_lock_lifetime_ended': True})
    return status


def main():
    os.umask(0o022)
    if os.environ.get('BATCHES', '8') != '8':
        raise RuntimeError('B8 only before GPU work')
    lock = (ROOT / '.canonical-official-supervisor.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROUND / 'supervisor.started.json').exists():
        raise RuntimeError('Refusing duplicate formal supervisor')
    write_json(ROUND / 'supervisor.started.json', {'pid': os.getpid(), 'epoch': time.time(), 'scope': SCOPE})
    status = 1
    try:
        manifest, manifest_file, site, protected = package_gate()
        assert digest(manifest_file) == '91e08eceb819e9e5f2b754e95581566c7d3bbee43de39fbca8d78295e7f5e398'
        assert digest(ROOT / 'fresh-source-icpx-v1/source-native-binary-manifest.json') == '071f7f85b0a2b28c1d81d182e3dc9c4ea75a32b7e2da591dc7920021a43d1446'
        queue = json.loads((ROOT.parent / 'ALL_ELEVEN_FULL_ICPX_V4_RESULTS.json').read_text())
        assert queue['actual_queue_exit'] == 0 and queue['host_queue_pid_absent']
        assert (ROOT / 'v4-after-accuracy-prewave-v1/supervisor.exit').read_text().strip() == '0'
        previous = json.loads((ROOT.parent / 'bf16-gemm/formal-v4-perf-gpu4-v1/GPU-drained.json').read_text())
        assert previous['actual_exit'] == 0 and previous['wrapper_returned']
        assert not Path('/proc/3123239').exists() and not Path('/proc/3124758').exists()
        container_root = str(ROOT).replace('/home/syk/', '/root/', 1)
        container_round = str(ROUND).replace('/home/syk/', '/root/', 1)
        wrapper = '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh'
        prefix = ['env', 'SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015', wrapper]
        cpu = prefix + ['--workdir', container_root, '--', 'bash', '-c',
                        'export ONEAPI_DEVICE_SELECTOR=level_zero:cpu PYTHONDONTWRITEBYTECODE=1; '
                        'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; '
                        'export PYTHONPATH="$1/canonical-v4-icpx/site"; '
                        'exec taskset -c 96-111 python -B "$2/cpu_gate.py"', '_', container_root, container_round]
        if run('cpu-gate', cpu):
            raise RuntimeError('Actual CPU gate failed')
        files, identity = driver_hashes()
        issued = time.time()
        epoch = issued + 120
        grant = {'stage': VERSION, 'scope': SCOPE, 'module': 'moe-permutation', 'gpu': 4, 'phase': 'performance',
                 'issuer': 'main', 'authorization': 'main explicit stable GPU4 formal performance authorization in current user message',
                 'coordinator_release': True, 'issued_epoch': issued, 'epoch': epoch, 'not_before_epoch': epoch,
                 'CPU_affinity': '96-111', 'all_eight_cards_parallel_verified': False,
                 'source_binary_manifest_sha256': digest(manifest_file), 'source_digest': protected['source_digest'],
                 'source_freeze_sha256': protected['freeze_sha256'],
                 'compiler_policy_sha256': digest(ROOT / VERSION / 'compiler-policy-verified.json'),
                 'source_native_manifest_sha256': digest(ROOT / 'fresh-source-icpx-v1/source-native-binary-manifest.json'),
                 'driver_sha256': identity, 'driver_files_sha256': files,
                 'cpu_gate_sha256': digest(ROUND / 'cpu-normal-package-gate.json'),
                 'full_queue_sha256': digest(ROOT.parent / 'ALL_ELEVEN_FULL_ICPX_V4_RESULTS.json'),
                 'AFTER_precision_summary_sha256': digest(ROOT / 'v4-after-accuracy-prewave-v1/summary.json'),
                 'previous_BF_drained_sha256': digest(ROOT.parent / 'bf16-gemm/formal-v4-perf-gpu4-v1/GPU-drained.json'),
                 'previous_bf16_drained': True, 'minimum_BAAB_rounds': 3,
                 'shared_SOURCE_inputs_sha256': digest(ROUND / 'shared-SOURCE-inputs.pt')}
        write_json(ROUND / 'fresh-grant.json', grant)
        (ROUND / 'fresh-grant.json').chmod(0o444)
        print(json.dumps({'supervisor_pid': os.getpid(), 'fresh_grant': grant, 'waiting_until_epoch': epoch}), flush=True)
        time.sleep(max(0, epoch - time.time()))
        grant_preflight(ROOT, ROUND / 'fresh-grant.json', False)
        gpu = prefix + ['--gpu-lock', '--gpu', '4', '--workdir', container_root,
                        '--', 'bash', container_round + '/gpu_container.sh']
        gpu_status = run('GPU-round', gpu)
        if gpu_status:
            raise RuntimeError('Actual GPU round failed: ' + str(gpu_status))
        analysis_statuses = []
        for cycle in [1, 2, 3]:
            analyzer = prefix + ['--workdir', container_root, '--', 'bash', '-c',
                                 'export ONEAPI_DEVICE_SELECTOR=level_zero:cpu PYTHONDONTWRITEBYTECODE=1; '
                                 'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; '
                                 'exec taskset -c 96-111 python -B "$1/analyze.py" "$2"', '_', container_round,
                                 'canonical-v4-icpx-formal-gpu4-v1-round' + str(cycle) + '-BAAB']
            code = run('analyze-round' + str(cycle), analyzer)
            if code not in [0, 3]:
                raise RuntimeError('Accuracy or analysis failure: ' + str(code))
            analysis_statuses.append(code)
        aggregate = prefix + ['--workdir', container_root, '--', 'bash', '-c',
                              'export ONEAPI_DEVICE_SELECTOR=level_zero:cpu PYTHONDONTWRITEBYTECODE=1; '
                              'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; '
                              'exec taskset -c 96-111 python -B "$1/aggregate.py"', '_', container_round]
        aggregate_status = run('aggregate', aggregate)
        if aggregate_status not in [0, 3]:
            raise RuntimeError('Aggregate accuracy or analysis failed: ' + str(aggregate_status))
        status = 3 if 3 in analysis_statuses or aggregate_status == 3 else 0
        manifest, manifest_file, site, protected = package_gate()
        write_json(ROUND / 'protected-source-postverify.json', protected)
        inventory = {path.name: digest(path) for path in sorted(ROUND.iterdir()) if path.is_file()}
        write_json(ROUND / 'manifest.json', {'files_sha256': inventory, 'epoch': time.time(),
                                          'source_digest': protected['source_digest'], 'AFTER_manifest_sha256': digest(manifest_file),
                                          'analysis_actual_exit': status, 'GPU_round_actual_exit': gpu_status,
                                          'all_eight_cards_parallel_verified': False, 'coordinator_review_pending': True})
    except Exception:
        status = 1
        traceback.print_exc()
    (ROUND / 'supervisor.exit').write_text(str(status) + '\n')
    print(json.dumps({'supervisor_actual_exit': status, 'evidence': str(ROUND), 'performance_not_auto_accepted': True}), flush=True)
    raise SystemExit(status)


if __name__ == '__main__':
    main()
