import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from host_scan import snapshot


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent
PREP = ART / 'v5-gpu5-performance-preparation'
MAIN = ART.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(name, data):
    with (ROOT / name).open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def issue():
    for name in ('PERFORMANCE_POOL_ROUTING_POST_DRAIN_v1', 'MAINTENANCE_QUIET_AFTER_ROUTING_v4', 'ROUTING_CURRENT_SOURCE_MAIN_PACKAGE_AUDIT_v1'):
        assert (MAIN / f'{name}.exit').read_text().strip() == '0'
    for role in ('source', 'port'):
        assert (ROOT / f'{role}-current-normal-CPU.exit').read_text().strip() == '0'
        gate = json.loads((ROOT / f'{role}-current-normal-CPU.json').read_text())
        assert gate['actual_exit'] == 0 and not gate['GPU_initialized'] and gate['backend_wrapper_actual_identity']
        assert gate['mapped_native_count'] == (156 if role == 'source' else 735)
    assert digest(PREP / 'performance_driver.py') == 'c96e671fbc392e8d1b4f4c54346cd40a89b82468206b0b8832cef4ea6bf016e7'
    scan = snapshot()
    write_new('before-grant-host-scan.json', scan)
    issued = time.time()
    gate = json.loads((ROOT / 'port-current-normal-CPU.json').read_text())
    grant = {'phase': 'fmha-v5-gpu5-source-port-all89-three-BAAB', 'gpu': 5, 'runtime_cpus': '16-31',
        'canonical_version': 'canonical-v5-icpx', 'source_digest': 'bce898956487b009425edb479cf26ba7e926a67572fc57943d836a8b0f0931be',
        'issued_epoch': issued, 'epoch': issued + 120, 'coordinator_release': True, 'closed': False,
        'runtime_reuse_permitted': False, 'NV_handoff_authorized': False, 'main_stable_pool_quiet_release': True,
        'preopt_source_authorized': False, 'workload': {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1, 'dtype': 'BF16'},
        'rounds': 3, 'slot_order': ['B0', 'A1', 'A2', 'B3'], 'cases_per_provider_per_slot': 89,
        'warmup': 5, 'iterations': 21, 'graph_replays': 8,
        'canonical_manifest_sha256': gate['canonical_manifest_sha256'],
        'driver_sha256': digest(PREP / 'performance_driver.py'), 'analyzer_sha256': digest(PREP / 'analyze_three_BAAB.py'),
        'finalseal_sha256': digest(PREP / 'GPU10_V5_PRECISION_FINAL_SEAL.json'),
        'baseline_audit_sha256': digest(PREP / 'ORIGINAL_SOURCE_PREOPT_CURRENT_NATIVE_CPU_AUDIT.json'),
        'model51_plan_sha256': digest(ART / 'model-config51-v1/plan.json'),
        'normal_CPU_gate_sha256': digest(ART / 'v5-model51-gpu10-preparation/normal-v5-CPU-gate-v2.json'),
        'compiler_policy_sha256': gate['compiler_scope_and_confound']['compiler_policy_sha256'],
        'source_package_manifest_sha256': digest(ART / 'matched-native-icpx-v1/source/normal-package-manifest.json'),
        'source_fresh_current_CPU_gate_sha256': digest(ROOT / 'source-current-normal-CPU.json'),
        'port_fresh_current_CPU_gate_sha256': digest(ROOT / 'port-current-normal-CPU.json'),
        'host_supervisor_sha256': digest(Path(__file__)), 'host_scan_sha256': digest(ROOT / 'host_scan.py'),
        'container_supervisor_sha256': digest(ROOT / 'container_suite.py'),
        'main_gate_SHA': {name: digest(MAIN / f'{name}.json') for name in
            ('PERFORMANCE_POOL_ROUTING_POST_DRAIN_v1', 'MAINTENANCE_QUIET_AFTER_ROUTING_v4', 'ROUTING_CURRENT_SOURCE_MAIN_PACKAGE_AUDIT_v1')},
        'before_grant_actual_host_scan_sha256': digest(ROOT / 'before-grant-host-scan.json'),
        'source_compiler_boundary': 'three fresh tested math natives real icpx; registration/non-target components historical GNU/icx; e878/profiling ON vs PORT publicpin/OFF confounds retained',
        'first_failure': 'preserve raw/exit and stop whole suite; no automatic retry',
        'authorization': 'main explicit source-port only stable GPU5 full89 three BAAB; no preopt/NV grant'}
    write_new('gpu5-source-port-barrier-v1.json', grant)
    print(json.dumps({'grant': str(ROOT / 'gpu5-source-port-barrier-v1.json'), 'epoch': grant['epoch'], 'GPU_started': False}), flush=True)


def run():
    grant_path = ROOT / 'gpu5-source-port-barrier-v1.json'
    grant = json.loads(grant_path.read_text())
    assert digest(Path(__file__)) == grant['host_supervisor_sha256']
    assert digest(ROOT / 'host_scan.py') == grant['host_scan_sha256']
    assert digest(ROOT / 'container_suite.py') == grant['container_supervisor_sha256']
    write_new('host-supervisor-start.json', {'supervisor_host_pid': os.getpid(), 'epoch': time.time(),
        'grant_sha256': digest(grant_path), 'target_epoch': grant['epoch'], 'GPU_started': False})
    print(json.dumps({'host_supervisor_pid': os.getpid(), 'barrier_epoch': grant['epoch'], 'GPU_started': False}), flush=True)
    time.sleep(max(0, grant['epoch'] - time.time()))
    write_new('before-launch-host-scan.json', snapshot())
    for role in ('source', 'port'):
        assert digest(ROOT / f'{role}-current-normal-CPU.json') == grant[f'{role}_fresh_current_CPU_gate_sha256']
    environment = dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015', PYTHONDONTWRITEBYTECODE='1')
    container_root = str(ROOT).replace('/home/syk/', '/root/', 1)
    command = ['/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh', '--gpu-lock', '--gpu', '5',
        '--workdir', str(ART).replace('/home/syk/', '/root/', 1), '--', 'bash', '-c',
        'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; '
        'export PYTHONDONTWRITEBYTECODE=1 FMHA_CANONICAL_VERSION=canonical-v5-icpx; '
        'exec taskset -c 16-31 /opt/gfx-deps/venv/bin/python "$1/container_suite.py"', 'bash', container_root]
    with (ROOT / 'suite.log').open('x') as log:
        child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        write_new('wrapper-start.json', {'wrapper_host_pid': child.pid, 'supervisor_host_pid': os.getpid(),
            'epoch': time.time(), 'command': command})
        print(json.dumps({'wrapper_host_pid': child.pid, 'GPU5_started': True}), flush=True)
        status = child.wait()
    with (ROOT / 'suite.exit').open('x') as stream:
        stream.write(f'{status}\n')
    write_new('after-suite-host-scan.json', snapshot())
    analysis_status = None
    if status == 0:
        analyzer = PREP / 'analyze_three_BAAB.py'
        assert digest(analyzer) == grant['analyzer_sha256']
        with (ROOT / 'analysis.log').open('x') as log:
            result = subprocess.run([sys.executable, '-S', str(analyzer), '--input-dir', str(ROOT),
                '--comparison', 'source-port', '--output', str(ROOT / 'source-port-three-BAAB-summary.json')],
                stdout=log, stderr=subprocess.STDOUT, env=environment)
        analysis_status = result.returncode
        with (ROOT / 'analysis.exit').open('x') as stream:
            stream.write(f'{analysis_status}\n')
    write_new('grant-closed.json', {'epoch': time.time(), 'actual_suite_exit': status,
        'actual_CPU_analysis_exit': analysis_status,
        'original_immutable_grant_sha256': digest(grant_path), 'closed': True,
        'runtime_reuse_permitted': False, 'GPU5_FM_logical_lease_retained': True, 'NV_handoff_authorized': False})
    print(json.dumps({'suite_actual_exit': status, 'grant_closed': True, 'no_retry': True}), flush=True)
    raise SystemExit(status if status != 0 else analysis_status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('issue', 'run'))
    options = parser.parse_args()
    issue() if options.action == 'issue' else run()
