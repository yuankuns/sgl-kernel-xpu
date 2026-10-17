import json
import os
from pathlib import Path
import signal
import subprocess
import time

from protocol import DIRECTORY, PERFORMANCE, save, ur_direct


def main():
    grant = json.loads((DIRECTORY / 'main-gpu0-diagnostic-grant.json').read_text())
    assert sorted(os.sched_getaffinity(0)) == grant['CPU_affinity']
    save(DIRECTORY / 'container-supervisor.json', {'PID': os.getpid(), 'epoch': time.time(), 'GPU': 0,
         'CPU_affinity': sorted(os.sched_getaffinity(0)), 'disk_cap_bytes': 6 * 1024**3, 'each_raw_log_cap_bytes': 32 * 1024**2})
    for kind in ('source', 'after'):
        check = json.loads((PERFORMANCE / f'CPU-normal-{kind}.json').read_text())
        package = Path(check['package'])
        site = check.get('site', str(package.parent.parent if package.name == '__init__.py' else package.parent))
        environment = os.environ.copy()
        environment.update({'PYTHONPATH': f'{site}:{DIRECTORY}:{DIRECTORY.parent}', 'PYTHONDONTWRITEBYTECODE': '1',
                            'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'SYCL_UR_TRACE': '3',
                            'UR_LOG_TRACING': 'level:info;output:stdout;flush:info'})
        command = ['/opt/gfx-deps/venv/bin/python', str(DIRECTORY / 'diagnose.py'), '--kind', kind,
                   '--serving-batch', '8', '--isl', '4096', '--osl', '16', '--tp', '1', '--ep', '1']
        log = DIRECTORY / f'{kind}-raw-UR.log'
        with log.open('xb') as stream:
            child = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            save(DIRECTORY / f'{kind}-child.json', {'PID': child.pid, 'PPID': os.getpid(), 'command': command, 'environment': environment})
            deadline = time.monotonic() + 900
            peak = 0
            guard = None
            while child.poll() is None:
                total = sum(path.stat().st_size for path in DIRECTORY.iterdir() if path.is_file())
                peak = max(peak, total)
                if total > 6 * 1024**3 or log.stat().st_size > 32 * 1024**2 or time.monotonic() > deadline:
                    guard = {'disk_bytes': total, 'log_bytes': log.stat().st_size, 'timeout': time.monotonic() > deadline}
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
                time.sleep(.5)
            actual = child.wait()
        save(DIRECTORY / f'{kind}-supervisor-result.json', {'actual_child_exit': actual, 'guard': guard,
             'peak_artifact_bytes': peak, 'no_automatic_retry': True, 'owned_child_drained': not Path(f'/proc/{child.pid}').exists()})
        (DIRECTORY / f'{kind}.exit').write_text(str(actual) + '\n')
        if guard is not None:
            raise SystemExit(75)
        if actual != 0:
            raise SystemExit(actual if actual > 0 else 128 - actual)
        direct_records = {shape: ur_direct(log.read_text(), shape, kind) for shape in ('262144-4096-4096', '262144-8192-4096')}
        save(DIRECTORY / f'{kind}-direct-submissions.json', direct_records)
    results = {}
    for shape in ('262144-4096-4096', '262144-8192-4096'):
        roles = {kind: json.loads((DIRECTORY / f'{kind}-{shape}-result.json').read_text()) for kind in ('source', 'after')}
        direct = {kind: json.loads((DIRECTORY / f'{kind}-direct-submissions.json').read_text())[shape] for kind in roles}
        comparisons = []
        for index, calls in enumerate((1, 8)):
            nodes = {kind: roles[kind]['captures'][index]['actual_GEMM_nodes'][0] for kind in roles}
            assert all(nodes[kind]['kernel_name'] == direct[kind]['name'] for kind in roles), 'direct/capture logical instance differs; preserve raw before review'
            comparisons.append({'calls': calls, 'logical_pipeline_tile_equal': nodes['source']['pipeline_and_tile'] == nodes['after']['pipeline_and_tile'],
                                'actual_driver_group_counts_equal': nodes['source']['group_counts'] == nodes['after']['group_counts'],
                                'actual_driver_local_size_equal': nodes['source']['local_size'] == nodes['after']['local_size'],
                                'SOURCE': nodes['source'], 'AFTER': nodes['after']})
        results[shape] = comparisons
    save(DIRECTORY / 'CAUSAL_COMPARISON.json', {'actual_evidence_collection_exit': 0, 'comparisons': results,
         'performance_acceptance': False, 'same_instance_grid_not_proof_of_equal_instruction_semantics_or_speed': True,
         'direct_full_grid_not_observed_by_UR': True, 'no_retakes_math_or_dispatch_changes': True})


if __name__ == '__main__':
    main()
