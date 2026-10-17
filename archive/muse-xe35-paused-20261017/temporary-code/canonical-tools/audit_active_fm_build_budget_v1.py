import json
import os
import time
from pathlib import Path

from build_pool_icpx_v2 import process_snapshot


ROOT = Path(__file__).resolve().parent.parent


def main():
    processes = process_snapshot()
    build_path = str(ROOT / 'canonical-icpx-pool-v4/build').replace('/home/syk/', '/root/', 1)
    parents = {pid: record for pid, record in processes.items()
               if record['name'] == 'cmake' and f'--build {build_path}' in record['command']}
    assert len(parents) == 1, parents
    selected = set(parents)
    while True:
        expanded = selected | {pid for pid, record in processes.items() if str(record['ppid']) in selected}
        if expanded == selected:
            break
        selected = expanded
    records = {}
    vanished = []
    for pid in sorted(selected):
        try:
            affinity = sorted(os.sched_getaffinity(int(pid)))
            status = (Path('/proc') / pid / 'status').read_text()
        except ProcessLookupError:
            vanished.append(pid)
            continue
        except FileNotFoundError:
            vanished.append(pid)
            continue
        assert affinity and set(affinity) <= set(range(128)), (pid, affinity)
        rss = next((line.split(':', 1)[1].strip() for line in status.splitlines() if line.startswith('VmRSS:')), None)
        records[pid] = {'process': processes[pid], 'actual_CPU_affinity': affinity, 'RSS': rss}
    preflight = json.loads((ROOT / 'fmha/canonical-v5-icpx/preflight.json').read_text())
    assert preflight['outer_jobs'] == 56 and preflight['inner_jobs'] == 1 and preflight['cpu_limit'] == 128
    policy = json.loads((ROOT / 'fmha/canonical-v5-icpx/compiler-policy-verified.json').read_text())
    assert policy['no_GNU_command'] and policy['cxx'].endswith('/icpx')
    report = {'actual_exit': 0, 'epoch': time.time(), 'build_parent': parents,
              'sampled_live_build_processes': records, 'processes_vanished_during_read': vanished,
              'all_sampled_build_process_CPU_affinities_within_0_127': True,
              'configured_outer_jobs': 56, 'configured_inner_jobs': 1,
              'compile_CPU_limit': 128, 'scope': 'One live build subtree and actual process affinities; wrapper/compiler subprocess count is not the CPU/job limit',
              'GPU_work_started': False, 'performance_acceptance_claimed': False}
    with (ROOT / 'fmha/COORDINATOR_ACTIVE_BUILD_BUDGET_v5.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'live_processes_checked': len(records), 'configured_outer_jobs': 56,
                      'configured_inner_jobs': 1, 'compile_CPU_limit': 128}))


if __name__ == '__main__':
    main()
