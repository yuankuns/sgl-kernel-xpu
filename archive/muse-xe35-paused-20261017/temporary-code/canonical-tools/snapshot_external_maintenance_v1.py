import json
import os
import time
from pathlib import Path

from build_pool_icpx_v2 import process_snapshot


ROOT = Path(__file__).resolve().parent.parent


def main():
    processes = process_snapshot()
    selected = {pid: record for pid, record in processes.items()
                if record['name'] in {'cc1', 'cc1plus', 'dkms', 'dpkg', 'make'}
                and ('dkms' in record['command'] or record['name'] in {'cc1', 'cc1plus', 'dpkg'})}
    modules = {}
    for name in ('xe', 'i915'):
        directory = Path('/sys/module') / name
        if directory.exists():
            modules[name] = {field: (directory / field).read_text().strip()
                             for field in ('version', 'srcversion', 'initstate')
                             if (directory / field).is_file()}
    report = {'epoch': time.time(), 'running_kernel': os.uname().release,
              'loaded_modules': modules, 'external_processes': selected,
              'observed_DKMS_target_kernel': '6.8.0-147-generic',
              'no_system_processes_stopped_or_modified': True,
              'agent_native_compile_workers_authorized': 0,
              'policy': 'keep accuracy under bound loaded-stack evidence; no new performance until external compiler drain and stack recheck',
              'native_SYCL_compiler': 'icpx; external system DKMS gcc/cc1 is unrelated',
              'performance_or_GPU_work_started_by_snapshot': False}
    with (ROOT / 'EXTERNAL_MAINTENANCE_SNAPSHOT_20261017_v1.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'running_kernel': report['running_kernel'], 'loaded_modules': modules,
                      'external_process_count': len(selected), 'agent_compile_workers_authorized': 0}))


if __name__ == '__main__':
    main()
