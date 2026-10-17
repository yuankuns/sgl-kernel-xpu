import hashlib
import json
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / 'routing-quant/fresh-current-source-icpx-native-v3'


def sample():
    processes = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            status = dict(line.split(':', 1) for line in (directory / 'status').read_text().splitlines() if ':' in line)
            if status['State'].strip().startswith(('Z', 'X')):
                continue
            command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
            processes[int(directory.name)] = {'pid': int(directory.name), 'ppid': int(status['PPid']),
                'command': command, 'affinity': sorted(os.sched_getaffinity(int(directory.name))),
                'RSS_KiB': int(status.get('VmRSS', '0 kB').split()[0])}
        except (FileNotFoundError, ProcessLookupError):
            continue
    anchors = {pid for pid, process in processes.items()
               if str(STAGE) in process['command'] and
               Path(process['command'].split()[0]).name in ('icpx', 'icx', 'ninja', 'cmake', 'clang++', 'clang')}
    selected = set(anchors)
    while True:
        descendants = {pid for pid, process in processes.items() if process['ppid'] in selected}
        if descendants <= selected:
            break
        selected.update(descendants)
    records = [processes[pid] for pid in sorted(selected)]
    assert records, 'No live stage compiler tree; this is not a drain audit'
    assert all(set(process['affinity']) <= set(range(64, 80)) for process in records)
    assert len(records) <= 128
    return {'epoch': time.time(), 'live_process_count': len(records),
            'RSS_sum_KiB_shared_pages_may_be_counted_multiple_times': sum(process['RSS_KiB'] for process in records),
            'processes': records}


def main():
    assert set(os.sched_getaffinity(0)) == set(range(112, 128))
    grant_path = ROOT / 'routing-quant/fresh-current-source-native-v3-preparation/main-current-source-four-native-compile-grant-v1.json'
    grant = json.loads(grant_path.read_text())
    assert grant['compile_authorized'] and not grant['GPU_authorized']
    assert grant['CPU_affinity'] == list(range(64, 80))
    assert grant['outer_jobs'] == 2 and grant['MAX_JOBS'] == 1
    samples = []
    for sample_index in range(3):
        if sample_index:
            time.sleep(5)
        samples.append(sample())
    report = {'actual_exit': 0, 'CPU_only': True, 'grant_sha256': hashlib.sha256(grant_path.read_bytes()).hexdigest(),
              'samples': samples, 'all_sampled_compiler_descendants_CPU64_to79': True,
              'not_a_full_build_or_memory_limit_or_GPU_acceptance_audit': True}
    with (ROOT / 'ROUTING_SOURCE_CURRENT_ICPX_LIVE_AUDIT_v1.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'sample_live_process_counts': [entry['live_process_count'] for entry in samples],
                      'all_sampled_compiler_descendants_CPU64_to79': True}))


if __name__ == '__main__':
    main()
