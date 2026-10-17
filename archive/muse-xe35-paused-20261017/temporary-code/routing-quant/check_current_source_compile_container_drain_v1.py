import json
import os
from pathlib import Path
import time


def main():
    directory = Path(__file__).resolve().parent / 'fresh-current-source-icpx-native-v3'
    known = [json.loads((directory / 'container.started.json').read_text())['pid']]
    known += [json.loads(path.read_text())['pid'] for path in directory.glob('*.started.json')
              if path.name not in ['container.started.json', 'supervisor.started.json', 'wrapper.started.json']]
    assert all(not Path('/proc', str(pid)).exists() for pid in known), known
    report = {'epoch': time.time(), 'actual_CPU_drain_exit': 0, 'known_container_builder_and_step_PIDs_absent': sorted(set(known)),
              'native_resource_recursive_tree_drained': json.loads((directory / 'native-build.resource-summary.json').read_text())['drained'],
              'GPU_wrapper_or_lease_used': False, 'CPU_affinity': sorted(os.sched_getaffinity(0))}
    (directory / 'container-build-actual-drain.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
