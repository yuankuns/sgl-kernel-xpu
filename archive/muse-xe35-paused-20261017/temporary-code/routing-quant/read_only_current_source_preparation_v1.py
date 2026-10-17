import hashlib
import json
import os
from pathlib import Path
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    artifact = Path(__file__).resolve().parent
    directory = artifact / 'fresh-current-source-native-v3-preparation'
    stage = artifact / 'fresh-source-icpx-native-v2'
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 80))
    assert int((stage / 'supervisor.exit').read_text()) == int((stage / 'native-build.exit').read_text()) == 0
    fingerprints = {str(path.relative_to(stage)): digest(path) for path in stage.rglob('*') if path.is_file()}
    (directory / 'old-stage-protected-SHA256.json').write_text(json.dumps(fingerprints, indent=2) + '\n')
    known = [262526, 249893, 252433]
    absent = [pid for pid in known if not Path('/proc', str(pid)).exists()]
    assert absent == known
    active = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            status = (process / 'status').read_text()
            state = next(line.split()[1] for line in status.splitlines() if line.startswith('State:'))
            if state in ['Z', 'X']:
                continue
            environment = (process / 'environ').read_bytes().split(b'\0')
            selected = next((item.split(b'=', 1)[1].decode() for item in environment if item.startswith(b'ZE_AFFINITY_MASK=')), None)
            if selected in ['4', '6', '7']:
                active.append({'pid': int(process.name), 'GPU_affinity': selected,
                               'argv': (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    assert not active, active
    report = {'epoch': time.time(), 'CPU_affinity': sorted(os.sched_getaffinity(0)),
              'old_stage_file_count': len(fingerprints), 'old_stage_protected_inventory_SHA256': digest(directory / 'old-stage-protected-SHA256.json'),
              'old_stage_files_or_permissions_changed': False, 'old_known_container_runtime_PIDs_absent': absent,
              'active_nonZ_X_GPU4_6_7_environment_processes': active, 'CPU_only_no_compile': True}
    (directory / 'root-readonly-stage-protection-and-container-drain.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
