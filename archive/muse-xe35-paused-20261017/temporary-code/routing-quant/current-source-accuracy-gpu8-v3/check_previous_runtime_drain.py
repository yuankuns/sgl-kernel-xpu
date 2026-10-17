import json
from pathlib import Path
import time


directory = Path(__file__).resolve().parent
previous = directory.parent / 'current-source-accuracy-gpu8-v1'
pid = int((previous / 'container-runtime.pid').read_text())
assert pid == 282219
assert not Path(f'/proc/{pid}').exists()
matching = []
for path in Path('/proc').iterdir():
    if path.name.isdigit():
        try:
            command = (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if 'current-source-accuracy-gpu8-v1/run_gpu_v2.sh' in command or 'current-source-accuracy-gpu8-v1/six_production_shapes_v2.py' in command:
            matching.append({'pid': int(path.name), 'command': command})
assert matching == [], matching
exits = {name: int((previous / name).read_text()) for name in ['host-launch.exit', 'gpu-runtime.exit', 'six-production-shapes.exit']}
assert all(value == 1 for value in exits.values())
report = {'epoch': time.time(), 'container_previous_runtime_PID_absent': pid, 'matching_old_runtime_processes': matching,
          'previous_true_failure_exits': exits, 'previous_original14_executed': 0, 'previous_original1300_executed': 0,
          'actual_CPU_drain_audit_exit': 0, 'no_GPU_retry': True}
(directory / 'previous-container-drain.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
