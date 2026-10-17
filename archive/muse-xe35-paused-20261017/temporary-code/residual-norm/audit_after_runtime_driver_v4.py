import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'after-pre-wave-accuracy-v4'
expected = json.loads((OUTPUT / 'driver-sha.json').read_text())
allowed = set(expected['GPU_driver_candidate_libraries_sha256'].values()) | {expected['SYCL_runtime']['sha256']}
processes = []
for directory in Path('/proc').iterdir():
    if not directory.name.isdigit():
        continue
    try:
        arguments = (directory / 'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if any(value.endswith(b'/after_pre_wave_accuracy_v4.py') for value in arguments) and b'accuracy' in arguments:
        processes.append(directory)
assert len(processes) == 1
libraries = {}
for line in (processes[0] / 'maps').read_text().splitlines():
    if any(name in line for name in ('libze_loader.so', 'libze_intel_gpu.so', 'libigc.so', 'libigdgmm.so', 'libsycl.so')):
        path = Path(line.split()[-1]).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest in allowed, str(path)
        libraries[str(path)] = digest
assert any('libze_intel_gpu.so' in name for name in libraries)
assert any('libsycl.so' in name for name in libraries)
record = {'container_provider_pid': int(processes[0].name), 'actual_loaded_driver_libraries_sha256': libraries,
          'all_actual_loaded_driver_libraries_match_grant_bound_CPU_proof': True,
          'CPU_readonly_proc_audit': True, 'GPU_work_started_by_audit': False}
(OUTPUT / 'actual-runtime-driver-maps.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
