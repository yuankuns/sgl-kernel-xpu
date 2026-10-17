import hashlib
import json
import os
from pathlib import Path

import torch


directory = Path(__file__).resolve().parent
previous = json.loads((directory.parent / 'fresh-three-source-diagnostic-v1/CPU-driver-provenance.json').read_text())
assert not torch.xpu.is_initialized() and sorted(os.sched_getaffinity(0)) == list(range(64, 72))
for name, expected in previous['driver_runtime_compiler_SHA256'].items():
    with Path(name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
report = previous | {'CPU_affinity': sorted(os.sched_getaffinity(0)), 'GPU_initialized': False,
                     'unchanged_since_bounded_three_case_diagnostic': True}
(directory / 'CPU-driver-provenance.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
