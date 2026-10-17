import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


parser = argparse.ArgumentParser()
parser.add_argument('--kind', choices=['source', 'after'], required=True)
options = parser.parse_args()
directory = Path(__file__).resolve().parent
artifact = directory.parent
evidence = artifact / ('fresh-source-full-accuracy-v1' if options.kind == 'source' else 'official-after-accuracy-v4-gpu11-v1')
expected_affinity = list(range(64, 72)) if options.kind == 'source' else list(range(72, 80))
assert sorted(os.sched_getaffinity(0)) == expected_affinity
assert not torch.xpu.is_initialized()
runtime_pid = int((evidence / 'container-runtime.pid').read_text())
assert not Path(f'/proc/{runtime_pid}').exists()
pre_name = 'normal-source-precheck.json' if options.kind == 'source' else 'normal-after-precheck.json'
before = json.loads((evidence / pre_name).read_text())
after = json.loads((directory / f'package-post-{options.kind}.json').read_text())
fields = ['package', 'Torch_provider', 'Torch_version', 'mapped_native_sha256', 'package_files_sha256', 'provenance_sha256'] if options.kind == 'source' else ['package', 'manifest_sha256', 'current_native_count', 'mapped_native_hashes', 'source_digest', 'compiler_policy']
for field in fields:
    assert before[field] == after[field], field
report = {'kind': options.kind, 'CPU_evidence_complete': True, 'package_provider_native_source_unchanged': True,
          'GPU_initialized': False, 'CPU_affinity': expected_affinity, 'container_runtime_PID_absent': runtime_pid}
(directory / f'provider-post-{options.kind}.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
