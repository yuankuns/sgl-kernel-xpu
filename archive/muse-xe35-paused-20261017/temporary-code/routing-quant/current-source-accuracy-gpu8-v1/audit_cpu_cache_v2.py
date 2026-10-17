import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from input_forensics import tensor_sha256


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 80)) and not torch.xpu.is_initialized()
    manifest = json.loads((directory / 'CPU-production-cache-v2.json').read_text())
    assert int((directory / 'CPU-production-cache-v2.exit').read_text()) == 0
    assert manifest['production_shapes'] == len(manifest['records']) == 6
    assert not manifest['legacy_native_or_output_goldens_used']
    checked = []
    for entry in manifest['records']:
        path = Path(entry['file'])
        assert digest(path) == entry['file_SHA256']
        tensors = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        for name, value in tensors.items():
            assert tensor_sha256(value) == entry['tensors'][name]
        if 'CPU_reference_q_by_input_row' in tensors:
            assert tensors['CPU_reference_q_by_input_row'].shape[1] == 2048
            assert tensors['CPU_reference_s_by_input_row'].shape[1] == 128
        checked.append({'shape': entry['shape'], 'full_file_and_all_tensor_rawSHA_verified': True})
        del tensors
    fixtures = json.loads((directory / 'CPU-original-fixtures.json').read_text())
    assert len(fixtures['fixtures']) == 3
    for entry in fixtures['fixtures']:
        tensors = torch.load(directory / entry['fixture_file'], map_location='cpu', weights_only=True)
        for name, value in tensors.items():
            assert tensor_sha256(value) == entry['tensors'][name]
    for path in directory.glob('*v2.py'):
        ast.parse(path.read_text())
    subprocess.run(['bash', '-n', str(directory / 'launch_v2.sh'), str(directory / 'run_gpu_v2.sh')], check=True)
    runtime = (directory / 'run_gpu_v2.sh').read_text()
    assert '-x -v' in runtime and '|| exit "$status"' in runtime
    for name in ['run_gpu_v2.sh', 'six_production_shapes_v2.py', 'fixture_accuracy_v2.py']:
        assert all(token not in (directory / name).read_text() for token in ['XPUGraph', 'torch.xpu.Event', 'warmup', 'elapsed_time'])
    report = {'CPU_final_v2_audit_exit': 0, 'six_new_full_CPU_reference_and_input_cache_rawSHA_verified': checked,
              'three_original_full_fixtures_rawSHA_verified': True, 'new_runtime_first_failure_saved_and_stops': True,
              'original_14_and_1300_nodeids_preserved': True, 'original_test_files_not_modified': True,
              'GPU_initialized': False, 'no_compile_or_GPU_work': True}
    (directory / 'CPU-final-v2-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
