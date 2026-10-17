import json
from pathlib import Path

import torch

from fixture_capture import save_snapshot


assert not torch.xpu.is_initialized()
directory = Path(__file__).resolve().parent / 'CPU-recording-unitcheck'
directory.mkdir(exist_ok=False)
input_values = torch.zeros((1, 64), dtype=torch.bfloat16)
expected_q = torch.zeros((1, 16), dtype=torch.uint8)
expected_s = torch.full((1, 1), 127, dtype=torch.uint8)
separate_q = expected_q.clone()
separate_q[0, 0] = 1
values = {'x': input_values, 'activated': input_values[:, :32].clone(), 'expected_activation': input_values[:, :32].clone(),
          'expected_q': expected_q, 'expected_s': expected_s, 'separate_q': separate_q,
          'separate_s': expected_s.clone(), 'actual_q': expected_q.clone(), 'actual_s': expected_s.clone()}
report = save_snapshot('CPU-only synthetic recorder unitcheck; NOT SOURCE/GPU case', values, directory)
assert not report['independent_full_fixture_zero_gates']['separate_q']['passed']
assert all(report['independent_full_fixture_zero_gates'][name]['passed'] for name in ['separate_s', 'fused_q', 'fused_s'])
saved = torch.load(report['fixture_file'], weights_only=True)
assert set(saved) == set(report['tensor_SHA256'])
assert not torch.xpu.is_initialized()
report['CPU_only_synthetic_unitcheck'] = True
report['SOURCE_tests_or_GPU_executed'] = False
(directory / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'CPU_recording_unitcheck': True, 'GPU_initialized': False, 'SOURCE_tests_or_GPU_executed': False}))
