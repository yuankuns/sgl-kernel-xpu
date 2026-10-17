import hashlib
import json
import os
from pathlib import Path

import torch

from input_forensics import tensor_sha256


directory = Path(__file__).resolve().parent
artifact = directory.parent
assert sorted(os.sched_getaffinity(0)) == list(range(32, 48)) and not torch.xpu.is_initialized()
source = artifact / 'historical-fresh-source-precision-v3'
after = artifact / 'official-after-precision-v4-gpu11-v2'
for name in ['host-launch', 'gpu-runtime', 'source-paired-precision', 'original-40-historical', 'original-40-fresh']:
    assert int((source / f'{name}.exit').read_text()) == 0
for name in ['host-launch', 'gpu-runtime', 'source-after-paired-precision', 'original-40-after', 'CPU-result-audit', 'normal-package-postcheck']:
    assert int((after / f'{name}.exit').read_text()) == 0
records = json.loads((source / 'round-1-fresh.json').read_text())
paired = json.loads((after / 'source-after-paired-precision.json').read_text())
assert len(records) == 5 and paired['strict005_and_full_bitwise_pass']
for record in records:
    tensor = torch.load(record['output_file'], map_location='cpu', weights_only=True, mmap=True)
    assert tensor_sha256(tensor) == record['full_valid_output']
    comparisons = paired['shapes'][record['shape']]['full_element_comparisons']
    assert len(comparisons) == 4
    assert all(entry['original_strict005_pass'] and entry['independent_reference_pass'] and entry['bitwise_different_elements'] == 0 for entry in comparisons.values())
    for oracle in paired['shapes'][record['shape']]['AFTER_rounds']:
        assert oracle['independent_FP64']['max_rows_per_expert'] == 4
    del tensor
report = {'CPU_evidence_complete': True, 'GPU_initialized': False, 'five_shapes': 5,
          'source_original_40_both_providers_passed': True, 'normal_AFTER_original_40_passed': True,
          'strict005_and_all_valid_bits_zero_four_rounds': True,
          'source_baseline': records, 'independent_FP64_scope': 'max_rows4/expert/fullcolumns, not every production row',
          'shared_inputs_scales_expected_raw_SHA256': {record['shape']: record['inputs'] for record in records},
          'saved_original_full_valid_outputs_reverified': True,
          'shared_actual_inputs_materialization': 'first SOURCE performance phase, before any warmup/timing; expected rawSHAs fixed by accepted original SOURCE accuracy fixtures'}
(directory / 'CPU-five-shape-precision-fixtures.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({name: value for name, value in report.items() if name not in ['source_baseline', 'shared_inputs_scales_expected_raw_SHA256']}, indent=2))
