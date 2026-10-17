import hashlib
import json
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'after-pre-wave-accuracy-v4'
assert os.environ['LD_LIBRARY_PATH'].split(':')[0] == '/opt/gfx-deps/profiling_tools/lib'
import torch

assert not torch.xpu.is_initialized()
before = {entry['case']: entry for entry in json.loads((ROOT / 'source-compiler-accuracy-v1/fresh-cases.json').read_text())}
after = json.loads((OUTPUT / 'AFTER-cases.json').read_text())
assert len(before) == 24 and len(after) == 17


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


results = []
for entry in after:
    key = entry['case']
    source_file = ROOT / 'source-compiler-accuracy-v1/fresh-outputs' / (key + '.pt')
    port_file = OUTPUT / 'AFTER-outputs' / (key + '.pt')
    assert digest(source_file) == before[key]['file_sha256'] and digest(port_file) == entry['file_sha256']
    source = torch.load(source_file, map_location='cpu', mmap=True, weights_only=True)
    port = torch.load(port_file, map_location='cpu', mmap=True, weights_only=True)
    assert source.keys() == port.keys()
    outputs = {}
    for name in source:
        assert source[name].shape == port[name].shape == (entry['rows'], 8192)
        different = 0
        for start in range(0, entry['rows'], 512):
            bits = torch.int16 if name == 'norm' else torch.int32
            different += (source[name][start:start + 512].view(bits) != port[name][start:start + 512].view(bits)).sum().item()
        outputs[name] = {'all_actual_elements_compared': source[name].numel(), 'differing_elements': different, 'bitwise_exact': different == 0}
    results.append({'case': key, 'outputs': outputs})
    (OUTPUT / 'failure-partial-comparison.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results[-1]), flush=True)
source_script = ROOT / 'source-before-icpx-bounded-v1/build/src/CMakeFiles/sgl-ops-sycl-FusedNormResidual.dir/sycl/sgl-ops-sycl-FusedNormResidual_gen_FusedNormResidual.cpp.o.Release.cmake'
after_script = ROOT.parent / 'canonical-icpx-pool-v4/build/src/CMakeFiles/sgl-ops-sycl-FusedNormResidual.dir/sycl/xe35/sgl-ops-sycl-FusedNormResidual_gen_FusedNormResidual.cpp.o.Release.cmake'
flags = {}
for name, path in [('SOURCE', source_script), ('AFTER', after_script)]:
    text = path.read_text()
    flags[name] = {'path': str(path), 'sha256': digest(path),
                  'SYCL_compile_flags': re.search(r'set\(SYCL_compile_flags (.*?)\) # list', text).group(1).split(';'),
                  'CMAKE_HOST_FLAGS': re.search(r'set\(CMAKE_HOST_FLAGS (.*?)\)', text).group(1)}
assert set(flags['SOURCE']['SYCL_compile_flags']) == set(flags['AFTER']['SYCL_compile_flags'])
assert '-fsycl-fp64-conv-emu' in flags['AFTER']['SYCL_compile_flags']
(OUTPUT / 'failure-compiler-flags-readonly-audit.json').write_text(json.dumps(flags, indent=2) + '\n')
record = {'completed_saved_cases': 17, 'required_production_cases': 24, 'all17_saved_outputs_actual_bitwise_exact_fresh_SOURCE': all(
    value['bitwise_exact'] for result in results for value in result['outputs'].values()),
    'failure_case': '32768_weight_0_1', 'failure': 'CPU-reference norm assert, 2 mismatches/4194304 in failing 512-row chunk, nan difference',
    'failed_case_full_outputs_saved': False, 'missing_failed_tensor_limits_rootcause': True,
    'SYCL_precision_compile_flag_sets_equal': True, 'fp64_conversion_emulation_present_both': True,
    'host_flags_difference': 'AFTER adds hidden visibility and -O3; no causal claim',
    'rootcause_proven': False, 'no_GPU_work_started': True, 'normal_AFTER_accuracy_pass': False,
    'full24_SOURCE_AFTER_comparison_completed': False, 'original_precision_pytest_executed': False,
    'no_automatic_GPU_retry': True, 'performance_approved': False}
(OUTPUT / 'failure-analysis.json').write_text(json.dumps(record, indent=2) + '\n')
assert not torch.xpu.is_initialized()
print(json.dumps(record, indent=2), flush=True)
