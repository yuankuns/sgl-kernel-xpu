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


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def finite_tensors(path):
    tensors = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    result = {}
    for name, tensor in tensors.items():
        nan_count = 0
        inf_count = 0
        for start in range(0, tensor.shape[0], 512):
            chunk = tensor[start:start + 512].float()
            nan_count += torch.isnan(chunk).sum().item()
            inf_count += torch.isinf(chunk).sum().item()
        result[name] = {'dtype': str(tensor.dtype), 'shape': list(tensor.shape), 'stride': list(tensor.stride()),
                        'elements': tensor.numel(), 'nan_count': nan_count, 'inf_count': inf_count,
                        'all_finite': nan_count == inf_count == 0}
    return result


failed_key = '32768_weight_0_1'
source_records = {record['case']: record for record in json.loads((ROOT / 'source-compiler-accuracy-v1/fresh-cases.json').read_text())}
source_failed_file = ROOT / 'source-compiler-accuracy-v1/fresh-outputs' / (failed_key + '.pt')
assert digest(source_failed_file) == source_records[failed_key]['file_sha256']
source_failed = finite_tensors(source_failed_file)
partial = []
for record in json.loads((OUTPUT / 'AFTER-cases.json').read_text()):
    path = OUTPUT / 'AFTER-outputs' / (record['case'] + '.pt')
    assert digest(path) == record['file_sha256']
    outputs = finite_tensors(path)
    partial.append({'case': record['case'], 'actual_saved_outputs': outputs})
    print(json.dumps(partial[-1]), flush=True)
source_cpp = ROOT / 'source-before-icpx-bounded-v1/source/src/sycl/FusedNormResidual.cpp'
port_cpp = ROOT.parent.parent / 'pr/muse-residual-norm-xe35/src/sycl/xe35/FusedNormResidual.cpp'


def body_tokens(path):
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', path.read_text(), flags=re.S)
    text = re.sub(r'^\s*#include[^\n]*', '', text, flags=re.M)
    text = re.sub(r'_xe35\b', '', text)
    return re.findall(r'"(?:\\.|[^"\\])*"|[A-Za-z_]\w*|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[fFuUlL]*|::|->|&&|\|\||<=|>=|==|!=|\+\+|--|\S', text)


source_tokens, port_tokens = body_tokens(source_cpp), body_tokens(port_cpp)
first_difference = next((index for index, pair in enumerate(zip(source_tokens, port_tokens)) if pair[0] != pair[1]), None)
record = {'failure_case': failed_key, 'scope': 'CPU-only forensic evidence; no recreated original failing inputs',
    'failure_flags': {'rows': 32768, 'hidden': 8192, 'post_mode': 'weight', 'post_weight_nonempty': True,
                      'post_norm_input_scale': False, 'round_pre_norm_input': False, 'output_fp32': True,
                      'POST_EPS': 1e-8, 'PRE_EPS': 1e-5},
    'input_declared_dtypes': {'branch': 'BF16', 'stream': 'FP32', 'post': 'BF16', 'pre': 'BF16', 'alpha': 'FP32', 'beta': 'FP32'},
    'input_generation': 'torch.manual_seed42; randn/rand from original immutable runner; actual input tensors/hashes NOT saved',
    'failed_case_inputs_available': False, 'failed_case_AFTER_outputs_available': False,
    'original_failing_CPU_oracle_tensor_available': False, 'original_failure_NaN_side_not_determined': True,
    'SOURCE_corresponding_case_full_saved_outputs': source_failed,
    'SOURCE_corresponding_case_full_CPUref_actual0': True,
    'SOURCE_historical_exact_same_input_as_failed_AFTER_proven': False,
    'SOURCE_same_seed_shape_dtype_recipe_not_raw_input_identity': True,
    'all17_saved_AFTER_outputs_finite': all(value['all_finite'] for entry in partial for value in entry['actual_saved_outputs'].values()),
    'all17_finite_details': partial,
    'source_kernel_body_sha256': digest(source_cpp), 'AFTER_kernel_body_sha256': digest(port_cpp),
    'body_normalization': 'remove include directives/comments; normalize architecture symbol suffix only',
    'normalized_CPP_tokens_equal': source_tokens == port_tokens,
    'normalized_token_counts': [len(source_tokens), len(port_tokens)], 'first_token_difference': first_difference,
    'case_API_domain_valid_by_declared_metadata': True,
    'declared_H8192_launch': {'workgroup': 1024, 'vec': 8, 'iters': 1, 'valid_columns': 8192,
                            'tail_or_partial_vector_slack_expected': False},
    'runtime_provider': 'normal canonical-v4-icpx/site sgl_kernel init; callable real torch.ops _xe35',
    'runtime_maps_712_and_manifest_SHA_verified': True,
    'driver_sampler_failure': 'supplementary CPU collector after provider termination, count !=1; not causal kernel evidence',
    'classification': 'legitimate strict accuracy rejection; cannot yet classify input NaN vs port/runtime vs harness corruption',
    'undefined_slack_explanation_proven': False, 'kernel_math_fix_proposal': None,
    'GPU_or_build_started_by_audit': False, 'rootcause_proven': False}
assert not torch.xpu.is_initialized()
destination = ROOT / 'RESIDUAL_FAILURE_FINITE_FORENSICS_v1.json'
destination.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({name:record[name] for name in ['failure_case','SOURCE_corresponding_case_full_saved_outputs',
      'all17_saved_AFTER_outputs_finite','normalized_CPP_tokens_equal','first_token_difference','classification']},indent=2),flush=True)
