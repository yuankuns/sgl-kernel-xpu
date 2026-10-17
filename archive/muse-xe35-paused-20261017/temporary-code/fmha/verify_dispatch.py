import importlib.util
import json
from pathlib import Path
import re
import sys


art = Path(__file__).resolve().parent
root = art.parent.parent / 'pr/muse-fmha-xe35'
entry = (root / 'src/sycl/xe35/fmha/entry.cpp').read_text()
upstream = (root / 'src/sycl/flash_attention.cpp').read_text()


def signature(text, name):
    clean = re.sub(r'//[^\n]*|/\*.*?\*/', '', text, flags=re.S)
    match = re.search(r'SGL_KERNEL_EXPORT void ' + name + r'\((.*?)\)\s*[;{]', clean, re.S)
    assert match, name
    parameters = [item.strip() for item in match.group(1).split(',')]
    names = [re.search(r'(\w+)\s*$', item).group(1) for item in parameters]
    types = [re.sub(r'\s+', '', item[:item.rfind(name)]).replace('constfloat', 'float').replace('floatconst', 'float')
             .replace('constint', 'int').replace('intconst', 'int').replace('constbool', 'bool').replace('boolconst', 'bool')
             for item, name in zip(parameters, names)]
    return names, types


names, types = signature(entry, 'mha_fwd')
assert types == signature(upstream, 'mha_fwd')[1]
assert types == signature(entry, 'mha_fwd_upstream_xe35')[1]
fallback = re.search(r'if \(!use_port_xe35\)\s*\{\s*mha_fwd_upstream_xe35\((.*?)\);\s*return;', entry, re.S)
assert fallback
assert [item.strip() for item in fallback.group(1).split(',')] == names
assert 'TORCH_CHECK' not in entry
assert (root / 'src/sycl/xe35/fmha/upstream_adapter.cpp').read_text() == (
    '#define mha_fwd mha_fwd_upstream_xe35\n#include "sycl/flash_attention.cpp"\n#undef mha_fwd\n')
assert '/xe35/fmha/(entry|upstream_adapter)\\\\.cpp$' in (root / 'src/CMakeLists.txt').read_text()


class TensorMetadata:
    def __init__(self, shape, dtype='bf16', contiguous=True):
        self.shape = shape
        self.dtype = dtype
        self.contiguous = contiguous

    def dim(self):
        return len(self.shape)

    def size(self, axis):
        return self.shape[axis]

    def sizes(self):
        return self.shape

    def scalar_type(self):
        return self.dtype

    def is_contiguous(self):
        return self.contiguous


class OptionalMetadata:
    def __init__(self, present):
        self.present = present

    def has_value(self):
        return self.present


def expression(name):
    match = re.search(r'const bool ' + name + r'\s*=\s*(.*?);', entry, re.S)
    assert match
    text = match.group(1).replace('at::kBFloat16', "'bf16'").replace('0.0f', '0.0')
    return '(' + re.sub(r'!(?!=)', ' not ', text.replace('&&', ' and ').replace('||', ' or ')).strip() + ')'


shape_expression = expression('production_shape_xe35')
port_expression = expression('use_port_xe35')


def environment(case):
    shape = (case.requests * case.query_length, 128, 64)
    cache = (case.requests * ((case.cache_length + 63) // 64), 64, 16, 64)
    result = {'batch_xe35': case.requests, 'max_seqlen_q': case.query_length, 'max_seqlen_k': case.cache_length,
              'q': TensorMetadata(shape), 'k': TensorMetadata(cache), 'v': TensorMetadata(cache),
              'out': TensorMetadata(shape), 'is_causal': True, 'window_size_left': case.window,
              'window_size_right': 0, 'softcap': 0.0, 'num_kv_splits': 0, 'sm_margin': 0}
    for name in ('softmax_lse', 'q_v_', 'kv_batch_idx_', 'leftpad_k_', 'rotary_cos_', 'rotary_sin_',
                 'seqlens_rotary_', 'scheduler_metadata_', 'rel_bias_', 'bidirectional_block_ids_',
                 'sinks_', 'q_descale_', 'k_descale_', 'v_descale_'):
        result[name] = OptionalMetadata(False)
    result['page_table'] = OptionalMetadata(True)
    return result


def select_port(values):
    values = dict(values)
    values['production_shape_xe35'] = eval(shape_expression, {'__builtins__': {}}, values)
    return eval(port_expression, {'__builtins__': {}}, values)


spec = importlib.util.spec_from_file_location('fmha_dispatch_cpu_cases', root / 'benchmark/bench_muse_fmha.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
cases = list(module.production_cases_xe35())
assert len(cases) == 38
assert all(select_port(environment(case)) for case in cases)
base = environment(cases[-1])
checks = []
for dimension in (72, 80, 96, 128, 192, 256, 512):
    values = dict(base, q=TensorMetadata((8, 128, dimension)))
    assert not select_port(values)
    checks.append(f'nonproduction_hd_{dimension}')
for name in ('softmax_lse', 'q_v_', 'kv_batch_idx_', 'leftpad_k_', 'rotary_cos_', 'rotary_sin_',
             'seqlens_rotary_', 'scheduler_metadata_', 'rel_bias_', 'bidirectional_block_ids_',
             'sinks_', 'q_descale_', 'k_descale_', 'v_descale_'):
    assert not select_port(dict(base, **{name: OptionalMetadata(True)}))
    checks.append(name)
for dtype in ('fp16', 'fp8_e4m3', 'fp8_e5m2'):
    assert not select_port(dict(base, q=TensorMetadata((8, 128, 64), dtype)))
    checks.append(dtype)
for name in ('q', 'k', 'v', 'out'):
    assert not select_port(dict(base, **{name: TensorMetadata(base[name].shape, contiguous=False)}))
    checks.append(name + '_strided')
assert not select_port(dict(base, v=TensorMetadata((520, 64, 16, 128))))
assert not select_port(dict(base, page_table=OptionalMetadata(False)))
assert not select_port(dict(base, q=TensorMetadata((7, 128, 64))))
assert not select_port(dict(base, num_kv_splits=4))
assert not select_port(dict(base, softcap=1.0))
assert select_port(dict(base, window_size_right=-1))
report = {'status': 'PASS', 'kind': 'CPU source-derived predicate and ABI/forward static checks; not GPU or compile proof',
          'upstream_public_abi_parameter_count': len(names), 'full_argument_forward_order': names,
          'production_cases_select_unchanged_port': len(cases), 'fallback_boundaries': checks,
          'generic_source_body_untouched': True, 'no_operator_registration_or_gpu_execution': True}
(art / 'dispatch-cpu-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
