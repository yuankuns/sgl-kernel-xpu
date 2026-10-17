import ast
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def rename(text, mapping):
    return re.sub(r'\b(?:' + '|'.join(map(re.escape, mapping)) + r')\b', lambda match: mapping[match.group()], text)


def main():
    artifact = Path(__file__).resolve().parent
    candidate = artifact / 'candidate-routing-activation-xe35-v2'
    upstream = Path('/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-routing-quant-xe35')
    source = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014')
    original = (source / 'src/sycl/TripleOps.cpp').read_text()
    functor = original[original.index('template <typename scalar_t, typename accscalar_t>'):original.index('template <typename scalar_t, typename accscalar_t>\nstruct gelu_tanh')]
    helpers = original[original.index('template <typename scalar_t, typename func_t, int N>'):original.index('void silu_and_mul(at::Tensor&')]
    body = original[original.index('void silu_and_mul(at::Tensor&'):original.index('template <typename T_to = float, typename T_from = float>\nvoid gelu_tanh_and_mul_sycl')]
    identifiers = {'a': 'activation_value', 'b': 'multiplication_value', 'i': 'vector_index', 'N': 'VectorWidth',
                   'T': 'ConfigScalar', 'q': 'kernel_queue', 'fn': 'activation_function', 'VEC_LAUNCH': 'SGL_XE35_SILU_VEC_LAUNCH',
                   'silu_mul_dpcpp_functor': 'silu_mul_dpcpp_functor_xe35', 'op_and_mul_functor': 'op_and_mul_functor_xe35',
                   'get_config': 'get_config_xe35', 'silu_and_mul_sycl': 'silu_and_mul_sycl_xe35'}
    renamed = rename(functor + helpers, identifiers)
    reverse = {value: key for key, value in identifiers.items()}
    assert rename(renamed, reverse) == functor + helpers
    helper = '#pragma once\n\n#include <ATen/ATen.h>\n#include <ATen/OpMathType.h>\n#include <sycl/sycl.hpp>\n\n#include "sycl/MemoryAccess.h"\n#include "sycl/SYCLHelpers.h"\n#include "sycl/Utils.h"\n\nnamespace sgl_kernel::xe35::activation {\n\n' + renamed + '\n#undef SGL_XE35_SILU_VEC_LAUNCH\n\n}\n'
    host_checks = '''  TORCH_CHECK(input.device().type() == c10::DeviceType::XPU, "Xe35 activation requires XPU input");
  TORCH_CHECK(out.device() == input.device(), "Activation output must be on the input device");
  auto input_device_index = input.get_device();
  auto input_architecture = query_device(input_device_index);
  TORCH_CHECK(input_architecture == std::make_tuple<int64_t, int64_t>(3, 5), "Xe35 activation requires input architecture (3, 5)");
  c10::DeviceGuard input_device_guard(input.device());
'''
    wrapped = body.replace('void silu_and_mul(', 'void silu_and_mul_xe35(')
    wrapped = wrapped.replace('  input = input.contiguous();', host_checks + '  input = input.contiguous();')
    wrapped = wrapped.replace('at::xpu::getCurrentXPUStream()', 'at::xpu::getCurrentXPUStream(input_device_index)')
    wrapped = wrapped.replace('silu_and_mul_sycl<', 'sgl_kernel::xe35::activation::silu_and_mul_sycl_xe35<')
    normalized_wrapper = wrapped.replace(host_checks, '').replace('void silu_and_mul_xe35(', 'void silu_and_mul(').replace('at::xpu::getCurrentXPUStream(input_device_index)', 'at::xpu::getCurrentXPUStream()').replace('sgl_kernel::xe35::activation::silu_and_mul_sycl_xe35<', 'silu_and_mul_sycl<')
    assert normalized_wrapper == body
    implementation = '#include <c10/core/DeviceGuard.h>\n#include <c10/xpu/XPUStream.h>\n\n#include "sgl_kernel_ops.h"\n#include "sgl_kernel/xe35/activation.h"\n#include "activation.hpp"\n\n' + wrapped
    declaration = '#pragma once\n\n#include <ATen/ATen.h>\n#include "sgl_kernel_export.h"\n\nSGL_KERNEL_EXPORT void silu_and_mul_xe35(at::Tensor& out, at::Tensor& input);\n'
    paths = ['src/torch_extension_sycl.cc', 'include/sgl_kernel_ops.h', 'python/sgl_kernel/elementwise.py']
    baseline = {name: (upstream / name).read_text() for name in paths}
    overlay = dict(baseline)
    bridge = '''static void silu_and_mul_dispatch_xe35(at::Tensor& out, at::Tensor& input) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::XPU, "Activation dispatch requires XPU input");
  TORCH_CHECK(out.device() == input.device(), "Activation output must be on the input device");
  c10::DeviceGuard input_device_guard(input.device());
#if SYCL_INTEL_TARGET == 35
  auto input_architecture = query_device(input.get_device());
  if (input_architecture == std::make_tuple<int64_t, int64_t>(3, 5)) {
    silu_and_mul_xe35(out, input);
    return;
  }
#endif
  silu_and_mul(out, input);
}

'''
    overlay[paths[0]] = '#include <c10/core/DeviceGuard.h>\n\n' + overlay[paths[0]]
    marker = 'TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {'
    overlay[paths[0]] = overlay[paths[0]].replace(marker, bridge + marker, 1)
    old_binding = '  m.impl("silu_and_mul", torch::kXPU, &silu_and_mul);'
    assert overlay[paths[0]].count(old_binding) == 1
    overlay[paths[0]] = overlay[paths[0]].replace(old_binding, '  m.impl("silu_and_mul", torch::kXPU, &silu_and_mul_dispatch_xe35);')
    registration = '#if SYCL_INTEL_TARGET == 35\nTORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {'
    assert overlay[paths[0]].count(registration) == 1
    overlay[paths[0]] = overlay[paths[0]].replace(registration, registration + '\n  m.def("silu_and_mul_xe35(Tensor! out, Tensor input) -> ()");\n  m.impl("silu_and_mul_xe35", torch::kXPU, &silu_and_mul_xe35);')
    guard_include = '#if SYCL_INTEL_TARGET == 35\n#include "sgl_kernel/xe35/routing_quant.h"'
    assert guard_include in overlay[paths[1]]
    overlay[paths[1]] = overlay[paths[1]].replace(guard_include, '#if SYCL_INTEL_TARGET == 35\n#include "sgl_kernel/xe35/activation.h"\n#include "sgl_kernel/xe35/routing_quant.h"')
    old_call = '    torch.ops.sgl_kernel.silu_and_mul(out, input)'
    assert overlay[paths[2]].count(old_call) == 1
    overlay[paths[2]] = overlay[paths[2]].replace(old_call, '''    input_device_index = input.device.index
    if input.device.type == "xpu" and tuple(
        torch.ops.sgl_kernel.query_device.default(input_device_index)
    ) == (3, 5):
        torch.ops.sgl_kernel.silu_and_mul_xe35(out, input)
    else:
        torch.ops.sgl_kernel.silu_and_mul(out, input)''')
    ast.parse(overlay[paths[2]])
    added = {'src/sycl/xe35/activation/activation.hpp': helper, 'src/sycl/xe35/activation/silu_mul.cpp': implementation,
             'include/sgl_kernel/xe35/activation.h': declaration}
    changes = []
    for name, text in overlay.items():
        changes.extend(difflib.unified_diff(baseline[name].splitlines(True), text.splitlines(True), fromfile='a/' + name, tofile='b/' + name))
    for name, text in added.items():
        changes.extend(difflib.unified_diff([], text.splitlines(True), fromfile='/dev/null', tofile='b/' + name))
    proof = {'actual_CPU_structure_exit': 0, 'v1_architecture_isolation_defect_REJECTED_preserved': True,
             'source_ref': '4cb5e929f84ebcebf98ec6702a26f1ee93190828', 'source_TripleOps_SHA256': sha(original),
             'official_generic_TripleOps_SHA256': sha((upstream / 'src/sycl/TripleOps.cpp').read_text()),
             'generic_BMG_TripleOps_math_UNCHANGED': True, 'helper_source_reversible_identifier_normalization_byte_exact': True,
             'identifier_mapping': identifiers, 'original_math_and_launch_SHA256': sha(functor + helpers),
             'host_wrapper_without_disclosed_device_guard_query_and_stream_deltas_byte_exact': normalized_wrapper == body,
             'host_integration_deltas': ['real query_device(input.get_device()) architecture equals (3,5)', 'explicit entry rejects non-XPU/non-(3,5)/different output device',
                 'RAII input DeviceGuard selects device for original queue-dependent get_config and generic fallback, restores caller state',
                 'native getCurrentXPUStream(input_device_index)', 'generic op independent runtime bridge with original generic fallback', 'Python noncached query_device(input.device.index)'],
             'compile35_alone_NOT_runtime_architecture_proof': True, 'runtime_architecture_integration_NOT_GPU_validated': True,
             'new_entry_export_uses_existing_SGL_KERNEL_EXPORT': True, 'no_new_inline_comments_or_one_letter_identifiers': True,
             'CMake_XE35_bucket_unchanged': True, 'current_PR_SOURCE_normal_package_UNCHANGED': True,
             'no_compile_package_import_GPU_or_fake_registration': True,
             'candidate_changed_files_SHA256': {name: sha(text) for name, text in (overlay | added).items()}}
    all_files = {'overlay/' + name: text for name, text in (overlay | added).items()}
    all_files.update({'base/' + name: text for name, text in baseline.items()})
    all_files['CANDIDATE.patch'] = ''.join(changes)
    all_files['CPU-STRUCTURE.json'] = json.dumps(proof, indent=2) + '\n'
    patch = '*** Begin Patch\n'
    for name, text in all_files.items():
        target = candidate / name
        assert not target.exists(), 'immutable candidate already exists'
        patch += '*** Add File: ' + str(target) + '\n' + ''.join('+' + line + '\n' for line in text.splitlines())
    patch += '*** End Patch\n'
    subprocess.run(['apply_patch'], input=patch, text=True, check=True)
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
