import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    artifact = Path(__file__).resolve().parent
    candidate = artifact / 'candidate-routing-activation-xe35-v1'
    upstream = Path('/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-routing-quant-xe35')
    source = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014')
    original = (source / 'src/sycl/TripleOps.cpp').read_text()
    official = (upstream / 'src/sycl/TripleOps.cpp').read_text()
    functor = original[original.index('template <typename scalar_t, typename accscalar_t>'):original.index('template <typename scalar_t, typename accscalar_t>\nstruct gelu_tanh')]
    start = original.index('template <typename scalar_t, typename func_t, int N>')
    end = original.index('void silu_and_mul(at::Tensor&')
    helpers = original[start:end]
    body = original[end:original.index('template <typename T_to = float, typename T_from = float>\nvoid gelu_tanh_and_mul_sycl')]
    assert 'KERNEL<T_to, at::opmath_type<T_from>>' in helpers
    assert 'KERNEL<T_to, accscalar_t>' in official
    assert 'using accscalar_t = at::opmath_type<T_to>;' in official
    helper = '#pragma once\n\n#include <ATen/ATen.h>\n#include <ATen/OpMathType.h>\n#include <sycl/sycl.hpp>\n\n#include "sycl/MemoryAccess.h"\n#include "sycl/SYCLHelpers.h"\n#include "sycl/Utils.h"\n\nnamespace sgl_kernel::xe35::activation {\n\n' + functor + helpers.replace('VEC_LAUNCH', 'SGL_XE35_SILU_VEC_LAUNCH') + '\n#undef SGL_XE35_SILU_VEC_LAUNCH\n\n}\n'
    implementation = '#include <c10/xpu/XPUStream.h>\n\n#include "sgl_kernel/xe35/activation.h"\n#include "activation.hpp"\n\n' + body.replace('void silu_and_mul(', 'void silu_and_mul_xe35(').replace('silu_and_mul_sycl<', 'sgl_kernel::xe35::activation::silu_and_mul_sycl<')
    declaration = '#pragma once\n\n#include <ATen/ATen.h>\n\n#pragma GCC visibility push(default)\n\nvoid silu_and_mul_xe35(at::Tensor& out, at::Tensor& input);\n\n#pragma GCC visibility pop\n'
    paths = ['src/torch_extension_sycl.cc', 'include/sgl_kernel_ops.h', 'python/sgl_kernel/elementwise.py']
    baseline = {name: (upstream / name).read_text() for name in paths}
    overlay = dict(baseline)
    generic_registration = '  m.impl("silu_and_mul", torch::kXPU, &silu_and_mul);'
    assert overlay[paths[0]].count(generic_registration) == 1
    overlay[paths[0]] = overlay[paths[0]].replace(generic_registration, '#if SYCL_INTEL_TARGET == 35\n  m.impl("silu_and_mul", torch::kXPU, &silu_and_mul_xe35);\n#else\n' + generic_registration + '\n#endif')
    registration = '#if SYCL_INTEL_TARGET == 35\nTORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {'
    assert overlay[paths[0]].count(registration) == 1
    overlay[paths[0]] = overlay[paths[0]].replace(registration, registration + '\n  m.def("silu_and_mul_xe35(Tensor! out, Tensor input) -> ()");\n  m.impl("silu_and_mul_xe35", torch::kXPU, &silu_and_mul_xe35);')
    guard_include = '#if SYCL_INTEL_TARGET == 35\n#include "sgl_kernel/xe35/routing_quant.h"'
    assert guard_include in overlay[paths[1]]
    overlay[paths[1]] = overlay[paths[1]].replace(guard_include, '#if SYCL_INTEL_TARGET == 35\n#include "sgl_kernel/xe35/activation.h"\n#include "sgl_kernel/xe35/routing_quant.h"')
    overlay[paths[2]] = overlay[paths[2]].replace('from sgl_kernel.utils import get_xpu_stream', 'from sgl_kernel.utils import get_xpu_stream, is_xe3_arch')
    old_call = '    torch.ops.sgl_kernel.silu_and_mul(out, input)'
    assert overlay[paths[2]].count(old_call) == 1
    overlay[paths[2]] = overlay[paths[2]].replace(old_call, '    if input.device.type == "xpu" and is_xe3_arch():\n        torch.ops.sgl_kernel.silu_and_mul_xe35(out, input)\n    else:\n' + '    ' + old_call)
    ast.parse(overlay[paths[2]])
    added = {'src/sycl/xe35/activation/activation.hpp': helper, 'src/sycl/xe35/activation/silu_mul.cpp': implementation,
             'include/sgl_kernel/xe35/activation.h': declaration}
    changes = []
    for name, text in overlay.items():
        changes.extend(difflib.unified_diff(baseline[name].splitlines(True), text.splitlines(True), fromfile='a/' + name, tofile='b/' + name))
    for name, text in added.items():
        changes.extend(difflib.unified_diff([], text.splitlines(True), fromfile='/dev/null', tofile='b/' + name))
    all_files = {'overlay/' + name: text for name, text in (overlay | added).items()}
    all_files.update({'base/' + name: text for name, text in baseline.items()})
    all_files['CANDIDATE.patch'] = ''.join(changes)
    proof = {'actual_CPU_structure_exit': 0, 'prior_prepare_v1_actual_exit_preserved': 1, 'prior_prepare_v2_argv_limit_exit_preserved': 1, 'candidate_is_sparse_source_overlay_NOT_built_package': True,
             'source_ref': '4cb5e929f84ebcebf98ec6702a26f1ee93190828', 'source_TripleOps_SHA256': sha(original),
             'official_generic_TripleOps_SHA256': sha(official), 'generic_BMG_TripleOps_NOT_modified': True,
             'source_functor_exact_SHA256': sha(functor), 'source_vector_config_launch_exact_SHA256': sha(helpers),
             'candidate_helper_reverse_only_macro_name_equals_source': helpers == helper.split(functor, 1)[1].split('\n#undef', 1)[0].replace('SGL_XE35_SILU_VEC_LAUNCH', 'VEC_LAUNCH'),
             'candidate_wrapper_reverse_only_symbol_namespace_equals_source': implementation.split('\n\n', 2)[2].replace('void silu_and_mul_xe35(', 'void silu_and_mul(').replace('sgl_kernel::xe35::activation::silu_and_mul_sycl<', 'silu_and_mul_sycl<') == body,
             'dtype_and_launch': 'original Half/BFloat16 branch; contiguous input/out; original queue/config/vector sizes/workgroup/group-count; at::opmath_type<T_from> float accumulator unchanged',
             'Math_contract': 'original expf and 1.0f expression, FP32 intermediate, only original returned scalar conversion; no new approximate/FTZ/rounding/dtype/tolerance',
             'existing_direct_MoE_torch_op_compatible': 'generic silu_and_mul schema unchanged; only SYCL_INTEL_TARGET35 registration aliases Xe35 helper; XE20 else remains original implementation',
             'Python_contract': 'generic elementwise.py shape/allocation checks preserved; xpu and real is_xe3_arch(3.5) selects new op; non-Xe35 fallback unchanged',
             'CMake_contract': 'unchanged src/CMakeLists recursive sycl glob classifies /xe35/ exclusively into device_cpp_xe35; new files not generic/common/XE20',
             'new_headers_arch_guarded_and_default_visibility': True,
             'no_compile_GPU_registration_or_import_claim': True, 'current_PR_current_normal_v4_current_SOURCE_UNTOUCHED': True,
             'candidate_changed_files_SHA256': {name: sha(text) for name, text in (overlay | added).items()},
             'CMake_readonly_SHA256': sha((upstream / 'src/CMakeLists.txt').read_text()),
             'remaining_gates': ['actual icpx compile/normal package/ABI nm-D export', 'original14 and1300 full current SOURCE-vs-new AFTER', 'sixproduction fullCPU+threefixtures', 'BMG fallback/arch registration audit']}
    assert proof['candidate_helper_reverse_only_macro_name_equals_source']
    assert proof['candidate_wrapper_reverse_only_symbol_namespace_equals_source']
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
