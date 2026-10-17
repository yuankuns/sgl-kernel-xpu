import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parent
target = root.parents[1] / 'pr/muse-qk-norm-rope-xe35'
source = root.parents[2] / 'sgl-kernel-xpu/muse-spark-qk-norm-rope-main-20261014'
names = 'divUp subGroupReduceSum FusedQKNormRopeKernel launchFusedQKNormRopeImpl qk_rope_frequencies QKRopeFrequencyTable qk_rope_cached_frequencies QKRopeFrequencyStorage qk_rope_frequency_storage LaunchConfig make_launch choose_launch partner_lanes_for lanes_per_head prefer_baseline Kernel launch'.split()
records = {}
includes = []
for old, new in [('FusedQKNormRopeFrequency.h', 'FusedQKNormRopeFrequency.h'), ('FusedQKNormRopeXe35.h', 'FusedQKNormRopeKernel.h'), ('FusedQKNormRope.cpp', 'FusedQKNormRope.cpp')]:
    original = (source / 'src/sycl' / old).read_text()
    expected = original
    for name in names:
        expected = re.sub(r'\b' + name + r'\b', name + '_xe35', expected)
    expected = expected.replace('FusedQKNormRopeXe35.h', 'FusedQKNormRopeKernel.h').replace('at::native::xpu::fused_qk_norm_rope_xe35', 'at::native::xpu::qk_norm_rope_kernel_xe35').replace('fused_qk_norm_rope_xe35::', 'qk_norm_rope_kernel_xe35::').replace('SYCL_DISPATCH_FLOATING_TYPES', 'SYCL_DISPATCH_QK_FLOATING_TYPES_XE35')
    if old.endswith('.cpp'):
        expected = expected.replace('#include "Norm.h"', '#include "Norm.h"\n#include "sgl_kernel_export.h"').replace('void fused_qk_norm_rope(', 'SGL_KERNEL_EXPORT void fused_qk_norm_rope_xe35(')
    for header in ('MemoryAccess.h', 'SYCLHelpers.h', 'Utils.h', 'Norm.h'):
        expected = expected.replace(f'#include "{header}"', f'#include "../{header}"')
    path = target / 'src/sycl/xe35' / new
    actual = path.read_text()
    assert expected == actual, path
    records[str(path.relative_to(target))] = {'source': old, 'algorithm_unchanged_after_names_exports_includes': True, 'source_sha256': hashlib.sha256(original.encode()).hexdigest(), 'port_sha256': hashlib.sha256(actual.encode()).hexdigest()}
    for include in re.findall(r'^#include "([^"]+)"', actual, re.M):
        candidates = [path.parent / include, target / 'src' / include, target / 'include' / include]
        if include.startswith('cutlass/'):
            candidates.append(root.parent / 'deps/sycl-tla-7b56/include' / include)
        resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        assert resolved is not None, (path, include)
        includes.append({'source': str(path.relative_to(target)), 'include': include, 'resolved': str(resolved)})
    assert not (target / 'src/sycl' / new).exists() or new == 'FusedQKNormRope.cpp'
for path in list((target / 'benchmark').glob('*muse*.py')) + [target / 'benchmark/validate_qk_norm_rope_b8.py', target / 'tests/test_fused_qk_norm_rope.py']:
    assert '_xe35' not in path.name and 'Xe35' not in path.name
    ast.parse(path.read_text())
old_cpp = subprocess.check_output(['git', '-C', str(target), 'show', 'HEAD:src/sycl/FusedQKNormRope.cpp'], text=True)
new_cpp = (target / 'src/sycl/FusedQKNormRope.cpp').read_text().replace('#include "sgl_kernel_ops.h"\n', '')
new_cpp = re.sub(r'#if SYCL_INTEL_TARGET == 35\n.*?#endif\n', '', new_cpp, flags=re.S)
assert old_cpp == new_cpp
old_tests = subprocess.check_output(['git', '-C', str(target), 'show', 'HEAD:tests/test_fused_qk_norm_rope.py'], text=True)
new_tests = (target / 'tests/test_fused_qk_norm_rope.py').read_text()
start = new_tests.index('@pytest.mark.arch("xe35")\n@pytest.mark.parametrize("tokens", [8, 32768])')
end = new_tests.index('if __name__ == "__main__":', start)
assert new_tests[:start] + new_tests[end:] == old_tests
result = {'cpp_mechanical_equivalence': records, 'quoted_includes': includes, 'existing_generic_cpp_unchanged_outside_guard': True, 'old_test_cases_preserved_byte_for_byte': True, 'cri_sources_only_under_xe35': True, 'python_filenames_generic': True}
(root / 'algorithm-equivalence.json').write_text(json.dumps(result, indent=2) + '\n')
print('PASS: mechanical algorithm equivalence, quoted includes, generic CPP, existing test preservation, filenames')
