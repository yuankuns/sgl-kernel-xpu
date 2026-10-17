import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Optional

root = Path(__file__).resolve().parent
target = root.parents[1] / 'pr/muse-qk-norm-rope-xe35'
records = []


def function(path, name, namespace):
    node = next(item for item in ast.parse(path.read_text()).body if isinstance(item, ast.FunctionDef) and item.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


for relative in ('tests/_xpu.py', 'benchmark/_bench.py'):
    for available, capability, expected in ((False, (3, 5), None), (True, (2, 0), 'xe20'), (True, (3, 0), None), (True, (3, 5), 'xe35'), (True, (3, 6), None), (True, (4, 0), None)):
        calls = []

        def current_device():
            calls.append('current_device')
            return 0

        def query_device(device):
            calls.append('query_device')
            return capability

        probe_torch = SimpleNamespace(xpu=SimpleNamespace(is_available=lambda: available, current_device=current_device), ops=SimpleNamespace(sgl_kernel=SimpleNamespace(query_device=SimpleNamespace(default=query_device))))
        namespace = {'torch': probe_torch, 'Optional': Optional, 'Arch': SimpleNamespace(XE20=SimpleNamespace(value='xe20'), XE35=SimpleNamespace(value='xe35'))}
        probe = function(target / relative, 'current_arch', namespace)
        assert probe() == expected
        assert calls == (['current_device', 'query_device'] if available else [])
        records.append({'probe': relative, 'xpu_available': available, 'offline_capability': capability, 'arch': expected, 'probe_calls': calls})
    namespace['torch'] = SimpleNamespace()
    calls.clear()
    assert probe() is None and not calls

tests_path = target / 'tests/test_fused_qk_norm_rope.py'
tests = ast.parse(tests_path.read_text())
new_cases = []
for node in tests.body:
    if isinstance(node, ast.FunctionDef) and node.name.endswith('_xe35'):
        markers = [decorator for decorator in node.decorator_list if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and decorator.func.attr == 'arch']
        assert len(markers) == 1 and [argument.value for argument in markers[0].args] == ['xe35']
        new_cases.append(node.name)
base = ast.parse(subprocess.check_output(['git', '-C', str(target), 'show', 'HEAD:tests/test_fused_qk_norm_rope.py'], text=True))
tests.body = [node for node in tests.body if not isinstance(node, ast.FunctionDef) or node.name not in new_cases]
assert ast.dump(tests) == ast.dump(base)
assert len(new_cases) == 2
for relative in ('benchmark/bench_muse_qk_norm_rope.py', 'benchmark/validate_qk_norm_rope_b8.py'):
    module = ast.parse((target / relative).read_text())
    preflight = next(index for index, node in enumerate(module.body) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'require_muse_b8_xe35')
    first_torch = next(index for index, node in enumerate(module.body) if isinstance(node, ast.Import) and any(name.name == 'torch' for name in node.names))
    assert preflight < first_torch
result = {'offline_cpu_only_not_gpu_acceptance': True, 'shared_probe_cases': records, 'new_function_only_arch_markers': new_cases, 'generic_test_module_ast_and_cases_unchanged': True, 'b8_preflight_before_torch_import': True, 'full_build_restarted': False}
(root / 'arch-guard-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print('PASS: strict3.5 shared probes, unavailable-XPU short circuit, new-case-only markers, generic cases retained, B8 preflight first')
