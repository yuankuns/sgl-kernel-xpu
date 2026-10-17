import ast
import json
import os
import pathlib
import types

root = pathlib.Path(__file__).resolve().parent
target = root.parents[1] / 'pr/muse-nvfp4-woq-xe35'
tests = ast.parse((target / 'tests/test_nvfp4_woq_moe.py').read_text())
skip_function = next(node for node in tests.body if isinstance(node, ast.FunctionDef) and node.name == 'skip_if_no_cri_xe35')
condition = compile(ast.Expression(skip_function.body[0].test), '<arch-guard>', 'eval')
records = []
for available, capability, expected in [(False, None, True), (True, (2, 0), True), (True, (3, 0), True), (True, (3, 5), False)]:
    probes = []
    def probe():
        probes.append(capability)
        return capability == (3, 5)
    namespace = {'torch': types.SimpleNamespace(xpu=types.SimpleNamespace(is_available=lambda: available)), 'is_xe3_arch': probe}
    skipped = eval(condition, namespace)
    assert skipped is expected
    assert bool(probes) is available
    records.append({'available': available, 'capability': capability, 'skipped': skipped, 'arch_probed': bool(probes)})
benchmark = ast.parse((target / 'benchmark/bench_nvfp4_woq_group_gemm.py').read_text())
names = {'validate_serving_scope_xe35', 'require_cri_device_xe35', 'bench_one_xe35', 'bench_config_projection_xe35', 'bench_from_config_xe35'}
definitions = ast.Module(body=[node for node in benchmark.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
probes = []
namespace = {'os': os, 'List': list, 'is_cri_device_xe35': lambda: probes.append('probe') or False}
exec(compile(definitions, '<benchmark-preflight>', 'exec'), namespace)
os.environ['BATCHES'] = '8'
args = types.SimpleNamespace(serving_batch=8, tokens='8,32768', config='unopened')
for name, arguments in [('bench_one_xe35', (1, 64, 64, 64, 1, 1)), ('bench_config_projection_xe35', ('bf16', 'down', 256, 8, [8], 8192, 4096, 1, 1)), ('bench_from_config_xe35', (args,))]:
    probes.clear()
    try:
        namespace[name](*arguments)
    except RuntimeError as error:
        assert '3.5' in str(error)
    else:
        raise AssertionError(name)
    assert probes == ['probe']
    os.environ['BATCHES'] = 'invalid'
    probes.clear()
    try:
        namespace[name](*arguments)
    except ValueError:
        pass
    else:
        raise AssertionError(name)
    assert not probes
    os.environ['BATCHES'] = '8'
report = {'cpu_only': True, 'gpu_initialized': False, 'arch_guard_cases': records, 'benchmark_entrypoints': sorted(names - {'validate_serving_scope_xe35', 'require_cri_device_xe35'}), 'strict_skip_expression': ast.unparse(skip_function.body[0].test), 'test_functions': sum(isinstance(node, ast.FunctionDef) and node.name.startswith('test_') for node in tests.body), 'generic_shared_tests_modified': False}
(root / 'arch-guard-preflight-audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
