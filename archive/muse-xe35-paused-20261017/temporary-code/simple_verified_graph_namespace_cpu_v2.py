import importlib
import inspect
import json
import os
import runpy
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent


def verify_namespace(namespace):
    names = ('capture_paths', '_capture', 'verify_proof_files', 'validate_node_ratio', 'validate_parameters', 'parse_dot_nodes', 'digest_file', 'time_verified_graph')
    signatures = {}
    for name in names:
        function = namespace.get(name)
        if not callable(function):
            raise RuntimeError('Actual helper namespace function missing: ' + name)
        signatures[name] = str(inspect.signature(function))
    assert namespace['_capture'].__globals__['capture_paths'] is namespace['capture_paths']
    assert namespace['time_verified_graph'].__globals__['_capture'] is namespace['_capture']
    assert namespace['time_verified_graph'].__globals__['verify_proof_files'] is namespace['verify_proof_files']
    assert namespace['verify_proof_files'].__globals__['validate_node_ratio'] is namespace['validate_node_ratio']
    assert namespace['CALLS'] == 128 and namespace['SAMPLES'] == 101 and namespace['WARMUP'] == 5
    paths = {}
    for calls in (1, 128):
        dot_path, proof_path = namespace['capture_paths'](Path('/root/CPU-path-only'), calls)
        assert dot_path.suffix == '.dot' and proof_path.suffix == '.json'
        paths[str(calls)] = {'DOT': str(dot_path), 'proof': str(proof_path)}
    return {'actual_function_signatures': signatures, 'actual_global_bindings_resolved': True, 'pure_Path_results': paths, 'GPU_functions_invoked': False}


def main():
    assert sorted(os.sched_getaffinity(0)) == list(range(80, 96))
    assert not torch.xpu.is_initialized()
    helper = importlib.import_module('simple_verified_graph_v2')
    cases = importlib.import_module('simple_representative_performance_cases_v2')
    assert Path(helper.__file__).resolve() == ROOT / 'simple_verified_graph_v2.py'
    assert Path(cases.__file__).resolve() == ROOT / 'simple_representative_performance_cases_v2.py'
    standard = verify_namespace(vars(helper))
    dynamic = verify_namespace(runpy.run_path(str(ROOT / 'simple_verified_graph_v2.py')))
    dynamic_cases = runpy.run_path(str(ROOT / 'simple_representative_performance_cases_v2.py'))
    for namespace in (vars(cases), dynamic_cases):
        assert namespace['time_verified_graph'] is helper.time_verified_graph
        assert namespace['RepresentativeCase'].measure.__globals__['time_verified_graph'] is helper.time_verified_graph
        assert callable(namespace['representative_cases'])
    cases.verify_frozen_factory()
    contract_path = ROOT / 'simple-verified-graph-v2-cpu-r1.json'
    contract = json.loads(contract_path.read_text())
    assert contract['actual_CPU_contract_exit'] == 0 and contract['tests_run'] == 12
    assert (ROOT / 'simple-verified-graph-v2-cpu-r1.exit').read_text().strip() == '0'
    assert all(helper.digest_file(ROOT / name) == checksum for name, checksum in contract['files_sha256'].items())
    assert not torch.xpu.is_initialized()
    report = {'actual_namespace_CPU_exit': 0, 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'actual_standard_import_namespace': standard, 'actual_runpy_dynamic_namespace': dynamic, 'case_standard_and_dynamic_import_bindings': {'RepresentativeCase.measure': str(inspect.signature(cases.RepresentativeCase.measure)), 'measure_global_time_verified_graph_is_actual_helper_v2': True, 'representative_cases': str(inspect.signature(cases.representative_cases))}, 'CPU_contract_actual_exit': 0, 'CPU_contract_tests': 12, 'CPU_contract_report_sha256': helper.digest_file(contract_path), 'source_files_sha256': {name: helper.digest_file(ROOT / name) for name in ('simple_verified_graph_v2.py', 'simple_representative_performance_cases_v2.py', 'simple_verified_graph_contract_cpu_v2.py', 'simple_verified_graph_namespace_cpu_v2.py')}, 'Torch_provider': torch.__file__, 'Torch_version': torch.__version__, 'XPU_initialized': False, 'XPUGraph_created': False, 'native_graph_debug_dump_or_instantiate_invoked': False, 'GPU_node_proof_passed': False, 'compiler_started': False, 'GPU_started': False, 'existing_helpers_cases_drivers_sources_natives_modified': False}
    target = ROOT / 'simple-verified-graph-v2-namespace-cpu-r1.json'
    with target.open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
