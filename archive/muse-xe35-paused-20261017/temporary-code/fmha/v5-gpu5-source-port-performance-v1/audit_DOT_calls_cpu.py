from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import time

from host_scan import snapshot


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw_path = ROOT / 'source-port-r1-B0.json'
    assert digest(raw_path) == 'bab498e81d545e97f063688a11739046477c7aa61e969c2eb975ab8994036384'
    raw = json.loads(raw_path.read_text())
    case = raw['cases'][0]
    graph = ROOT / 'source-port-r1-B0-graphs' / Path(case['actual_graph_debug_dump']).name
    text = graph.read_text()
    assert digest(graph) == case['actual_graph_debug_dump_sha256']
    labels = re.findall(r'(L0_S\d+_C\d+)\s*\[label=(.*?)\];', text, flags=re.S)
    nodes = []
    for identifier, label in labels:
        fields = re.findall(r'<TR><TD[^>]*>([^<]+)</TD><TD[^>]*><FONT[^>]*>(.*?)</FONT></TD></TR>', label, flags=re.S)
        record = {'node': identifier, 'fields': {html.unescape(key): html.unescape(value) for key, value in fields}}
        operation = re.search(r'<B>(.*?)</B>', label)
        record['operation'] = html.unescape(operation.group(1)) if operation else None
        kernel = record['fields'].get('kernelName')
        if kernel:
            record['kernel_demangled'] = subprocess.check_output(['c++filt', kernel], text=True).strip()
        nodes.append(record)
    assert len(nodes) > 0
    forward = case['capture_actual_registered_fwd_nodes']
    assert len(forward) == 8 and all(row['native_operator_completed'] for row in forward)
    assert all(row['active_sequences'] == 3 and row['actual_fwd_maxK_before_private_resolver'] == 0
        and row['guard_evaluation_from_actual_metadata'] == {'optimized': True, 'private_maxK': 4096, 'fallback_original_maxK': None}
        for row in forward)
    assert case['capture_registered_operator_counts']['sgl_kernel.fwd.default'] == 8
    assert case['wholebits_before_timing_PASS'] and case['wholebits_after_timing_PASS']
    assert len(case['samples_ms']) == 21
    output = ROOT / 'source-port-r1-B0-full-outputs' / Path(case['saved_output_file']).name
    assert digest(output) == case['saved_output_file_sha256']
    log = (ROOT / 'source-port-r1-B0.log').read_text().splitlines()
    markers = [line for line in log if 'debug_dump' in line or 'print_graph' in line]
    plan = json.loads((ART / 'model-config51-v1/plan.json').read_text())
    source = json.loads((ART / 'model-config51-v1/source-accuracy.json').read_text())
    source_cases = {row['name']: row for row in source['cases']}
    first, second = plan['cases'][:2]
    assert first['name'] == case['name']
    first_golden, second_golden = source_cases[first['name']], source_cases[second['name']]
    report = {'epoch': time.time(), 'actual_exit': 0, 'CPU_only': True,
        'fresh_owned_host_drain_and_actual_GPU5_flock_free': snapshot(),
        'raw_sha256': digest(raw_path), 'graph_sha256': digest(graph), 'full_saved_output_sha256': digest(output),
        'first_case_full_SOURCE_golden_output_sha256': case['output_sha256'],
        'last_existing_driver_markers': markers,
        'marker_boundary': 'old immutable driver has no begin-case/capture-end/replay boundary stdout markers; raw first-case completed and core capture_end prove a later graph-instantiation crash, exact later case inferred from ordering until core Python locals identify it',
        'actual_DOT_command_node_count': len(nodes),
        'actual_DOT_kernel_counts': dict(Counter(node['fields']['kernelName'] for node in nodes if 'kernelName' in node['fields'])),
        'actual_DOT_nodes': nodes, 'actual_registered_operator_counts': case['capture_registered_operator_counts'],
        'actual_fwd_calls_and_private_routes': forward, 'first_case_wholebits_before_after_PASS': True,
        'actual_completed_samples': len(case['samples_ms']), 'completed_BAAB_rounds': 0,
        'first_two_immutable_model_cases': [first, second],
        'first_two_SOURCE_golden_input_SHA_equal': first_golden['input_sha256'] == second_golden['input_sha256'],
        'first_two_SOURCE_golden_output_SHA_equal': first_golden['output_sha256'] == second_golden['output_sha256'],
        'core_metadata_and_backtrace': {'matched_host_pid': 3303047, 'signal': 11,
            'crash_stage': 'at::xpu::XPUGraphImpl::capture_end/instantiate -> SYCL finalize -> UR/LevelZero graph instantiate -> L0::CommandList::setKernelState',
            'core_sha256': json.loads((ROOT / 'CORE_EXTRACTION_CPU.json').read_text())['core_sha256']},
        'not_device_event_profiler_trace_or_shared_main7_cause_proof': True,
        'no_old_script_ABI_math_tolerance_change_or_GPU_compile_restart': True}
    with (ROOT / 'DOT_CALLS_ROUTES_LIFECYCLE_CPU_v1.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({'actual_exit': 0, 'DOT_nodes': len(nodes), 'kernel_counts': report['actual_DOT_kernel_counts'],
        'fwd_calls': len(forward), 'first_two_inputs_equal': report['first_two_SOURCE_golden_input_SHA_equal'],
        'first_two_outputs_equal': report['first_two_SOURCE_golden_output_SHA_equal']}))


if __name__ == '__main__':
    main()
