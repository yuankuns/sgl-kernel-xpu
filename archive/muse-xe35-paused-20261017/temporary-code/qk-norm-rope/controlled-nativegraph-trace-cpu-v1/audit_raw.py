import ast
from collections import Counter
import html
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
RUNNER = ROOT.parent / 'controlled-method-runner-cpu-v1'
RUN = RUNNER / 'runs/stable6-main-controlled-v1'
sys.path.insert(0, str(RUNNER))
from protocol import code_hashes, digest, read, require, save


def main():
    require(sorted(os.sched_getaffinity(0)) == list(range(16)) and 'torch' not in sys.modules, 'Pure stdlib CPU0–15 audit')
    terminal = read(ROOT.parent / 'controlled-gpu6-release-v1/TERMINAL.json')
    require(terminal['independent_terminal_CPU_audit_actual_exit'] == 0 and terminal['physical_GPU6_lock_free_actual_exit'] == 0, 'Own prior physical drain0 required')
    require(terminal['actual_runtime_exits'] == {'host_supervisor': 1, 'wrapper': 1, 'first_phase': 1}, 'Original true failure must remain')
    directory = RUN / 'phases/r00-s0-port/prefill-wave-3req-12288/captured_reset_adapter'
    trace = read(directory / 'actual-replay.trace.json')
    events = trace['traceEvents']
    kinds = Counter((event.get('ph'), event.get('cat')) for event in events)
    device = [event for event in events if event.get('ph') == 'X' and event.get('dur', 0) > 0
        and re.search(r'kernel|gpu_memcpy|xpu_memcpy|gpu_memset', event.get('cat', ''), re.I)]
    drivers = [event for event in events if event.get('cat') == 'xpu_driver']
    require(len(events) == 442 and len(drivers) == 192 and not device, 'Strict trace facts mismatch')
    dot = (directory / 'capture.dot').read_text()
    commands = Counter(re.findall(r'<B>([^<]+)</B>', dot))
    names = [html.unescape(name) for name in re.findall(r'>kernelName</TD><TD[^>]*><FONT[^>]*>(.*?)</FONT>', dot)]
    kernel_names = Counter(names)
    require(commands == {'zeCommandListAppendLaunchKernelWithArguments': 64, 'zeCommandListAppendMemoryCopy': 32}, 'Real DOT commands mismatch')
    norm = sum(count for name, count in kernel_names.items() if 'qk_norm_rope_kernel_xe35' in name)
    casts = sum(count for name, count in kernel_names.items() if 'CopyScalarFuncIlE' in name)
    require(norm == casts == 32, 'Actual adapter32 norm and32 position casts required in captured DOT')
    tree = ast.parse((RUNNER / 'runner.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'captured')
    contexts = [node for node in ast.walk(function) if isinstance(node, ast.With)]
    capture_line = next(node.lineno for node in contexts if any(isinstance(item.context_expr, ast.Call) and ast.unparse(item.context_expr.func) == 'torch.xpu.graph' for item in node.items))
    profile_line = next(node.lineno for node in contexts if any(isinstance(item.context_expr, ast.Call) and ast.unparse(item.context_expr.func) == 'torch.profiler.profile' for item in node.items))
    require(capture_line < profile_line, 'Actual old collector window must be correctly identified')
    capability = read(ROOT / 'installed-capabilities.json')
    support = read(ROOT / 'profiling-support.json')
    require(capability['actual_CPU_probe_exit'] == support['actual_CPU_probe_exit'] == 0 and not capability['XPU_initialized'], 'CPU capability probe not0')
    require(capability['kineto_available'] and 'ProfilerActivity.XPU' in capability['supported_activities'], 'Installed XPU/Kineto support missing')
    require(support['actual_PTIdriver_table_has_native_launch_WithArguments'], 'Cannot claim native API absent from installed PTI table')
    for name in ('installed-probe', 'CPU-wrapper', 'profiling-support-v2', 'support-wrapper-v2'):
        require((ROOT / f'{name}.exit').read_text().strip() == '0', 'Actual CPU proof failed')
    grant = read(ROOT.parent / 'controlled-gpu6-release-v1/main-controlled-diagnostic-grant.json')
    require(code_hashes() == grant['runner_files_sha256'] and digest(RUNNER / 'READY.json') == grant['original_READY_sha256'], 'Frozen runner/READY changed')
    evidence = {str(path): digest(path) for path in [directory / 'actual-replay.trace.json', directory / 'capture.dot',
        ROOT / 'installed-capabilities.json', ROOT / 'profiling-support.json', ROOT / 'REPORT.md',
        ROOT.parent / 'controlled-gpu6-release-v1/TERMINAL.json', RUN / 'r00-s0-port.log', RUN / 'r00-s0-port.exit']}
    save(ROOT / 'audit.json', {'actual_independent_CPU_audit_exit': 0, 'CPU_affinity': list(range(16)),
        'original_trace_metadata': {key: value for key, value in trace.items() if key != 'traceEvents'},
        'actual_trace_category_counts': [{'ph': key[0], 'category': key[1], 'count': count} for key, count in kinds.items()],
        'actual_xpu_driver_name_counts': dict(Counter(event['name'] for event in drivers)),
        'actual_all_xpu_driver_event_args': [event.get('args', {}) for event in drivers],
        'actual_ac2g_flow_records': [event for event in events if event.get('cat') == 'ac2g'],
        'actual_DOT_commands': dict(commands), 'actual_DOT_kernel_names_counts': dict(kernel_names),
        'actual_DOT_named_norm_nodes': norm, 'actual_DOT_positions_cast_nodes': casts, 'actual_device_timeline_kernel_memcpy_events': 0,
        'frozen_capture_context_line': capture_line, 'frozen_profiler_context_line': profile_line,
        'collector_active_during_capture_or_instantiate': False, 'collector_active_during_unmeasured_replay': True,
        'installed_XPU_Kineto_capability_present': True, 'native_L0_WithArguments_API_declared_in_installed_PTI_table': True,
        'missing_device_timeline_does_NOT_prove_kernel_not_executed': True,
        'all_XPU_profilers_or_all_native_graph_timeline_unsupported_claim': False,
        'exact_collection_vs_serialization_vs_graph_instrumentation_root_cause': 'NOT_PROVEN_WITH_CURRENT_LOCAL_EVIDENCE',
        'failed_phase_environment_dump_NOT_available_current_CPU_environment_NOT_retrospective_proof': True,
        'actual_new_GPU_retry_compile_or_existing_script_READY_protocol_changes': False,
        'original_firstfail1_and_zero_timing_samples_preserved': True,
        'initial_CPU_only_missing_LD_prepend_tool_exit1_log_preserved_not_GPU_failure_cause': True,
        'evidence_sha256': evidence})
    print('CPU investigation audit0: actual DOT32 norm+32 cast+32 copy; XPU/Kineto and PTI native API declared; replay timeline absent, not proof of no kernel execution; old failure preserved/no GPU retry')


if __name__ == '__main__':
    main()
