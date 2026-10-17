import hashlib
import json
import math
from pathlib import Path
import re
import statistics

SLICE = Path(__file__).resolve().parent
PLAN = SLICE.parent / 'method-decomposition-cpu-v1'
VERSION = 'qk-controlled-method-diagnostic-v2'
METHODS = ('per_call', 'captured_reset_adapter', 'captured_reset_only')
ORDER = ('port', 'source', 'source', 'port')
SAMPLES = 21
REPEATS = 32
WARMUP = 5
PAIRED = SLICE.parent / 'nativegraph-window-paired12288-v1'
REPLAY_MARKER = 'QK_V2_PROOF_REPLAY_ONLY'


def paired_binding():
    require(digest(PAIRED / 'final-evidence-audit.json') == '4e53da89a92da2c8de4b1a0b23efda7873eb2149dda7a750ac71c01d1b30b7bd', 'Reviewed paired12288 audit SHA')
    require(read(PAIRED / 'TERMINAL.json')['actual_exit'] == 0, 'Paired12288 actual0 required')
    audited = read(PAIRED / 'final-evidence-audit.json')['all_run_files_SHA256']
    for role in ('source', 'port'):
        for name in ('strict-replay-proof.json', 'full-window.trace.json', 'capture.dot', 'raw-Kineto-events.json', 'PTI-device-timestamps.json', 'sentinel-full-output.json'):
            path = f'run/{role}/{name}'
            require(digest(PAIRED / path) == audited[path], 'Paired actual raw/proof SHA changed')
        proof = read(PAIRED / 'run' / role / 'strict-replay-proof.json')
        require(proof['actual_replay_device_events'] == 96 and proof['actual_replay_norm_kernels'] == 32, 'Actual paired12288 proof required; not all19 proof')
    return {'terminal_sha256': digest(PAIRED / 'TERMINAL.json'), 'only12288_NOT_all19': True}


def dot_structure(dot, method):
    require(method in METHODS[1:], 'Only actual graph methods')
    nodes = re.findall(r'(?m)^\s*("?[-\w.]+"?)\s*\[([^]]*)\]\s*;?', dot)
    nodes = [(name, label) for name, label in nodes if name not in ('node', 'edge', 'graph')]
    require(len(nodes) >= REPEATS and len({name for name, label in nodes}) == len(nodes), 'Real unique DOT nodes required')
    kernels = [label for name, label in nodes if re.search(r'kernel', label, re.I)]
    copies = [label for name, label in nodes if re.search(r'copy|memcpy|memset', label, re.I)]
    if method == METHODS[1]:
        require(len(nodes) >= 2 * REPEATS and len(kernels) >= REPEATS, 'Real reset+adapter kernel nodes required')
    else:
        require(len(kernels) + len(copies) >= REPEATS, 'Real reset-only copy/kernel nodes required')
    return {'actual_DOT_nodes': len(nodes), 'actual_DOT_kernel_nodes': len(kernels), 'actual_DOT_copy_nodes': len(copies),
        'actual_WithArguments_kernel_nodes': sum('zeCommandListAppendLaunchKernelWithArguments' in label for name, label in nodes),
        'actual_legacy_AppendLaunchKernel_nodes': sum(bool(re.search(r'zeCommandListAppendLaunchKernel(?!WithArguments)', label)) for name, label in nodes),
        'DOT_alone_NOT_device_execution_trace': True}


def replay_proof(dot, trace, method):
    markers = [event for event in trace.get('traceEvents', []) if event.get('ph') == 'X' and event.get('name') == REPLAY_MARKER and event.get('dur', 0) > 0]
    require(len(markers) == 1, 'Unique positive real replay marker required')
    begin, end = markers[0]['ts'], markers[0]['ts'] + markers[0]['dur']
    selected = [event for event in trace['traceEvents'] if event.get('ph') == 'X' and 'ts' in event and 'dur' in event
        and begin <= event['ts'] and event['ts'] + event['dur'] <= end]
    proof = graph_proof(dot, {'traceEvents': selected}, method)
    return {**proof, **dot_structure(dot, method), 'replay_begin_us': begin, 'replay_end_us': end,
        'capture_or_sentinel_events_NOT_substituted': True}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        result = hashlib.sha256()
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def load_plan():
    plan = read(PLAN / 'method-plan.json')
    fields = read(PLAN / 'plan-fields-hashes.json')
    require(digest(PLAN / 'method-plan.json') == fields['plan_SHA'], 'Reviewed CPU plan SHA mismatch')
    require(digest(PLAN / 'shared-19-fixture-plan.json') == fields['fixture_plan_SHA'], 'Reviewed fixture-plan SHA mismatch')
    cases = read(PLAN / 'shared-19-fixture-plan.json')
    require(len(cases) == 19 and len({case['name'] for case in cases}) == 19, 'Exactly all19 unique fixtures required')
    require(sum(case['group'] == 'actual-config-semantic-synthetic' for case in cases) == 17, 'Retain all17 semantic cases')
    require({case['name']: case['fixture']['sha256'] for case in cases} == plan['fixtures']['all_shared_original_inputs_SHA'], 'All fixture SHA bindings required')
    return plan, fields, cases


def code_hashes():
    return {name: digest(SLICE / name) for name in ('protocol.py', 'runner.py', 'graphs.py', 'analyze.py', 'launch.sh')}


def phase_plan(cases):
    names = [case['name'] for case in cases]
    return [{'phase': f'r{round_index:02d}-s{slot}-{role}', 'round': round_index, 'slot': slot, 'role': role,
        'cases': list(reversed(names)) if round_index % 2 else list(names)}
        for round_index in range(3) for slot, role in enumerate(ORDER)]


def method_order(round_index, slot, case_index, sample):
    offset = (round_index + slot + case_index + sample) % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]


def validate_release(release, plan, fields, cases, hashes, now, mask, affinity, initial):
    require(release.get('version') == VERSION and release.get('main_authorized') is True, 'New explicit main controlled-diagnostic grant required')
    require(release.get('release_id') and re.fullmatch(r'[A-Za-z0-9_-]+', release['release_id']), 'Immutable safe release identifier required')
    require(release.get('gpu') == 6 and release.get('stable_gpu_exclusive_owner') == 'QK controlled diagnostic', 'New explicit GPU6 lease required')
    require(str(release['gpu']) == mask, 'Wrapper physical ZE_AFFINITY_MASK mismatch')
    require(release.get('runtime_cpu_owner') == 'QK controlled diagnostic', 'Main must separately assign runtime CPUs')
    require(sorted(affinity) == release.get('runtime_cpus') == list(range(32, 48)), 'Runtime CPUs32–47 required')
    require(release.get('scope') == plan['scope'], 'Fixed B8/ISL4096/OSL16/TP1EP1 BF16 scope required')
    require(release.get('plan_sha256') == fields['plan_SHA'] and release.get('fixture_plan_sha256') == fields['fixture_plan_SHA'], 'Reviewed plan/fixture binding mismatch')
    require(release.get('all_fixture_sha256') == {case['name']: case['fixture']['sha256'] for case in cases}, 'All19 original inputs bound unchanged')
    require(release.get('runner_files_sha256') == hashes, 'New actual runner/analyzer/launcher code binding mismatch')
    require(release.get('providers') == plan['providers'], 'Genuine current full AFTER / original SOURCE provider bindings mismatch')
    require(release.get('drain_evidence_path') and re.fullmatch(r'[a-f0-9]{64}', release.get('drain_evidence_sha256', '')), 'New compiler/package-manager drain proof required')
    stack = release.get('expected_core_loaded_stack_sha256', {})
    require(stack and all(re.fullmatch(r'[a-f0-9]{64}', value) for value in stack.values()), 'Actual shared core loaded-stack SHA required')
    require(all(any(token in path for path in stack) for token in ('libtorch_xpu', 'libsycl', 'libur_adapter', 'libze_')), 'Torch/SYCL/UR/LevelZero stack identity cannot be omitted')
    paired_binding()
    require(release.get('paired12288_terminal_sha256') == digest(PAIRED / 'TERMINAL.json'), 'New paired12288 binding')
    require(release.get('proof_timing_process_separation') is True, 'Fresh process instrumentation isolation required')
    require(re.fullmatch(r'[a-f0-9]{64}', release.get('cpu_READY_sha256', '')), 'CPU-only v2 READY binding required')
    require(not release.get('eight_card_concurrency_claim', False), 'No eight-card concurrency claim')
    require(release.get('start_epoch', 0) > now + 5 if initial else now >= release.get('start_epoch', float('inf')), 'Fresh future epoch for suite; phase cannot start before epoch')


def summarize(raw):
    require(len(raw) == SAMPLES and all(math.isfinite(item) and item > 0 for item in raw), 'Keep exactly21 real positive finite samples')
    median = statistics.median(raw)
    return {'raw_us': raw, 'median_us': median, 'MAD_us': statistics.median(abs(item - median) for item in raw)}


def derived(combined, reset):
    require(len(combined) == len(reset) == SAMPLES, 'Difference requires full independent raw columns')
    values = [after - before for after, before in zip(combined, reset)]
    median = statistics.median(values)
    return {'raw_index_aligned_delta_us': values, 'median_delta_us': median,
        'MAD_delta_us': statistics.median(abs(item - median) for item in values),
        'difference_of_method_medians_us': statistics.median(combined) - statistics.median(reset),
        'diagnostic_only_not_production_kernel_time': True, 'negative_deltas_preserved': True}


def graph_proof(dot, trace, method):
    require(method in METHODS[1:], 'Only real captured graph methods accept graph proof')
    nodes = re.findall(r'(?m)^\s*("?[-\w.]+"?)\s*\[([^]]*)\]\s*;?', dot)
    nodes = [(name, label) for name, label in nodes if name not in ('node', 'edge', 'graph')]
    require(len(nodes) >= REPEATS, 'Actual DOT must show at least32 captured nodes; empty/debugless graph rejected')
    require(len({name for name, label in nodes}) == len(nodes), 'Duplicate node declarations cannot inflate graph proof')
    kernels = [(name, label) for name, label in nodes if 'kernel' in label.lower()]
    copies = [(name, label) for name, label in nodes if re.search('copy|memcpy|memset', label, re.I)]
    events = [event for event in trace.get('traceEvents', []) if event.get('ph') == 'X' and event.get('dur', 0) > 0
        and re.search(r'kernel|gpu_memcpy|xpu_memcpy|gpu_memset', event.get('cat', ''), re.I)]
    norm = [event for event in events if re.search(r'FusedQKNormRopeKernel|(?:fused_qk_norm_rope_xe35|qk_norm_rope_kernel_xe35).*Kernel', event.get('name', ''))]
    require(len(events) >= REPEATS, 'Real replay device trace required, not CPU launch labels alone')
    if method == 'captured_reset_adapter':
        require(len(nodes) >= 2 * REPEATS and len(events) >= 2 * REPEATS and len(kernels) >= REPEATS and len(norm) >= REPEATS,
            'Real32 reset+norm captured nodes/device work and32 named norm kernels required')
    else:
        require(not norm and len(kernels) + len(copies) >= REPEATS, 'Reset-only must contain real copy work, zero norm/RoPE kernels')
    return {'actual_DOT_nodes': len(nodes), 'actual_DOT_kernel_nodes': len(kernels), 'actual_DOT_copy_nodes': len(copies),
        'actual_replay_device_events': len(events), 'actual_replay_norm_kernels': len(norm),
        'DOT_and_real_device_trace_required': True, 'graph_evidence_format_unsupported_is_failure_not_fake_pass': True}
