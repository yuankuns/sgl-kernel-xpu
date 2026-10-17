import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'controlled-method-runner-cpu-v1'
sys.path.insert(0, str(OLD))
from protocol import digest, graph_proof, load_plan, read, require, save
from runner import local, scope

VERSION = 'qk-nativegraph-window-paired12288-v1'
ROLES = ('source', 'port')
MARKER = 'QK_DIAG_REPLAY_ONLY'


def hashes():
    return {name: digest(ROOT / name) for name in ('contract.py', 'diagnose.py', 'launch.sh', 'prepare_grant.py')}


def case():
    plan, fields, cases = load_plan()
    value = next(item for item in cases if item['name'] == 'prefill-wave-3req-12288')
    require(value['case']['tokens'] == 12288 and value['group'] == 'actual-config-semantic-synthetic', 'Only original first12288 fixture')
    require(value['parameters']['num_heads_q'] == 128 and value['parameters']['num_heads_k'] == 16
        and value['parameters']['tail_heads'] == 144 and value['parameters']['head_dim'] == 64, 'Original actual model parameters required')
    return plan, fields, value


def unchanged():
    ready = read(OLD / 'READY.json')
    require(digest(OLD / 'READY.json') == '5ed20090ee1e151a1e9198d7eb4ac242fadd07e35152a4a6a2147ee0dbfa3e38', 'Old immutable READY changed')
    require(all(digest(OLD / name) == checksum for name, checksum in ready['runner_files_sha256'].items()), 'Old scripts changed')
    old_grant = ROOT.parent / 'controlled-gpu6-release-v1/main-controlled-diagnostic-grant.json'
    require(digest(old_grant) == 'caa8a67f5ab2fa8e6b79fc1f1df49ffac7883d01845b2a0b9808f8eaccb3d37a', 'Old failed grant changed')
    plan, fields, fixture = case()
    for path, checksum in plan['framework_and_source_file_SHA'].items():
        require(digest(local(path)) == checksum, 'Frozen source/adapter/reference changed')
    return ready


def validate(value, initial):
    scope()
    plan, fields, fixture = case()
    require(value.get('version') == VERSION and value.get('main_authorized') is True, 'New explicit diagnostic grant required')
    require(value.get('accuracy_trace_diagnostic_only') is True and value.get('performance_authorized') is False, 'No performance grant')
    require(value.get('gpu') == 6 and value.get('runtime_cpus') == list(range(32, 48)), 'Exact granted GPU6/CPU32–47')
    require(value.get('roles') == list(ROLES) and value.get('calls_per_graph') == 32, 'Paired roles once, unchanged32 calls')
    require(value.get('case') == fixture and value.get('providers') == plan['providers'], 'Exact original fixture and genuine providers')
    require(value.get('driver_files_sha256') == hashes(), 'New immutable diagnostic code changed')
    require(value.get('core_loaded_stack_sha256'), 'Actual full core stack binding required')
    require(value.get('drain_sha256') and value.get('old_owned_drain_sha256'), 'Fresh and own drain SHA required')
    require(value.get('strict_original_device_kernel_trace_gate_preserved') is True and not value.get('eight_card_claim', False), 'Strict proof/no eightcard claim')
    now = time.time()
    require(value.get('start_epoch', 0) > now + 5 if initial else now >= value.get('start_epoch', math.inf), 'Future epoch/new once-only release')


def replay_trace(full):
    markers = [event for event in full.get('traceEvents', []) if event.get('ph') == 'X' and event.get('name') == MARKER and event.get('dur', 0) > 0]
    require(len(markers) == 1, 'Unique actual replay annotation/timestamps required')
    marker = markers[0]
    begin, end = marker['ts'], marker['ts'] + marker['dur']
    selected = [event for event in full['traceEvents'] if event.get('ph') == 'X' and 'ts' in event and 'dur' in event
        and event['ts'] >= begin and event['ts'] + event['dur'] <= end]
    return {'traceEvents': selected}, {'actual_replay_begin_us': begin, 'actual_replay_end_us': end,
        'actual_events_inside_replay_interval': len(selected), 'capture_events_NOT_substituted_for_replay': True}


def strict(dot, trace):
    replay, window = replay_trace(trace)
    return {**graph_proof(dot, replay, 'captured_reset_adapter'), **window}
