import collections
from decimal import Decimal
import html
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / 'controlled-method-runner-cpu-v2'
sys.path.insert(0, str(V2))
from protocol import REPLAY_MARKER, digest, read, require, save


def fields(body):
    result = {}
    for name, value in re.findall(r'<TR><TD[^>]*>(.*?)</TD><TD[^>]*>(.*?)</TD></TR>', body, re.S):
        name = html.unescape(re.sub('<[^>]+>', '', name)).strip().splitlines()[-1].strip()
        result[name] = html.unescape(re.sub('<[^>]+>', '', value)).strip()
    return result


def kind(event):
    name = event['name']
    if 'qk_norm_rope_kernel_xe35' in name:
        return 'QK'
    if 'CopyScalarFunc<long>' in name:
        return 'CAST'
    return 'RESET'


def ns(value, base):
    return int(Decimal(str(value)) * 1000) + base


def audit(case):
    path = V2 / 'runs/stable6-controlled-v2-main-once/proof-phases/r00-s0-port' / case / 'captured_reset_adapter'
    dot = (path / 'capture.dot').read_text()
    nodes = [(name, re.search('<B>(.*?)</B>', body).group(1), fields(body))
        for name, body in re.findall(r'(?ms)^\s*(\w+)\s*\[label=(.*?)\];', dot)]
    edges = re.findall(r'(?m)^\s*(\w+)\s*->\s*(\w+)\s*;', dot)
    require(len(nodes) == 224 and len(edges) == 223, 'Complete real DOT')
    require(edges == [(nodes[index][0], nodes[index + 1][0]) for index in range(223)], 'Exact sequential dumped edge chain')
    trace = read(path / 'full-window.trace.json')
    raw = read(path / 'raw-Kineto-events.json')
    marker = next(event for event in trace['traceEvents'] if event.get('name') == REPLAY_MARKER and event.get('ph') == 'X')
    base = trace['baseTimeNanoseconds']
    markers = [event for event in raw if event['name'] == REPLAY_MARKER]
    require(len(markers) == 1, 'Actual raw CPU parent marker')
    raw_marker = markers[0]
    events = [event for event in trace['traceEvents'] if event.get('ph') == 'X' and 'ts' in event and 'dur' in event
        and marker['ts'] <= event['ts'] and event['ts'] + event['dur'] <= marker['ts'] + marker['dur']
        and event.get('cat') in ('kernel', 'gpu_memcpy')]
    require(len(events) == 96, 'Complete96 actual replay device events')
    records = []
    for event in events:
        matches = [item for item in raw if item.get('device_type') == 'DeviceType.XPU' and item.get('correlation_id') == event['args']['correlation'] and item['name'] == event['name']]
        require(len(matches) == 1, 'Unique actual Kineto correlation/name match')
        native = matches[0]
        require(raw_marker['start_ns'] <= native['start_ns'] < native['end_ns'] <= raw_marker['end_ns'], 'Real device timestamps inside actual parent marker')
        records.append({'kind': kind(event), 'name': event['name'], 'correlation': native['correlation_id'],
            'raw_start_ns': native['start_ns'], 'raw_end_ns': native['end_ns'], 'Chrome_args': event['args'],
            'Chrome_raw_start_delta_ns': ns(event['ts'], base) - native['start_ns'],
            'Chrome_raw_end_delta_ns': ns(Decimal(str(event['ts'])) + Decimal(str(event['dur'])), base) - native['end_ns']})
    records.sort(key=lambda item: item['raw_start_ns'])
    require([item['kind'] for item in records] == ['RESET', 'CAST', 'QK'] * 32, 'Actual starts reset/cast/QK ordering')
    groups = []
    expected_kinds = ['zeCommandListAppendMemoryCopy', 'zeCommandListAppendWaitOnEvents', 'zeCommandListAppendSignalEvent',
        'zeCommandListAppendLaunchKernelWithArguments', 'zeCommandListAppendLaunchKernelWithArguments',
        'zeCommandListAppendWaitOnEvents', 'zeCommandListAppendSignalEvent']
    for index in range(32):
        group = nodes[index * 7:index * 7 + 7]
        require([node[1] for node in group] == expected_kinds, 'Actual seven-node group')
        reset, cast, norm = records[index * 3:index * 3 + 3]
        copies, wait_copy, signal_copy, cast_node, norm_node, wait_norm, signal_norm = [node[2] for node in group]
        queues = [int(item['Chrome_args']['l0 queue'], 16) for item in (reset, cast, norm)]
        commandlists = {int(node[2]['hCommandList'], 16) for node in group}
        require(len(commandlists) == 1 and set(queues) == commandlists, 'Actual Chrome queue and DOT commandlist numeric identity')
        require(wait_copy['phEvents[0]'] == copies['hSignalEvent'] and wait_norm['phEvents[0]'] == norm_node['hSignalEvent'], 'Actual recorded copy/QK signal-to-wait handle match')
        groups.append({'logical_index_zero_based': index, 'actual_DOT_nodes': [node[0] for node in group],
            'actual_recorded_fields': {node[0]: node[2] for node in group},
            'actual_DOT_sequential_edges': edges[index * 7:min(index * 7 + 7, 223)],
            'raw_device_reset_cast_QK': [reset, cast, norm],
            'reset_end_to_cast_start_ns': cast['raw_start_ns'] - reset['raw_end_ns'],
            'cast_end_to_QK_start_ns': norm['raw_start_ns'] - cast['raw_end_ns'],
            'QK_end_to_next_reset_start_ns': records[(index + 1) * 3]['raw_start_ns'] - norm['raw_end_ns'] if index < 31 else None,
            'same_actual_SYCL_queue': len({item['Chrome_args']['sycl queue'] for item in (reset, cast, norm)}) == 1,
            'copy_signal_wait_matches': True, 'QK_signal_wait_matches': True})
    last_reset, last_cast, last_norm = records[-3:]
    negatives = [{key: row[key] for key in ('logical_index_zero_based', 'reset_end_to_cast_start_ns', 'cast_end_to_QK_start_ns', 'QK_end_to_next_reset_start_ns')}
        for row in groups if row['reset_end_to_cast_start_ns'] < 0 or row['cast_end_to_QK_start_ns'] < 0
        or row['QK_end_to_next_reset_start_ns'] is not None and row['QK_end_to_next_reset_start_ns'] < 0]
    return {'actual_CPU_audit_exit': 0, 'DOT_sha256': digest(path / 'capture.dot'),
        'Chrome_trace_sha256': digest(path / 'full-window.trace.json'), 'raw_Kineto_sha256': digest(path / 'raw-Kineto-events.json'),
        'DOT_edge_comment': '// Sequential edges:', 'dumped_edges_NOT_claimed_full_backend_SYCL_DAG': True,
        'actual_nodes': 224, 'actual_sequential_edges': 223, 'all32_actual_groups': groups,
        'actual_Chrome_baseTimeNanoseconds': base, 'Chrome_ts_dur_unit_microseconds_display_unit_NOT_measurement_unit': True,
        'actual_raw_CPU_marker': raw_marker, 'all96_unique_raw_correlation_matches_inside_parent_marker': True,
        'maximum_Chrome_raw_conversion_error_ns': max(abs(item[key]) for item in records for key in ('Chrome_raw_start_delta_ns', 'Chrome_raw_end_delta_ns')),
        'negative_observed_dependency_gaps': negatives, 'all_actual_SYCL_queues': sorted({item['Chrome_args']['sycl queue'] for item in records}),
        'all_actual_L0_commandlists': sorted({node[2]['hCommandList'] for node in nodes}),
        'last_reset_raw_start_ns': last_reset['raw_start_ns'], 'last_reset_raw_end_ns': last_reset['raw_end_ns'],
        'last_QK_raw_start_ns': last_norm['raw_start_ns'], 'last_QK_raw_end_ns': last_norm['raw_end_ns'],
        'last_reset_start_after_last_QK_end': last_reset['raw_start_ns'] > last_norm['raw_end_ns'],
        'last_reset_end_after_last_QK_end': last_reset['raw_end_ns'] > last_norm['raw_end_ns'],
        'last_event_completed_is_QK': max(records, key=lambda item: item['raw_end_ns'])['kind'] == 'QK',
        'raw_pArguments_values_NOT_dereferenced_or_claimed': True, 'no_dead_process_read': True,
        'timing_order_does_NOT_prove_QK_pointee_binding_or_correctness': True}


if __name__ == '__main__':
    require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'Bounded metadataCPU0–15')
    output = {case: audit(case) for case in ('prefill-wave-3req-12288', 'prefill-wave-5req-20480')}
    save(ROOT / 'DEPENDENCY_AUDIT.json', output)
    for case, value in output.items():
        print(case, 'groups32', 'negative_gaps', len(value['negative_observed_dependency_gaps']),
            'last_reset_after_QK', value['last_reset_end_after_last_QK_end'], 'raw_Chrome_error_ns', value['maximum_Chrome_raw_conversion_error_ns'])
