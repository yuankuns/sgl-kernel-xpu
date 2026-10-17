import collections
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / 'controlled-method-runner-cpu-v2'
sys.path.insert(0, str(V2))
from protocol import REPLAY_MARKER, read, require, save


def kind(event):
    name = event['name']
    return 'QK' if 'qk_norm_rope_kernel_xe35' in name else 'CAST' if 'CopyScalarFunc<long>' in name else 'RESET'


result = {}
for case in ('prefill-wave-3req-12288', 'prefill-wave-5req-20480'):
    directory = V2 / 'runs/stable6-controlled-v2-main-once/proof-phases/r00-s0-port' / case / 'captured_reset_adapter'
    trace = read(directory / 'full-window.trace.json')
    marker = next(event for event in trace['traceEvents'] if event.get('ph') == 'X' and event.get('name') == REPLAY_MARKER)
    events = [event for event in trace['traceEvents'] if event.get('ph') == 'X' and 'ts' in event and 'dur' in event
        and marker['ts'] <= event['ts'] and event['ts'] + event['dur'] <= marker['ts'] + marker['dur']
        and event.get('cat') in ('kernel', 'gpu_memcpy')]
    events.sort(key=lambda event: event['ts'])
    require(len(events) == 96, 'Actual complete replay trace')
    sequence = [kind(event) for event in events]
    tables = re.findall(r'<TABLE.*?</TABLE>', (directory / 'capture.dot').read_text(), re.S)
    selected = []
    for table in tables:
        fields = {}
        for name, value in re.findall(r'<TR><TD[^>]*>(.*?)</TD><TD[^>]*>(.*?)</TD></TR>', table, re.S):
            name, value = re.sub('<[^>]+>', '', name).strip(), re.sub('<[^>]+>', '', value).strip()
            if name in ('srcptr', 'dstptr', 'size', 'kernelName', 'groupCounts', 'groupSizes', 'pArguments'):
                fields[name] = value
        if fields:
            selected.append(fields)
    result[case] = {'actual_device_sequence': sequence,
        'device_sequence_is_RESET_CAST_QK_times32': sequence == ['RESET', 'CAST', 'QK'] * 32,
        'last_device_event_to_complete': kind(max(events, key=lambda event: event['ts'] + event['dur'])),
        'per_kind_count': dict(collections.Counter(sequence)),
        'actual_DEVICE_events_with_timestamps_args': events, 'actual_DOT_pointer_and_grid_fields': selected,
        'DOT_does_NOT_expose_failed_QK_pointee_buffer_contents': True,
        'timestamp_diagnostic_only_NOT_performance_samples': True,
        'normal_trace_order_does_NOT_prove_correct_QK_argument_binding': True}
save(ROOT / 'trace-order.json', result)
print('CPU actual trace sequencing/DOT metadata saved; both RESET CAST QK ×32, last completed QK; not rootcause or performance proof')
