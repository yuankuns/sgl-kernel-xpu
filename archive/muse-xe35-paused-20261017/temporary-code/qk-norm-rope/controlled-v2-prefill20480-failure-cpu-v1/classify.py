import collections
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / 'controlled-method-runner-cpu-v2'
sys.path.insert(0, str(V2))
from protocol import METHODS, REPLAY_MARKER, digest, load_plan, read, replay_proof, require, save
from runner import bits_sha, local


def classify():
    require(sorted(os.sched_getaffinity(0)) == list(range(32, 48)), 'Full CPU tensors on assignedCPU32–47')
    import torch

    torch.set_num_threads(1)
    require(not torch.xpu.is_initialized(), 'CPU-only classification')
    run = V2 / 'runs/stable6-controlled-v2-main-once'
    directory = run / 'proof-phases/r00-s0-port/prefill-wave-5req-20480/captured_reset_adapter'
    plan, fields, cases = load_plan()
    case = next(case for case in cases if case['name'] == 'prefill-wave-5req-20480')
    inputs = torch.load(local(case['fixture']['path']), map_location='cpu', weights_only=True, mmap=True)
    golden = torch.load(local(case['expected_SOURCE_output']), map_location='cpu', weights_only=True, mmap=True)['actual']
    witness = torch.load(directory / 'full-output.pt', map_location='cpu', weights_only=True, mmap=True)
    actual = witness['actual']
    original = inputs['original']
    require(actual.dtype == original.dtype == golden.dtype == torch.bfloat16 and actual.shape == original.shape == golden.shape == (20480, 18432), 'Actual complete tensor layout')
    identities = {}
    for name, expected in inputs.items():
        identities[name] = {'dtype': str(witness[name].dtype), 'shape': list(witness[name].shape),
            'full_bytes_equal_original_fixture': witness[name].dtype == expected.dtype and witness[name].shape == expected.shape
                and torch.equal(witness[name].view(torch.uint8), expected.view(torch.uint8))}
    require(all(value['full_bytes_equal_original_fixture'] for value in identities.values()), 'Actual saved complete input identity')
    positions = inputs['positions']
    require(positions.dtype == torch.int64 and torch.equal(positions, torch.arange(4096, dtype=torch.int64).repeat(5)), 'Exact authoritative repeated positions')
    gpu_inputs = read(run / 'proof-phases/r00-s0-port/prefill-wave-5req-20480-actual-inputs.json')
    require(gpu_inputs['actual_GPU_input_full_bits_checked'] and gpu_inputs['fixture_sha256'] == case['fixture']['sha256'], 'Beforecapture actual GPU complete input checks')
    require(bits_sha(original) == gpu_inputs['complete_QKVO_uint16_sha256'] and bits_sha(positions) == gpu_inputs['complete_positions_bytes_sha256'], 'Recorded original GPU qkv/positions hashes')
    require(all(bits_sha(inputs[name].to(torch.bfloat16)) == checksum for name, checksum in gpu_inputs['complete_GPU_BF16_weight_uint16_sha256'].items()), 'Actual GPU weight hashes')
    sentinel = torch.load(directory / 'sentinel.pt', map_location='cpu', weights_only=True, mmap=True)
    require(sentinel.shape == actual.shape and torch.isnan(sentinel).all().item(), 'Complete pre-replay allNaN sentinel')
    blocks = {'Q128': (0, 8192), 'K16': (8192, 9216), 'V16': (9216, 10240), 'O_gate128': (10240, 18432)}
    statistics = {name: {'elements': 0, 'NaN': 0, 'positive_Inf': 0, 'negative_Inf': 0,
        'zero': 0, 'bit_differences_to_golden': 0, 'bit_differences_to_input': 0, 'NaN_bits_equal_sentinel': 0,
        'numeric_5e3_vs_BF16_golden_violations_NOT_independent_reference': 0,
        'max_abs_difference_to_golden_finite': 0.0, 'samples': [], 'mismatch_rows_by_request_wave': [0] * 5} for name in blocks}
    complete_unchanged_QK_rows = 0
    for start in range(0, 20480, 256):
        end = min(start + 256, 20480)
        complete_unchanged_QK_rows += int(torch.eq(actual[start:end, :9216].view(torch.uint16), original[start:end, :9216].view(torch.uint16)).all(dim=1).sum().item())
        for name, (begin, stop) in blocks.items():
            stats = statistics[name]
            got = actual[start:end, begin:stop]
            expected = golden[start:end, begin:stop]
            baseline = original[start:end, begin:stop]
            mismatch = got.view(torch.uint16) != expected.view(torch.uint16)
            stats['elements'] += got.numel()
            stats['NaN'] += int(torch.isnan(got).sum().item())
            stats['positive_Inf'] += int(torch.isposinf(got).sum().item())
            stats['negative_Inf'] += int(torch.isneginf(got).sum().item())
            stats['zero'] += int((got == 0).sum().item())
            stats['bit_differences_to_golden'] += int(mismatch.sum().item())
            stats['bit_differences_to_input'] += int((got.view(torch.uint16) != baseline.view(torch.uint16)).sum().item())
            stats['NaN_bits_equal_sentinel'] += int((torch.isnan(got) & (got.view(torch.uint16) == sentinel[start:end, begin:stop].view(torch.uint16))).sum().item())
            finite = torch.isfinite(got) & torch.isfinite(expected)
            difference = (got.float() - expected.float()).abs()
            stats['numeric_5e3_vs_BF16_golden_violations_NOT_independent_reference'] += int(((~finite) | (difference > 0.005 + 0.005 * expected.float().abs())).sum().item())
            if finite.any().item():
                stats['max_abs_difference_to_golden_finite'] = max(stats['max_abs_difference_to_golden_finite'], float(difference[finite].max().item()))
            for row in mismatch.any(dim=1).nonzero().flatten().tolist():
                stats['mismatch_rows_by_request_wave'][(start + row) // 4096] += 1
            if len(stats['samples']) < 12:
                for row, column in mismatch.nonzero()[:12 - len(stats['samples'])].tolist():
                    stats['samples'].append({'row': start + row, 'request_wave': (start + row) // 4096,
                        'position': int(positions[start + row]), 'column': begin + column,
                        'head_within_block': column // 64, 'dimension': column % 64,
                        'actual': repr(float(got[row, column])), 'golden': repr(float(expected[row, column])),
                        'original_input': repr(float(baseline[row, column])),
                        'actual_uint16': int(got.view(torch.uint16)[row, column]), 'golden_uint16': int(expected.view(torch.uint16)[row, column])})
    save(ROOT / 'block-classification.json', {'actual_CPU_classification_exit': 0, 'blocks': statistics,
        'actual_complete_inputs': identities, 'positions': {'dtype': str(positions.dtype), 'repeat_waves': 5,
            'tokens_each_wave': 4096, 'minimum': int(positions.min()), 'maximum': int(positions.max()), 'full_original_identity': True},
        'actual_complete_beforecapture_GPU_input_checks': gpu_inputs,
        'complete_QK_rows_bitwise_equal_input': complete_unchanged_QK_rows,
        'unwritten_NOT_inferred_from_value_equality': True, 'all_complete_sentinel_NaN_verified': True})
    sys.path.insert(0, str(ROOT.parent / 'normal-v4-accuracy-only-v1'))
    from driver import cpu_reference

    reference = {}
    for label, output in (('actual_failed_output', actual), ('original_optimized_SOURCE_golden', golden)):
        violations, elements, maximum = cpu_reference(original, output, positions, inputs['query_weight'].float(), inputs['key_weight'].float())
        reference[label] = {'violations': int(violations), 'QK_elements': int(elements), 'max_abs': float(maximum) if math.isfinite(maximum) else repr(maximum),
            'atol': 0.005, 'rtol': 0.005, 'original_approved_full_CPU_reference': True}
    trace = read(directory / 'full-window.trace.json')
    proof = replay_proof((directory / 'capture.dot').read_text(), trace, METHODS[1])
    marker = next(event for event in trace['traceEvents'] if event.get('name') == REPLAY_MARKER and event.get('ph') == 'X')
    events = [event for event in trace['traceEvents'] if event.get('ph') == 'X' and 'ts' in event and 'dur' in event
        and marker['ts'] <= event['ts'] and event['ts'] + event['dur'] <= marker['ts'] + marker['dur']]
    device = [event for event in events if any(token in event.get('cat', '').lower() for token in ('kernel', 'gpu_memcpy', 'xpu_memcpy', 'gpu_memset'))]
    counts = collections.Counter(event['name'] for event in device)
    classified = {'actual_QK_kernel_events': sum(count for name, count in counts.items() if 'qk_norm_rope_kernel_xe35::Kernel_xe35' in name),
        'actual_positions_long_cast_kernel_events': sum(count for name, count in counts.items() if 'CopyScalarFunc<long>' in name),
        'actual_D2D_reset_memcpy_events': sum(count for name, count in counts.items() if name == 'Memcpy D2D (DEVICE -> DEVICE)')}
    require(classified == {'actual_QK_kernel_events': 32, 'actual_positions_long_cast_kernel_events': 32, 'actual_D2D_reset_memcpy_events': 32}, 'Real replay trace32QK/32cast/32reset')
    raw = {str(path.relative_to(directory)): digest(path) for path in directory.iterdir() if path.is_file()}
    require(digest(local(case['fixture']['path'])) == case['fixture']['sha256'], 'Failed original fixture SHA')
    require(digest(local(case['expected_SOURCE_output'])) == case['expected_SOURCE_output_sha256'], 'Original complete optimized SOURCE golden SHA')
    require(not torch.xpu.is_initialized(), 'CPU-only classification end')
    save(ROOT / 'CLASSIFICATION.json', {'actual_CPU_classification_exit': 0, 'epoch': time.time(), 'case': case,
        'reference': reference, 'exactbits_gate_pass': all(stats['bit_differences_to_golden'] == 0 for stats in statistics.values()),
        'independent_strict_full_reference_pass': reference['actual_failed_output']['violations'] == 0,
        'complete_blocks': statistics, 'complete_input_identity_verified': True,
        'strict_real_replay_trace_gate_CPU_verified': proof, 'actual_replay_classification': classified,
        'actual_device_event_names_and_counts': dict(counts), 'actual_failed_case_files_SHA256': raw,
        'XPU_initialized': False, 'no_GPU_retry_compile_math_tolerance_change': True,
        'failure_graph_pointer_JSON_unavailable_due_to_raise_before_proof_json': True})
    print(json.dumps({'actual_CPU_classification_exit': 0, 'reference': reference,
        'blocks': {name: {key: value for key, value in stats.items() if key != 'samples'} for name, stats in statistics.items()},
        'actual_replay_trace': classified}, indent=2), flush=True)


if __name__ == '__main__':
    classify()
