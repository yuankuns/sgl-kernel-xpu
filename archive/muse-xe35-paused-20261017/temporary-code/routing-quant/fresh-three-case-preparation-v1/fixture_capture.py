import hashlib
import json
import os
from pathlib import Path
import sys

import pytest


SOURCE = Path('/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014/tests/test_per_token_group_quant_mxfp4.py')
SOURCE_SHA = '4656ddb0d3161582c5d0a2015cdf8bd5b2b563e8653e929b7286b7b1e1cd1aab'
NODES = {
    'test_fused_swiglu_matches_separate_dtype_surface[dtype0-0.5-2.40625]',
    'test_fused_swiglu_cpu_reference[256-dtype0]',
    'test_fused_swiglu_cpu_reference[8192-dtype0]',
}


def save_snapshot(nodeid, local_values, evidence):
    import torch
    from input_forensics import tensor_sha256

    tensors = {name: value.detach().cpu().contiguous() for name, value in local_values.items() if isinstance(value, torch.Tensor)}
    surface = 'activated' in tensors
    roles = {'separate_q': 'separate_q', 'separate_s': 'separate_s',
             'fused_q': 'actual_q' if surface else 'fused_q', 'fused_s': 'actual_s' if surface else 'fused_s'}
    assert {'expected_q', 'expected_s', *roles.values()}.issubset(tensors)
    original_input = tensors['x'] if surface else tensors['values']
    hidden = original_input.shape[-1] // 2
    tensors['diagnostic_CPU_FP32_activation'] = torch.nn.functional.silu(original_input[:, :hidden].float()) * original_input[:, hidden:].float()
    fingerprints = {name: tensor_sha256(value) for name, value in tensors.items()}
    gates = {}
    for component, field in roles.items():
        reference = tensors['expected_s' if component.endswith('_s') else 'expected_q']
        actual = tensors[field]
        try:
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            passed = True
            failure = None
        except AssertionError as error:
            passed = False
            failure = str(error)
        different = torch.count_nonzero(actual != reference).item() if actual.shape == reference.shape else None
        gates[component] = {'passed': passed, 'rtol': 0, 'atol': 0, 'checked_elements': reference.numel(),
                            'bitwise_different_elements': different, 'failure': failure,
                            'scope': 'independent same original zero assertion on complete saved fixture; not proof original test reached this assertion'}
    key = hashlib.sha256(nodeid.encode()).hexdigest()[:16]
    output = evidence / f'{key}-full-fixture.pt'
    torch.save(tensors, output)
    return {'nodeid': nodeid, 'fixture_file': str(output), 'tensor_SHA256': fingerprints,
            'original_named_locals': list(sorted(name for name in tensors if name != 'diagnostic_CPU_FP32_activation')),
            'independent_full_fixture_zero_gates': gates, 'kernel_rerun': False,
            'reference_or_inputs_modified': False, 'original_assertions_entered': [], 'original_assertion_failure': None}


def assertion_role(frame):
    if frame.f_code.co_name == 'test_fused_swiglu_matches_separate_dtype_surface':
        return {433: 'separate_q', 434: 'separate_s', 435: 'fused_s', 436: 'fused_q', 437: 'fused_q',
                438: 'fused_q', 439: 'fused_q', 440: 'fused_q', 441: 'fused_q'}.get(frame.f_lineno)
    actual = frame.f_locals.get('actual_q')
    prefix = 'separate' if actual is frame.f_locals.get('separate_q') else 'fused' if actual is frame.f_locals.get('fused_q') else 'unresolved'
    return prefix + ('_s' if frame.f_lineno == 463 else '_q')


@pytest.fixture(autouse=True)
def capture_original_fixture(request):
    if request.node.name not in NODES:
        yield
        return
    evidence = Path(os.environ['ROUTING_FRESH_THREE_EVIDENCE_DIR'])
    assert evidence.is_dir() and evidence.parent == Path(__file__).resolve().parents[1]
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    assert sys.gettrace() is None, 'An existing trace hook requires separate review, not replacement'
    record = None
    first_line = 433 if request.node.name.startswith('test_fused_swiglu_matches') else 462

    def local_trace(frame, event, argument):
        nonlocal record
        if event == 'line' and frame.f_lineno == first_line and record is None:
            record = save_snapshot(request.node.nodeid, frame.f_locals, evidence)
            record['source_sha256'] = SOURCE_SHA
            record['source_assertion_anchor'] = first_line
        if record is not None and event == 'line':
            if frame.f_lineno in ([433, 434, 435, 436] if first_line == 433 else [462, 463]):
                record['original_assertions_entered'].append({'line': frame.f_lineno, 'component': assertion_role(frame)})
        if record is not None and event == 'exception' and isinstance(argument[1], AssertionError):
            record['original_assertion_failure'] = {'line': frame.f_lineno, 'component': assertion_role(frame), 'error': str(argument[1])}
        return local_trace

    def global_trace(frame, event, argument):
        if event == 'call' and Path(frame.f_code.co_filename).resolve() == SOURCE and frame.f_code.co_name in {
            'test_fused_swiglu_matches_separate_dtype_surface', 'test_fused_swiglu_cpu_reference'}:
            return local_trace
        return None

    sys.settrace(global_trace)
    try:
        yield
    finally:
        sys.settrace(None)
        key = hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]
        if record is None:
            record = {'nodeid': request.node.nodeid, 'fixture_captured': False, 'all_zero_gates_proven': False}
        else:
            record['fixture_captured'] = True
            record['fused_original_zero_asserts_independently_proven'] = all(record['independent_full_fixture_zero_gates'][name]['passed'] for name in ['fused_q', 'fused_s'])
        (evidence / f'{key}-fixture-report.json').write_text(json.dumps(record, indent=2) + '\n')
