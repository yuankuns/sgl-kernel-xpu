import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


ART = Path(__file__).resolve().parent
ROOT = ART / 'model-config51-v1'
HELPER = ART / 'launch-v3-icpx/source_only_accuracy.py'
OUTPUT = ART / 'MODEL51_NUMERICAL_QUALITY_CPU_v1.json'


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def instrument_reference(observer):
    tree = ast.parse(HELPER.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'full_cpu_reference')
    reference = copy.deepcopy(original)
    added = ast.parse('audit_chunk(observed, expected, request * case.query_length + indices, torch)').body[0]
    assignments = 0
    for node in ast.walk(reference):
        if isinstance(node, ast.For):
            for index, statement in enumerate(node.body):
                if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'observed' for target in statement.targets):
                    node.body.insert(index + 1, copy.deepcopy(added))
                    assignments += 1
                    break
    assert assignments == 1
    restored = copy.deepcopy(reference)
    for node in ast.walk(restored):
        if isinstance(node, ast.For):
            node.body = [statement for statement in node.body if not (isinstance(statement, ast.Expr) and
                isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name) and statement.value.func.id == 'audit_chunk')]
    original_dump = ast.dump(original, include_attributes=False)
    assert ast.dump(restored, include_attributes=False) == original_dump
    namespace = {'audit_chunk': observer}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[reference], type_ignores=[])), str(HELPER), 'exec'), namespace)
    return namespace['full_cpu_reference'], hashlib.sha256(original_dump.encode()).hexdigest()


def blank_stats():
    return {'elements': 0, 'max_abs': 0.0, 'sum_squared_error_FP64': 0.0, 'sum_signed_error_FP64': 0.0,
            'violations_atol_plus_rtol': 0, 'nonfinite': 0, 'max_tolerance_normalized_error': 0.0}


def update_stats(stats, observed, expected, torch):
    difference = observed - expected
    error = difference.abs()
    allowed = 5e-3 + 5e-3 * expected.abs()
    finite = torch.isfinite(observed) & torch.isfinite(expected)
    stats['elements'] += observed.numel()
    stats['max_abs'] = max(stats['max_abs'], error.max().item())
    stats['sum_squared_error_FP64'] += difference.double().square().sum().item()
    stats['sum_signed_error_FP64'] += difference.double().sum().item()
    stats['violations_atol_plus_rtol'] += ((error > allowed) | ~finite).count_nonzero().item()
    stats['nonfinite'] += (~finite).count_nonzero().item()
    stats['max_tolerance_normalized_error'] = max(stats['max_tolerance_normalized_error'], (error / allowed).max().item())


def finish(stats):
    result = dict(stats)
    result['RMS_error'] = math.sqrt(stats['sum_squared_error_FP64'] / stats['elements'])
    result['mean_signed_error'] = stats['sum_signed_error_FP64'] / stats['elements']
    return result


def main():
    assert not OUTPUT.exists(), 'use a new version, never overwrite quality evidence'
    assert os.sched_getaffinity(0) == set(range(16, 32))
    assert (ART / 'model51-accuracy-suite-v1.exit').read_text().strip() == '0'
    sys.path.insert(0, str(ART / 'launch-v3-icpx'))
    from model_semantic_accuracy import fixture, verify_plan
    plan = verify_plan()
    raws = {provider: json.loads((ROOT / f'{provider}-accuracy.json').read_text()) for provider in ('source', 'port')}
    import torch
    assert not torch.xpu.is_initialized()
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    bundles = {name: torch.load(ROOT / name, map_location='cpu', weights_only=True) for name in plan['fixture_files']}
    result = {'epoch': time.time(), 'CPU_only': True, 'GPU_initialized': False, 'CPU': '16-31', 'threads': 8,
              'plan_sha256': digest(ROOT / 'plan.json'), 'original_reference_helper_sha256': digest(HELPER),
              'raw_source_sha256': digest(ROOT / 'source-accuracy.json'), 'raw_port_sha256': digest(ROOT / 'port-accuracy.json'),
              'reference_math_preserved': 'original helper AST plus observer only; removing observer produces identical AST',
              'coverage': 'all51 full query/request/head/channel to one identical CPU FP32 oracle per case, original chunk64',
              'atol': 5e-3, 'rtol': 5e-3, 'ULP_definition': 'BF16 monotonic ordered integer distance, negative order reversed; +/-zero distance0 but byte difference recorded',
              'ULP_boundary': 'large ULP near zero is not an absolute-error or reference equivalence proof', 'cases': []}
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
    with torch.inference_mode():
        for case, source_record, port_record in zip(plan['cases'], raws['source']['cases'], raws['port']['cases'], strict=True):
            assert case['name'] == source_record['name'] == port_record['name']
            assert source_record['input_sha256'] == port_record['input_sha256']
            tensors = fixture(case, bundles[case['bundle']], torch, 'cpu')
            outputs = {}
            for provider, raw in [('source', source_record), ('port', port_record)]:
                path = ROOT / f'{provider}-full-outputs' / f'{case["name"]}.pt'
                assert digest(path) == raw['saved_full_output_file_sha256']
                outputs[provider] = torch.load(path, map_location='cpu', weights_only=True)
                assert outputs[provider].dtype == torch.bfloat16 and outputs[provider].device.type == 'cpu'
            assert all(tensor.device.type == 'cpu' for tensor in tensors)
            provider_stats = {'source': blank_stats(), 'port': blank_stats()}
            paired = blank_stats()
            comparison = {'port_closer_to_oracle_elements': 0, 'source_closer_to_oracle_elements': 0,
                          'equal_absolute_error_elements': 0, 'BF16_bitwise_mismatched_elements': 0,
                          'max_BF16_ULP': 0, 'sum_BF16_ULP': 0,
                          'BF16_ULP_histogram': {name: 0 for name in ('0', '1', '2', '3-4', '5-8', '9-16', '17-32', '33-64', '65-128', '>128')}}
            def observe(source_chunk, expected, indices, torch_module):
                port_chunk = outputs['port'][indices].float()
                update_stats(provider_stats['source'], source_chunk, expected, torch_module)
                update_stats(provider_stats['port'], port_chunk, expected, torch_module)
                update_stats(paired, port_chunk, source_chunk, torch_module)
                source_error, port_error = (source_chunk - expected).abs(), (port_chunk - expected).abs()
                comparison['port_closer_to_oracle_elements'] += (port_error < source_error).count_nonzero().item()
                comparison['source_closer_to_oracle_elements'] += (source_error < port_error).count_nonzero().item()
                comparison['equal_absolute_error_elements'] += (source_error == port_error).count_nonzero().item()
                source_bits = outputs['source'][indices].view(torch.int16).to(torch.int32) & 0xffff
                port_bits = outputs['port'][indices].view(torch.int16).to(torch.int32) & 0xffff
                comparison['BF16_bitwise_mismatched_elements'] += (source_bits != port_bits).count_nonzero().item()
                source_order = torch_module.where(source_bits & 0x8000 != 0, 0x8000 - (source_bits & 0x7fff), 0x8000 + source_bits)
                port_order = torch_module.where(port_bits & 0x8000 != 0, 0x8000 - (port_bits & 0x7fff), 0x8000 + port_bits)
                ulp = (source_order - port_order).abs()
                comparison['max_BF16_ULP'] = max(comparison['max_BF16_ULP'], ulp.max().item())
                comparison['sum_BF16_ULP'] += ulp.sum().item()
                bounds = [('0', 0, 0), ('1', 1, 1), ('2', 2, 2), ('3-4', 3, 4), ('5-8', 5, 8), ('9-16', 9, 16),
                          ('17-32', 17, 32), ('33-64', 33, 64), ('65-128', 65, 128), ('>128', 129, 65536)]
                for name, lower, upper in bounds:
                    comparison['BF16_ULP_histogram'][name] += ((ulp >= lower) & (ulp <= upper)).count_nonzero().item()
            reference, ast_sha = instrument_reference(observe)
            print(json.dumps({'CPU_reference_starting': case['name']}), flush=True)
            original_report = reference(SimpleNamespace(**case), (*tensors, outputs['source']), torch)
            expected_elements = case['requests'] * case['query_length'] * 128 * 64
            assert provider_stats['source']['elements'] == provider_stats['port']['elements'] == paired['elements'] == expected_elements
            assert original_report['checked_elements'] == expected_elements
            assert provider_stats['source']['max_abs'] == original_report['max_abs']
            assert comparison['BF16_bitwise_mismatched_elements'] == 0 if port_record['full_output_bitwise'] else comparison['BF16_bitwise_mismatched_elements'] > 0
            source_stats, port_stats = finish(provider_stats['source']), finish(provider_stats['port'])
            record = {'name': case['name'], 'source_to_CPU_oracle': source_stats, 'port_to_same_CPU_oracle': port_stats,
                      'SOURCE_to_PORT': finish(paired), **comparison, 'original_helper_full_result': original_report,
                      'RMS_port_over_source': port_stats['RMS_error'] / source_stats['RMS_error'] if source_stats['RMS_error'] else None,
                      'port_RMS_delta_percent': (port_stats['RMS_error'] / source_stats['RMS_error'] - 1) * 100 if source_stats['RMS_error'] else None}
            result['original_reference_function_AST_sha256'] = ast_sha
            result['cases'].append(record)
            OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({'name': case['name'], 'source_max_abs': source_stats['max_abs'], 'source_RMS': source_stats['RMS_error'],
                              'port_max_abs': port_stats['max_abs'], 'port_RMS': port_stats['RMS_error'],
                              'source_violations': source_stats['violations_atol_plus_rtol'], 'port_violations': port_stats['violations_atol_plus_rtol'],
                              'RMS_delta_percent': record['port_RMS_delta_percent'], 'max_BF16_ULP': comparison['max_BF16_ULP']}), flush=True)
            del outputs, tensors
    assert len(result['cases']) == 51 and not torch.xpu.is_initialized()
    totals = {}
    for label in ('source_to_CPU_oracle', 'port_to_same_CPU_oracle', 'SOURCE_to_PORT'):
        stats = blank_stats()
        for case in result['cases']:
            record = case[label]
            for key in ('elements', 'sum_squared_error_FP64', 'sum_signed_error_FP64', 'violations_atol_plus_rtol', 'nonfinite'):
                stats[key] += record[key]
            stats['max_abs'] = max(stats['max_abs'], record['max_abs'])
            stats['max_tolerance_normalized_error'] = max(stats['max_tolerance_normalized_error'], record['max_tolerance_normalized_error'])
        totals[label] = finish(stats)
        assert stats['elements'] == 808255488
    totals['BF16_bitwise_mismatched_elements'] = sum(case['BF16_bitwise_mismatched_elements'] for case in result['cases'])
    totals['max_BF16_ULP'] = max(case['max_BF16_ULP'] for case in result['cases'])
    totals['BF16_ULP_histogram'] = {name: sum(case['BF16_ULP_histogram'][name] for case in result['cases']) for name in result['cases'][0]['BF16_ULP_histogram']}
    for key in ('port_closer_to_oracle_elements', 'source_closer_to_oracle_elements', 'equal_absolute_error_elements'):
        totals[key] = sum(case[key] for case in result['cases'])
    totals['weighted_RMS_port_delta_percent'] = (totals['port_to_same_CPU_oracle']['RMS_error'] / totals['source_to_CPU_oracle']['RMS_error'] - 1) * 100
    totals['cases_port_RMS_greater'] = [case['name'] for case in result['cases'] if case['port_to_same_CPU_oracle']['RMS_error'] > case['source_to_CPU_oracle']['RMS_error']]
    totals['cases_port_RMS_lower'] = [case['name'] for case in result['cases'] if case['port_to_same_CPU_oracle']['RMS_error'] < case['source_to_CPU_oracle']['RMS_error']]
    result['totals'] = totals
    result['quality_claim_boundary'] = 'strict tolerance compliance is not equality; report signed RMS changes/worst cases/ULP, never infer zero degradation merely from within_tol'
    result['GPU_initialized'] = torch.xpu.is_initialized()
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(totals), flush=True)
    if any(totals[label]['violations_atol_plus_rtol'] for label in ('source_to_CPU_oracle', 'port_to_same_CPU_oracle')):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
