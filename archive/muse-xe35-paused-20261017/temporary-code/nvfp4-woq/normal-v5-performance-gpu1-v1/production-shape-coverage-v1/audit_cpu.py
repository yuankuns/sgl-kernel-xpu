import ast
import hashlib
import json
import os
import sys
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
PERFORMANCE = DIRECTORY.parent
ARTIFACT = PERFORMANCE.parent
OLD = ARTIFACT / 'normal-after-v5-icpx-accuracy-v4'
sys.path.insert(0, str(PERFORMANCE))
import policy


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    assert os.environ['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:cpu'
    assert set(os.sched_getaffinity(0)) == set(range(80, 96))
    import torch
    assert not torch.xpu.is_initialized()
    original_audit = json.loads((OLD / 'independent-completed-CPU-audit-v1.json').read_text())
    ready = json.loads((OLD / 'AFTER_ACCURACY_READY.json').read_text())
    assert original_audit['actual_independent_CPU_audit_exit'] == ready['host_supervisor_actual_exit'] == 0
    assert ready['production_records'] == 16 and ready['all_valid_outputs_and_prep_c_sf_scales_bitwise']
    expected = policy.PRODUCTION
    rows = []
    seals = {}
    first = {}
    for phase, label in enumerate(policy.ORDER):
        path = OLD / 'results/production' / f'round0-phase{phase}-{label}.json'
        relative = path.relative_to(OLD).as_posix()
        sha = digest(path)
        assert sha == ready['raw_sha256'][relative]
        seals[relative] = sha
        data = json.loads(path.read_text())
        assert [tuple(result['shape']) for result in data['results']] == expected
        assert (OLD / f'production-p{phase}-{label}.exit').read_text().strip() == '0'
        for index, result in enumerate(data['results']):
            experts, routed, channels, inner = result['shape']
            assert result['mode'] == 'bf16' and len(result['counts']) == experts and sum(result['counts']) == routed
            assert result['output']['all_elements_checked'] and result['output']['exact'] and result['output']['max_abs'] == 0
            assert all(result['scales'][name]['all_elements_checked'] and result['scales'][name]['exact'] for name in ('c', 'sf'))
            assert result['samples_us'] == []
            identity = {'counts': result['counts'], 'inputs': result['inputs'], 'output_sha256': result['output']['sha256'], 'scale_sha256': {name: result['scales'][name]['sha256'] for name in ('c', 'sf')}}
            assert first.setdefault(index, identity) == identity
            rows.append({'old_phase': phase, 'old_label': label, 'actual_shape_E_totalR_N_K': result['shape'], 'new_production_index': index, 'same_shape': True, 'active_experts': sum(count > 0 for count in result['counts']), 'rows_per_active_expert_min': min(count for count in result['counts'] if count), 'rows_per_active_expert_max': max(result['counts']), 'independent_reference_coverage': result['independent_reference_coverage'], 'full_valid_output_and_c_sf_bits': True, 'old_actual_phase_exit': 0, 'identity': identity})
    dtype_evidence = []
    for experts, routed, channels, inner in expected:
        chunk = OLD / 'results/production' / f'E{experts}-M{routed}-N{channels}-K{inner}/bf16/output-0.pt'
        actual = torch.load(chunk, map_location='cpu', weights_only=True, mmap=True)
        assert actual.dtype == torch.float32 and actual.shape[1] == channels
        dtype_evidence.append({'actual_old_cache_chunk': str(chunk), 'chunk_sha256': digest(chunk), 'chunk_dtype': str(actual.dtype), 'chunk_shape': list(actual.shape), 'full_XPU_output_bytes': routed * channels * 4, 'full_XPU_output_gib': routed * channels * 4 / 2**30, 'BF16_activation_bytes': routed * inner * 2, 'BF16_activation_gib': routed * inner * 2 / 2**30, 'output_CPU_copy_strategy': 'split1024rows, max32MiB per output chunk, not entire8GiB output copied to CPU at once'})
    old_tree = ast.parse((OLD / 'run_accuracy_case.py').read_text())
    new_tree = ast.parse((PERFORMANCE / 'run_case.py').read_text())
    old_output = [node for node in ast.walk(old_tree) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'output' for target in node.targets)]
    new_output = [node for node in ast.walk(new_tree) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'output' for target in node.targets)]
    assert len(old_output) == len(new_output) == 1 and ast.dump(old_output[0]) == ast.dump(new_output[0])
    assert not torch.xpu.is_initialized()
    result = {'actual_exit': 0, 'old_actual_precision_records': 16, 'new_production_cases': 4, 'all_four_production_shapes_exactly_previously_covered': True, 'production_expert_or_routed_row_scope_expanded': False, 'shape_order': 'E,total_routed_rows,N,K; R is not per-expert rows or serving batch', 'records': rows, 'actual_old_output_dtype_CPU_evidence': dtype_evidence, 'reference_scope': 'original CPU FP32 phases0/1 max3rows/active expert; largeprefill768rows directly referenced and261376rows not independently oracled; whole valid output/c_sf SOURCE bits all16records, not FP64', 'dtype_correction': 'BF16 means activation/compute path; actual original and new output tensor FP32. R262144/N8192 output is8GiB XPU, not4GiB BF16 output. K8192 BF16 activations are4GiB XPU', 'RSS_scope': 'old reported2.64GiB process RSS is CPU/host memory and excludes XPU VRAM; streaming output32MiB chunks explains why8GiB GPU output need not yield8GiB RSS', 'resource_policy': 'no live/new cap changes; existing16GiB singleprocess/24GiB tree/minAvail96 remains pending actual newjob sampling, no blind expansion or automatic retry', 'representatives': {'shapes': policy.REPRESENTATIVE, 'source': 'four original default NVWOQ MXFP8 benchmark shapes; Mconverted from per-expert to totalR', 'part_of_old16production_records': False, 'new_exact_shape_source_after_accuracy_claimed': False, 'must_not_infer_representative_precision_from_production_READY': True}, 'formal_performance_grant_created': False, 'GPU1_started': False, 'GPU_started_by_this_audit': False, 'compiler_started': False, 'XPU_initialized': False, 'old_raw_seals': seals, 'new_runner_SHA256': digest(PERFORMANCE / 'run_case.py'), 'new_policy_SHA256': digest(PERFORMANCE / 'policy.py'), 'old_independent_audit_SHA256': digest(OLD / 'independent-completed-CPU-audit-v1.json'), 'old_READY_SHA256': digest(OLD / 'AFTER_ACCURACY_READY.json')}
    with (DIRECTORY / 'PRODUCTION_SHAPE_COVERAGE.json').open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'four_production_shapes_exactly_covered': True, 'old_records': 16, 'scope_expanded': False, 'actual_output_dtype': 'FP32', 'performance_grant_created': False}))


if __name__ == '__main__':
    main()
