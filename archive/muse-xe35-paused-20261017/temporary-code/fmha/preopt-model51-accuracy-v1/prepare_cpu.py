import inspect
import json
import os
import sys
import time
from types import SimpleNamespace
from contracts import ROOT, ART, WORKLOAD, BASE, scope_guard, provider, observe_registered, digest, write_json, code_hashes


def main():
    scope_guard()
    if os.environ.get('ONEAPI_DEVICE_SELECTOR') != 'level_zero:cpu' or os.sched_getaffinity(0) != set(range(96, 112)):
        raise RuntimeError('Explicit CPU-only96–111 prep required')
    torch, semantic, reference, framework, plan, bundles, proof = provider()
    assert not torch.xpu.is_initialized()
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    cases = []
    for case in plan['cases']:
        tensors = semantic.fixture(case, bundles[case['bundle']], torch, 'cpu')
        query, key, value, table, lengths, offsets = tensors
        assert list(query.shape) == case['output_shape'] and query.dtype == key.dtype == value.dtype == torch.bfloat16
        assert table.dtype == lengths.dtype == offsets.dtype == torch.int32
        assert table.min().item() >= 0 and table.max().item() < key.shape[0]
        assert lengths.tolist() == case['cache_lengths'] and offsets.tolist() == case['cu_seqlens_q']
        inspect.signature(flash_attn_with_kvcache).bind(q=query, k_cache=key, v_cache=value, page_table=table,
            cache_seqlens=lengths, cu_seqlens_q=offsets, cu_seqlens_k_new=None, max_seqlen_q=case['query_length'],
            softmax_scale=0.125, causal=True, window_size=(case['window'], case['window_right']), softcap=0.0,
            k_descale=None, v_descale=None, return_softmax_lse=False)
        calls = []
        observer, boundary_stop = observe_registered(torch, calls, True)
        try:
            with observer:
                semantic.backend_call(case, tensors, torch, framework)
        except boundary_stop:
            pass
        else:
            raise RuntimeError('CPU registered-op interception not reached')
        assert len(calls) == 1 and calls[0]['CPU_pre_dispatch_only'] and not calls[0]['native_operator_completed']
        assert calls[0]['actual_registered_parameters']['max_seqlen_k'] == case['source_wrapper_inferred_max_seqlen_k']
        record = {'name': case['name'], 'input_sha256': [reference.tensor_digest(tensor, torch) for tensor in tensors],
                  'expected_checked_elements': case['requests'] * case['query_length'] * 128 * 64,
                  'real_CPU_registered_parameters': calls[0]}
        cases.append(record)
        write_json(ROOT / 'CPU_CASES_PARTIAL_v2.json', {'cases': cases, 'GPU_started': False})
        print(json.dumps({'CPU_fixture_and_registered_boundary': case['name'], 'native_work': False}), flush=True)
    unit_case = SimpleNamespace(requests=8, query_length=65, cache_length=65, window=2047)
    query = torch.zeros((8 * 65, 128, 64), dtype=torch.bfloat16)
    key = torch.zeros((16, 64, 16, 64), dtype=torch.bfloat16)
    table = torch.arange(16, dtype=torch.int32).reshape(8, 2)
    lengths = torch.full((8,), 65, dtype=torch.int32)
    offsets = torch.arange(9, dtype=torch.int32) * 65
    unit = reference.full_cpu_reference(unit_case, (query, key, key, table, lengths, offsets, torch.zeros_like(query)), torch)
    assert unit['reference_PASS'] and unit['checked_elements'] == 8 * 65 * 128 * 64
    assert 'range(0, case.query_length, 64)' in inspect.getsource(reference.full_cpu_reference)
    assert 'atol=5e-3, rtol=5e-3' in inspect.getsource(reference.full_cpu_reference)
    assert len(cases) == 51 and sum(row['expected_checked_elements'] for row in cases) == 808255488
    assert not torch.xpu.is_initialized()
    record = {'epoch': time.time(), 'actual_exit': 0, 'workload': WORKLOAD, 'source_head': BASE,
              'provider': proof, 'plan_sha256': digest(ART / 'model-config51-v1/plan.json'),
              'fixture_file_sha256': {name: record['sha256'] for name, record in plan['fixture_files'].items()},
              'cases': cases, 'reference_chunk64_CPU_unit': unit, 'driver_files_sha256': code_hashes(),
              'actual_container_CPU_affinity': sorted(os.sched_getaffinity(0)), 'GPU_initialized': False,
              'model51_GPU_precision_not_run': True, 'real_registered_boundary_CPU_only_no_fake_output': True,
              'full_reference_all_808255488_elements_required_on_GPU_run': True}
    write_json(ROOT / 'CPU_PREPARATION.json', record)
    print(json.dumps({'CPU_preparation_actual_exit': 0, 'cases': 51, 'required_full_reference_elements': 808255488, 'GPU_started': False}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        write_json(ROOT / 'CPU_PREPARATION_FAILURE_v2.json', {'epoch': time.time(), 'traceback': traceback.format_exc(), 'GPU_started': False})
        raise
