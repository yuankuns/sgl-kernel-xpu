import argparse
import json
import os
from pathlib import Path
import runpy
import time

from check_official import canonical_preflight, digest
from check_preopt import verify_loaded
from settings import ART, CANONICAL_VERSION, STAGE
from source_only_accuracy import PR, full_cpu_reference, tensor_digest


WORKLOAD = {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1, 'dtype': 'BF16'}
MANIFEST_SHA = '250e49cf59d4e2852383fbd54c8bac377546857d56304184a40cde9e9b5621d1'


def preflight():
    bench = runpy.run_path(str(PR / 'benchmark/bench_muse_fmha.py'))
    _, cases = bench['arguments_xe35'](['--guard-only', '--mode', 'accuracy', '--reference', 'full'])
    assert len(cases) == 38
    assert CANONICAL_VERSION == 'canonical-v4-icpx'
    manifest, manifest_file, package, required = canonical_preflight()
    assert digest(manifest_file) == MANIFEST_SHA
    for label in ('official-import-canonical-v4-icpx', 'matched-native-icpx-resource-v4-supervisor',
                  'matched-native-icpx-resource-v4-normal-package-cpu', 'source-only-resource-v4-all-provider-cpu',
                  'source-only-gpu5-suite-v1', 'source-only-gpu5-summary-v1'):
        assert (ART / f'{label}.exit').read_text().strip() == '0', label
    readiness = json.loads((STAGE / 'normal-import-readiness.json').read_text())
    assert readiness['canonical_manifest_sha256'] == MANIFEST_SHA
    assert readiness['source_digest'] == manifest['source']['source_digest']
    assert readiness['gpu_initialized'] is False and readiness['diagnostic_AFTER_bridge'] is False
    assert len(readiness['loaded_libraries']) == len(manifest['binaries']) == 735
    summary = json.loads((ART / 'matched-native-icpx-v1/source-only-gpu5-accuracy-v1-summary.json').read_text())
    assert summary['GPU_lock_released'] is True and summary['live_runtime_processes'] == []
    assert summary['historical_to_fresh_optimized_full_output_bitwise_cases'] == 38
    assert summary['historical_to_fresh_optimized_all_inputs_byte_SHA_equal'] is True
    records = {}
    for provider in ('historical-optimized', 'fresh-optimized', 'fresh-preopt'):
        label = f'source-only-v1-{provider}'
        assert (ART / f'{label}.exit').read_text().strip() == '0'
        raw = json.loads((ART / f'{label}.json').read_text())
        assert raw['all_38_reference_PASS'] is True and len(raw['cases']) == 38
        assert all(record['reference_PASS'] and record['coverage'] == 'full' for record in raw['cases'])
        records[provider] = {record['name']: record for record in raw['cases']}
    for case in cases:
        old = records['historical-optimized'][case.name]
        fresh = records['fresh-optimized'][case.name]
        assert fresh['input_sha256'] == old['input_sha256']
        assert fresh['output_sha256'] == old['output_sha256'] and fresh['full_output_bitwise'] is True
    return bench, cases, manifest, package, required, records['fresh-optimized']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grant', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--guard-only', action='store_true')
    parser.add_argument('--issue-grant', action='store_true')
    args = parser.parse_args()
    bench, cases, manifest, package, required, source = preflight()
    if args.issue_grant:
        assert not args.grant.exists()
        issued = time.time()
        grant = {'canonical_version': CANONICAL_VERSION, 'coordinator_release': True, 'gpu': 5,
                 'runtime_cpus': '16-31', 'issued_epoch': issued, 'epoch': issued + 45,
                 'source_digest': manifest['source']['source_digest'], 'canonical_manifest_sha256': MANIFEST_SHA,
                 'compiler_policy_sha256': digest(STAGE / 'compiler-policy-verified.json'),
                 'source_summary_sha256': digest(ART / 'matched-native-icpx-v1/source-only-gpu5-accuracy-v1-summary.json'),
                 'fresh_source_raw_sha256': digest(ART / 'source-only-v1-fresh-optimized.json'),
                 'normal_import_readiness_sha256': digest(STAGE / 'normal-import-readiness.json'),
                 'workload': WORKLOAD, 'phase': 'normal-v4-AFTER-accuracy-only', 'official_performance_wave': False,
                 'no_timing_warmup_events_graphs': True, 'additional_model_matching_cases_authorized': False,
                 'authorization': 'latest user authorizes genuine normal v4 all38 fullrefs and shared SOURCE bits only'}
        text = json.dumps(grant, indent=2) + '\n'
        args.grant.write_text(text)
        args.grant.with_name(args.grant.stem + '-issued.json').write_text(text)
        print(json.dumps(grant), flush=True)
        return
    grant = json.loads(args.grant.read_text())
    assert grant['coordinator_release'] is True and grant['gpu'] == 5
    assert grant['canonical_version'] == CANONICAL_VERSION and grant['workload'] == WORKLOAD
    assert grant['source_digest'] == manifest['source']['source_digest']
    assert grant['canonical_manifest_sha256'] == MANIFEST_SHA
    assert grant['compiler_policy_sha256'] == digest(STAGE / 'compiler-policy-verified.json')
    assert grant['source_summary_sha256'] == digest(ART / 'matched-native-icpx-v1/source-only-gpu5-accuracy-v1-summary.json')
    assert grant['fresh_source_raw_sha256'] == digest(ART / 'source-only-v1-fresh-optimized.json')
    assert grant['normal_import_readiness_sha256'] == digest(STAGE / 'normal-import-readiness.json')
    assert grant['phase'] == 'normal-v4-AFTER-accuracy-only'
    assert grant['official_performance_wave'] is False and grant['no_timing_warmup_events_graphs'] is True
    if args.guard_only:
        print('B8 preflight PASS; current full normal package; 38 unchanged fixtures; full CPU reference; no performance')
        return
    assert os.environ.get('ZE_AFFINITY_MASK') == '5'
    assert os.sched_getaffinity(0) == set(range(16, 32))
    import torch
    import sgl_kernel
    assert Path(sgl_kernel.__file__).resolve().parent == package.resolve()
    assert not torch.xpu.is_initialized()
    hashes = {name: record['sha256'] for name, record in manifest['binaries'].items()}
    verify_loaded(package, hashes, set(hashes))
    assert torch._C._dispatch_has_kernel_for_dispatch_key('sgl_kernel::fwd', 'XPU')
    while time.time() < grant['epoch']:
        time.sleep(0.25)
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    from sgl_kernel.utils import is_xe3_arch
    assert bench['is_fmha_arch_xe35'](torch, is_xe3_arch)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    result = {'runtime_pid': os.getpid(), 'canonical_version': CANONICAL_VERSION,
              'grant_sha256': digest(args.grant), 'canonical_manifest_sha256': MANIFEST_SHA,
              'compiler_policy_sha256': grant['compiler_policy_sha256'], 'source_digest': grant['source_digest'],
              'workload': WORKLOAD, 'scope': 'normal full v4 AFTER accuracy only; representative unchanged38, not actual-model-wave replay',
              'no_timing_warmup_events_graphs': True, 'native_launches_per_case': 1,
              'required_native_sha256': {name: hashes[name] for name in sorted(required)},
              'all_current_native_sha256': hashes, 'source_raw_sha256': grant['fresh_source_raw_sha256'],
              'source_output_provider': 'fresh optimized icpx SOURCE; saved historical tensors independently proved full-output bitwise equal for all38',
              'fixture_model_provenance_sha256': digest(ART / 'MODEL_FIXTURE_PROVENANCE_v1.json'), 'cases': []}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'runtime_pid': os.getpid(), 'normal_package': str(package), 'mapped_native_count': len(hashes)}), flush=True)
    with torch.inference_mode():
        for case in cases:
            torch.xpu.synchronize()
            time.sleep(2)
            print(json.dumps({'starting_case': case.name}), flush=True)
            tensors = bench['tensors_xe35'](case, torch)
            input_hashes = [tensor_digest(tensor, torch) for tensor in tensors[:-1]]
            assert input_hashes == source[case.name]['input_sha256'], 'SOURCE input bytes differ; do not compare unmatched inputs'
            bench['launch_xe35'](case, tensors, flash_attn_with_kvcache)
            torch.xpu.synchronize()
            actual = tensors[-1].cpu()
            record = {'name': case.name, 'requests': case.requests, 'query_length': case.query_length,
                      'cache_length': case.cache_length, 'window': case.window, 'input_sha256': input_hashes,
                      'output_sha256': tensor_digest(actual, torch), 'native_launches': 1}
            result['cases'].append(record)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            try:
                record.update(full_cpu_reference(case, tensors, torch))
                original = torch.load(ART / 'matched-native-icpx-v1/source-only-paired-v1' / f'{case.name}.pt',
                                      map_location='cpu', weights_only=True)
                assert tensor_digest(original, torch) == source[case.name]['output_sha256']
                assert actual.dtype == original.dtype == torch.bfloat16 and actual.shape == original.shape
                record['full_output_bitwise'] = torch.equal(actual.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8))
                record['output_sha256_equals_fresh_SOURCE'] = record['output_sha256'] == source[case.name]['output_sha256']
                record['source_port_max_abs'] = (actual.float() - original.float()).abs().max().item()
                record['accuracy'] = 'PASS' if record['reference_PASS'] and record['full_output_bitwise'] else 'FAIL'
            except Exception as error:
                record['accuracy'] = 'FAIL'
                record['failure'] = repr(error)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(record), flush=True)
            torch.xpu.synchronize()
            del tensors, actual
    verify_loaded(package, hashes, set(hashes))
    result['actual_runtime_provenance'] = bench['provenance_xe35'](sgl_kernel, torch)
    result['all_38_reference_PASS'] = len(result['cases']) == 38 and all(record.get('reference_PASS') for record in result['cases'])
    result['all_38_full_output_bitwise'] = len(result['cases']) == 38 and all(record.get('full_output_bitwise') for record in result['cases'])
    result['mapped_native_count'] = len(hashes)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if not result['all_38_reference_PASS'] or not result['all_38_full_output_bitwise']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
