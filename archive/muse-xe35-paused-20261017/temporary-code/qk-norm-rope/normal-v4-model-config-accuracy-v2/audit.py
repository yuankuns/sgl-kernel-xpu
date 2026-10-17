import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from run_accuracy import ARTIFACT, RUN, ROLES, cpu_reference, digest, inventory, local_path, record, save, verify_inputs


def main():
    import torch

    assert set(os.sched_getaffinity(0)) == set(range(32, 48)) and not torch.xpu.is_initialized()
    assert (RUN / 'host-supervisor.exit').read_text().strip() == '0'
    data = verify_inputs()
    grant = json.loads((RUN / 'accuracy-only-grant.json').read_text())
    assert grant['driver_sha256'] == data['driver_sha256'] and grant['launcher_sha256'] == data['launcher_sha256']
    assert grant['input_manifest_sha256'] == digest(RUN / 'input-manifest.json')
    fixtures = json.loads((RUN / 'input-manifest.json').read_text())
    mapped = {}
    observed = [json.loads((RUN / 'gpu-supervisor.json').read_text())['pid'],
                json.loads((RUN / 'cpu-prepare.json').read_text())['pid']]
    for role in ROLES:
        assert (RUN / role / 'gpu-role.exit').read_text().strip() == '0'
        runtime = json.loads((RUN / role / 'runtime-audit.json').read_text())
        if role == 'port':
            manifest = json.loads(local_path(data['canonical_manifest']).read_text())
            site = local_path(data['port_site'])
            expected = {name: item['sha256'] for name, item in (manifest['installed_files'] | manifest['python_snapshot_files']).items()}
            assert runtime['before']['mapped_count'] == runtime['after']['mapped_count'] == 712
        else:
            site = local_path(data['source_provider']['source_site'])
            expected = json.loads(local_path(data['source_provider']['source_manifest']).read_text())
        actual = {str(path.relative_to(site)): digest(path) for path in site.rglob('*') if path.is_file()}
        assert actual == expected
        for stage in ('before', 'after'):
            for path, checksum in runtime[stage]['mapped_native_SHA'].items():
                assert expected[str(local_path(path).relative_to(site))] == checksum
            assert 'sgl_kernel::fused_qk_norm_rope(' in runtime[stage]['real_generic_schema']
        assert runtime['case_count'] == 17 and runtime['architecture'] == [3, 5]
        mapped[role] = runtime['after']['mapped_count']
        for mode in ('cpu-normal', 'gpu-role'):
            observed.append(json.loads((RUN / role / f'{mode}.pid.json').read_text())['pid'])
    results = {}
    raw = {}
    for case in data['cases']:
        name = case['name']
        fixture = fixtures[name]
        assert digest(local_path(fixture['path'])) == fixture['sha256']
        inputs = torch.load(local_path(fixture['path']), map_location='cpu', weights_only=True)
        outputs = {role: torch.load(RUN / role / f'{name}.pt', map_location='cpu', weights_only=True) for role in ROLES}
        assert torch.equal(outputs['source']['actual'].view(torch.uint16), outputs['port']['actual'].view(torch.uint16))
        references = {}
        for role in ROLES:
            for field in ('original', 'query_weight', 'key_weight'):
                assert torch.equal(inputs[field].view(torch.uint16), outputs[role][field].view(torch.uint16))
            assert torch.equal(inputs['positions'], outputs[role]['positions'])
            assert torch.equal(outputs[role]['actual'][:, 9216:].view(torch.uint16), inputs['original'][:, 9216:].view(torch.uint16))
            prefix = RUN / role / name
            phase = json.loads(prefix.with_suffix('.json').read_text())
            assert prefix.with_suffix('.exit').read_text().strip() == '0'
            assert phase['violations'] == 0 and phase['atol'] == phase['rtol'] == 5e-3 and phase['tail_bitwise']
            assert phase['native_invocations'] == 1 and not phase['timing_events_graphs_warmup_executed']
            violations, elements, maximum = cpu_reference(inputs['original'], outputs[role]['actual'], inputs['positions'].to(torch.int32),
                                                         inputs['query_weight'].float(), inputs['key_weight'].float())
            assert violations == 0 and elements == phase['elements'] and maximum == phase['max_abs']
            references[role] = {'actual_exit': 0, 'strict_CPU_reference_violations': 0, 'elements': elements, 'max_abs': maximum}
            for suffix in ('.pt', '.json', '.exit'):
                path = prefix.with_suffix(suffix)
                raw[str(path.relative_to(RUN))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        results[name] = {'source_and_port_reference': references, 'full_QKV_SOURCE_PORT_bitwise_differences': 0,
            'input_and_tail_bitwise': True, 'output_elements': outputs['port']['actual'].numel(), 'case': case}
        del inputs, outputs
    live = {}
    for pid in observed:
        process = Path('/proc') / str(pid)
        try:
            command = (process / 'cmdline').read_bytes().replace(b'\x00', b' ').decode()
            cwd = (process / 'cwd').resolve()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if 'run_accuracy.py' in command and (cwd == RUN or str(RUN) in command):
            live[str(pid)] = command
    save(RUN / 'drain-check.json', {'observed_own_PIDs': observed, 'matching_live_own_processes': live, 'drained': not live})
    assert not live and not torch.xpu.is_initialized()
    save(RUN / 'final-evidence-audit.json', {'actual_exit': 0, 'version': data['version'], 'cases': results,
        'case_count': 17, 'provider_case_count': 34, 'mapped_native_counts': mapped,
        'genuine_model_adapter_generic_dispatcher_verified': True, 'raw_files': raw,
        'grant_sha256': digest(RUN / 'accuracy-only-grant.json'), 'driver_sha256': data['driver_sha256'],
        'canonical_manifest_sha256': data['canonical_manifest_sha256'], 'CPU_auditor_XPU_initialized': False,
        'GPU_processes_drained': True, 'timing_events_graphs_warmup_executed': False,
        'formal_performance_verified': False, 'actual_model_tensor_raw_replay': False,
        'all_production_invocations_verified': False, 'positions_gap': data['gap']})
    print(json.dumps({'actual_exit': 0, 'cases': results, 'mapped_native_counts': mapped}, indent=2), flush=True)


if __name__ == '__main__':
    main()
