import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from driver import ARTIFACT, RUN, SOURCE, cpu_reference, digest, normal_audit, save, verify_inputs


def main():
    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    assert (RUN / 'host-supervisor.exit').read_text().strip() == '0'
    record = verify_inputs()
    import torch

    assert not torch.xpu.is_initialized()
    mapped = normal_audit()
    grant = json.loads((RUN / 'accuracy-only-grant.json').read_text())
    assert grant['package_manifest_sha256'] == record['canonical_manifest_sha256']
    assert grant['driver_sha256'] == digest(RUN / 'driver.py')
    assert grant['launcher_sha256'] == digest(RUN / 'launch.sh')
    assert grant['SOURCE_manifest_sha256'] == record['source_manifest_sha256']
    assert grant['compiler_policy_sha256'] == record['compiler_policy_proof']['compiler_policy_sha256']
    assert grant['source_digest'] == record['ready_source_sha256']
    assert not grant['formal_performance_wave'] and not grant['timing_events_graphs_warmup_allowed']
    raw = {}
    results = {}
    observed = [json.loads((RUN / 'cpu-preflight.json').read_text())['pid'],
                json.loads((RUN / 'gpu-supervisor.json').read_text())['pid']]
    for tokens in (8, 32768):
        prefix = RUN / f'accuracy-t{tokens}'
        assert prefix.with_suffix('.exit').read_text().strip() == '0'
        case = json.loads(prefix.with_suffix('.json').read_text())
        assert case['violations'] == 0 and case['tail_equal']
        assert case['atol'] == case['rtol'] == 5e-3
        assert case['full_QKV_SOURCE_output_bitwise_differences'] == 0
        assert case['actual_registered_op'] == 'sgl_kernel.fused_qk_norm_rope_xe35'
        assert case['native_invocations'] == 1 and not case['timing_events_graphs_warmup_executed']
        assert case['architecture'] == [3, 5]
        for stage in ('mapped_before', 'mapped_after'):
            assert case[stage]['mapped_native_count'] == 712
            assert case[stage]['all_mapped_native_hashes_verified'] == mapped['all_mapped_native_hashes_verified']
        source = torch.load(SOURCE / f'accuracy-t{tokens}.pt', map_location='cpu', weights_only=True)
        port = torch.load(prefix.with_suffix('.pt'), map_location='cpu', weights_only=True)
        for name in ('original', 'actual'):
            assert torch.equal(source[name].view(torch.uint16), port[name].view(torch.uint16))
        for name in ('positions', 'query_weight', 'key_weight'):
            assert torch.equal(source[name], port[name])
        assert torch.equal(port['original'][:, 144 * 64:].view(torch.uint16), port['actual'][:, 144 * 64:].view(torch.uint16))
        violations, elements, maximum_error = cpu_reference(port['original'], port['actual'], port['positions'], port['query_weight'], port['key_weight'])
        assert violations == 0 and elements == case['elements'] and maximum_error == case['max_abs']
        results[str(tokens)] = {'strict_CPU_reference_violations': violations, 'elements': elements,
            'max_abs': maximum_error, 'tail_bitwise': True, 'actual_full_QKV_SOURCE_PORT_bitwise': True,
            'full_QKV_output_elements': port['actual'].numel(), 'actual_exit': 0}
        for suffix in ('.pt', '.json', '.log', '.exit', '.command.json', '.pid.json'):
            path = prefix.with_suffix(suffix)
            raw[path.name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        observed.append(json.loads(prefix.with_suffix('.pid.json').read_text())['pid'])
        del source, port
    active = {}
    for pid in observed:
        process = Path('/proc') / str(pid)
        try:
            command = (process / 'cmdline').read_bytes().replace(b'\x00', b' ').decode()
            workdir = (process / 'cwd').resolve()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if 'driver.py' in command and (str(RUN) in command or workdir == RUN):
            active[str(pid)] = command
    save(RUN / 'drain-check.json', {'observed_own_PIDs': observed, 'matching_live_processes': active, 'drained': not active})
    assert not active, 'Own recorded accuracy processes still live'
    assert not torch.xpu.is_initialized()
    save(RUN / 'final-evidence-audit.json', {'actual_exit': 0, 'version': record['version'],
        'SOURCE_PORT_genuine_normal_full_accuracy': 'PASS', 'cases': results,
        'grant_sha256': digest(RUN / 'accuracy-only-grant.json'),
        'driver_sha256': digest(RUN / 'driver.py'), 'launcher_sha256': digest(RUN / 'launch.sh'),
        'source_raw_files_unchanged': record['source_raw_files'], 'raw_files': raw,
        'mapped_native_count': 712, 'normal_package_CPU_audit_GPU_initialized': False,
        'own_GPU_accuracy_processes_drained': True, 'timing_events_graphs_warmup_executed': False,
        'formal_performance_verified': False, 'preopt_rerun': False,
        'canonical_manifest_sha256': record['canonical_manifest_sha256']})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
