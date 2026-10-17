import ast
import hashlib
import json
import os
from pathlib import Path

from canonical_check import ARTIFACT, inventory, local_path
from source_accuracy_v1 import GRANT, ROLES, RUN, VERSION, provider


def audit():
    import torch

    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    assert not torch.xpu.is_initialized()
    assert (RUN / 'host-supervisor.exit').read_text().strip() == '0'
    assert (RUN / 'cpu-preflight.exit').read_text().strip() == '0'
    assert (RUN / 'drain-check.exit').read_text().strip() == '0'
    assert (RUN / 'gpu6-release-lock.exit').read_text().strip() == '1'
    assert not (RUN / 'gpu6-release-lock.log').read_text().strip()
    grant = json.loads(GRANT.read_text())
    assert grant['version'] == VERSION and grant['grant_kind'] == 'SOURCE-accuracy-only'
    assert grant['PORT_comparison_authorized'] is False
    assert grant['timing_graph_events_warmup_allowed'] is False
    assert grant['W4_lease_transfer_authorized'] is False
    records = json.loads((RUN / 'phase-results.json').read_text())
    assert len(records) == 6
    raw_files = {}
    for role in ROLES:
        information, site, expected = provider(role)
        assert information == grant['providers'][role]
        assert (RUN / role / 'cpu-import.exit').read_text().strip() == '0'
        cpu = json.loads((RUN / role / 'cpu-import-audit.json').read_text())
        assert cpu['GPU_initialized'] is False and cpu['whole_site_files_verified'] == len(expected)
        for tokens in (8, 32768):
            prefix = RUN / role / f'accuracy-t{tokens}'
            actual_exit = int(prefix.with_suffix('.exit').read_text())
            entry = json.loads(prefix.with_suffix('.json').read_text())
            assert records[f'{role}-t{tokens}'] == {'actual_exit': actual_exit, 'result': entry}
            assert entry['atol'] == entry['rtol'] == 5e-3
            assert entry['dtype'] == 'bf16' and entry['tokens'] == tokens
            assert entry['heads'] == [128, 16, 144] and entry['head_dim'] == 64
            assert entry['elements'] == tokens * 144 * 64
            assert entry['tail_equal'] and entry['native_invocations'] == 1
            assert entry['timing_graph_events_warmup_executed'] is False
            if role == 'fresh-preopt' and tokens == 32768:
                assert actual_exit == 1 and entry['violations'] == 3508
                assert 'Strict production CPU reference failed' in prefix.with_suffix('.log').read_text()
            else:
                assert actual_exit == 0 and entry['violations'] == 0
            for stage in ('mapped_before', 'mapped_after'):
                mapped = entry[stage]
                assert mapped['manifest_sha256'] == information['manifest_sha256']
                assert mapped['native_sha256'] == information['native_sha256']
                assert mapped['whole_site_files_verified'] == len(expected)
                for absolute, checksum in mapped['all_mapped_native_hashes_verified'].items():
                    relative = str(local_path(absolute).relative_to(site))
                    assert expected[relative] == checksum
            for suffix in ('.pt', '.json', '.log', '.exit', '.pid.json', '.command.json'):
                path = prefix.with_suffix(suffix)
                raw_files[str(path.relative_to(RUN))] = {'sha256': inventory.digest_file(path), 'bytes': path.stat().st_size}
    bitwise = {}
    for tokens in (8, 32768):
        tensors = {role: torch.load(RUN / role / f'accuracy-t{tokens}.pt', map_location='cpu', weights_only=True)
                   for role in ROLES}
        historical, fresh, preopt = (tensors[role] for role in ROLES)
        assert torch.equal(historical['original'].view(torch.uint16), fresh['original'].view(torch.uint16))
        assert torch.equal(historical['original'].view(torch.uint16), preopt['original'].view(torch.uint16))
        assert torch.equal(historical['actual'].view(torch.uint16), fresh['actual'].view(torch.uint16))
        for role, data in tensors.items():
            assert torch.equal(data['actual'][:, 144 * 64:].view(torch.uint16), data['original'][:, 144 * 64:].view(torch.uint16))
            for name in ('positions', 'query_weight', 'key_weight'):
                assert torch.equal(historical[name], data[name])
            entry = records[f'{role}-t{tokens}']['result']
            for name, field in (('original', 'input_sha256'), ('actual', 'output_sha256')):
                assert hashlib.sha256(data[name].view(torch.uint16).numpy().tobytes()).hexdigest() == entry[field]
        bitwise[str(tokens)] = {'actual_tensor_entire_QKV_bitwise': True, 'all_three_inputs_bitwise': True,
                               'all_three_tail_bitwise': True, 'output_elements': historical['actual'].numel()}
        del tensors, historical, fresh, preopt, data
    assert not torch.xpu.is_initialized()
    script = ARTIFACT / 'source_accuracy_v1.py'
    ast.parse(script.read_text())
    forbidden = ('CUDAGraph', 'XPUGraph', 'elapsed_time', 'perf_counter', 'monotonic')
    assert not any(name in script.read_text() for name in forbidden)
    payload = {'actual_exit': 0, 'version': VERSION, 'handle': 99006,
               'host_supervisor_pid': 2978761, 'gpu_supervisor_pid': 188114,
               'grant_sha256': inventory.digest_file(GRANT), 'source_accuracy_script_sha256': inventory.digest_file(script),
               'source_accuracy_launcher_sha256': inventory.digest_file(ARTIFACT / 'source_accuracy_v1.sh'),
               'phase_count': 6, 'all_QKV_CPU_tensor_recheck': bitwise,
               'fresh_preopt_prefill_accuracy': 'FAIL:3508 violations; true phase exit1 preserved',
               'SOURCE_equivalence_accuracy': 'PASS: both BF16 T8/T32768 strict0 and full QKV/tail bitwise',
               'timing_graph_events_warmup_executed': False, 'PORT_v4_tested': False,
               'audit_GPU_initialized': False, 'physical_GPU6_lock_released': True,
               'logical_GPU6_lease_transferred': False, 'raw_files': raw_files}
    (RUN / 'final-evidence-audit.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({key: value for key, value in payload.items() if key != 'raw_files'}, indent=2))


if __name__ == '__main__':
    audit()
