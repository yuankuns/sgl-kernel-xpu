import hashlib
import json
import os
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
FORENSIC = DIRECTORY.parent
RAW = FORENSIC / 'fresh-unmeasured-artifact'


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    assert os.environ['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:cpu'
    assert set(os.sched_getaffinity(0)) == set(range(80, 96))
    import torch
    assert not torch.xpu.is_initialized()
    assert (DIRECTORY / 'wrapper.exit').read_text().strip() == (DIRECTORY / 'runner.exit').read_text().strip() == '1'
    assert (DIRECTORY / 'RUNTIME_SOURCE_CPU_PROVIDER.exit').read_text().strip() == '0'
    provider = json.loads((DIRECTORY / 'RUNTIME_SOURCE_CPU_PROVIDER.json').read_text())
    grant = json.loads((DIRECTORY / 'main-grant.json').read_text())
    assert not provider['Torch_XPU_initialized']
    assert provider['normal_stage_files'] == grant['normal_stage_files']
    assert provider['actual_native_mappings'] == grant['actual_SOURCE_native_mappings']
    assert provider['Torch_native_mappings_SHA256'] == grant['Torch_native_mappings_SHA256']
    assert grant['runner_sha256'] == digest(FORENSIC / 'forensic_runner.py')
    assert grant['CPU_audit_sha256'] == digest(FORENSIC / 'SOURCE_MASK_FAILURE_CPU_AUDIT.json')
    assert grant['CPU_seal_sha256'] == digest(FORENSIC / 'CPU_PROPOSAL_READY_SEAL.json')
    terminal = json.loads((RAW / 'terminal.json').read_text())
    assert terminal['actual_exit'] == 1 and terminal['events_created'] is False and terminal['timing_samples'] == 0
    initial_meta = json.loads((RAW / 'initial-eager.json').read_text())
    replay_meta = json.loads((RAW / 'singleton-replay-before-assert.json').read_text())
    assert digest(RAW / 'initial-eager.pt') == initial_meta['PT_sha256']
    assert digest(RAW / 'singleton-replay-before-assert.pt') == replay_meta['PT_sha256']
    eager = torch.load(RAW / 'initial-eager.pt', map_location='cpu', weights_only=True)
    replay = torch.load(RAW / 'singleton-replay-before-assert.pt', map_location='cpu', weights_only=True)
    assert torch.equal(eager['ids'], replay['ids'])
    assert torch.equal(replay['ids'], torch.tensor(grant['input_recipe']['ids'], dtype=torch.int64))
    differences = {}
    for name in ('mapped', 'invalid'):
        expected = eager['expected_' + name]
        torch.testing.assert_close(eager[name], expected, atol=0, rtol=0)
        assert torch.equal(expected, replay['expected_' + name])
        assert replay[name].dtype == expected.dtype and replay[name].shape == expected.shape
        differences[name] = {'new_saved_actual': replay[name].tolist(), 'saved_expected': expected.tolist(), 'eager_actual': eager[name].tolist(), 'mismatched_elements': (replay[name] != expected).sum().item(), 'wholebits_equal': torch.equal(replay[name].reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8))}
    assert differences['mapped']['mismatched_elements'] == 5
    capture = json.loads((RAW / 'capture-tensor-pointers.json').read_text())
    instantiate = json.loads((RAW / 'instantiate-pointers.json').read_text())
    current_bindings = replay_meta['actual_kernel_argument_tensor_bindings']
    assert capture == instantiate == current_bindings
    native = json.loads((RAW / 'native-DOT-before-check.json').read_text())
    assert digest(RAW / 'single-call.dot') == native['DOT_sha256']
    assert native['DOT_saved_before_instantiate_replay_and_check'] and native['pArguments_is_not_input_output_data_ptr']
    assert native['direct_capture_tensor_bindings'] == capture
    assert not native['native_arg_value_addresses_exposed']
    drain = json.loads((DIRECTORY / 'runner-drain.json').read_text())
    assert drain['actual_runner_exit'] == 1 and drain['all_owned_drained']
    ids = [json.loads((DIRECTORY / 'container-owned.json').read_text())['PID'], json.loads((DIRECTORY / 'runner-owned.json').read_text())['PID']]
    assert all(not Path('/proc', str(pid)).exists() for pid in ids)
    old = json.loads((FORENSIC / 'SOURCE_MASK_FAILURE_CPU_AUDIT.json').read_text())
    for path, record in old['files'].items():
        assert digest(path) == record['sha256'], 'Old failure/eager evidence changed'
    assert not torch.xpu.is_initialized()
    record = {'actual_CPU_postcheck_exit': 0, 'actual_forensic_runner_exit': 1, 'actual_wrapper_exit': 1, 'grant_sha256': digest(DIRECTORY / 'main-grant.json'), 'grant_epoch': grant['start_epoch'], 'SOURCE_stage_files_reverified_before_GPU_init': len(provider['normal_stage_files']), 'SOURCE_native_maps_reverified_before_GPU_init': provider['native_mapped_count'], 'Torch_native_providers_reverified_before_GPU_init': len(provider['Torch_native_mappings_SHA256']), 'eager_exact_reference_passed': True, 'new_whole_saved_failure': differences, 'new_isolated_mask_reproduces_mapped_5of8_failure_without_embedding_prefix': True, 'old_failed_values_inferred': False, 'capture_instantiate_replay_Tensor_binding_and_storage_ptrs_unchanged': True, 'native_arg_value_addresses_exposed': False, 'native_pArguments_is_not_tensor_data_ptr': True, 'root_cause_established': False, 'FM_shared_cause_claimed': False, 'events_created': False, 'timing_samples': 0, 'performance_accepted': False, 'original_partial75case_acceptance': False, 'container_actor_and_runner_absent': ids, 'original_raw_seals_unchanged': True, 'XPU_initialized_by_CPU_postcheck': False, 'raw_file_seals': {path.name: digest(path) for path in RAW.iterdir() if path.is_file()}, 'launch_file_seals': {path.name: digest(path) for path in DIRECTORY.iterdir() if path.is_file()}}
    with (DIRECTORY / 'FORENSIC_CPU_POSTCHECK.json').open('x') as output:
        output.write(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'actual_CPU_postcheck_exit': 0, 'actual_forensic_exit': 1, 'new_saved_failure': differences, 'root_cause_established': False, 'timing_samples': 0}))


if __name__ == '__main__':
    main()
