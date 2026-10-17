import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
from contract import ROOT, ROLES, case, digest, hashes, local, read, require, save, strict, unchanged
from runner import bits_sha, validate_output


def main():
    require(sorted(os.sched_getaffinity(0)) == list(range(32, 48)), 'Assigned audit CPUs32–47')
    unchanged()
    grant = read(ROOT / 'main-paired-trace-grant.json')
    require(hashes() == grant['driver_files_sha256'], 'Frozen driver SHA changed')
    for path in ('host-supervisor.exit', 'host-run/wrapper.exit', 'run/source.exit', 'run/port.exit'):
        require((ROOT / path).read_text().strip() == '0', 'Actual terminal must be0')
    pids = [read(ROOT / 'run/supervisor.json')['pid']]
    pids.extend(read(ROOT / f'run/{role}.pid.json')['pid'] for role in ROLES)
    require(all(not Path(f'/proc/{pid}').exists() for pid in pids), 'Own container processes not drained')
    matching = {}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            arguments = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (OSError, PermissionError):
            continue
        if 'diagnose.py --mode' in arguments and str(ROOT) in arguments:
            matching[process.name] = arguments
    require(not matching, 'Own diagnostic process still active')
    import torch

    torch.set_num_threads(1)
    require(not torch.xpu.is_initialized(), 'CPU-only audit')
    plan, fields, fixture = case()
    require(digest(local(fixture['fixture']['path'])) == fixture['fixture']['sha256'], 'Original complete fixture SHA')
    require(digest(local(fixture['expected_SOURCE_output'])) == fixture['expected_SOURCE_output_sha256'], 'Original golden SHA')
    inputs = torch.load(local(fixture['fixture']['path']), map_location='cpu', weights_only=True, mmap=True)
    golden = torch.load(local(fixture['expected_SOURCE_output']), map_location='cpu', weights_only=True, mmap=True)['actual']
    roles = {}
    for role in ROLES:
        directory = ROOT / 'run' / role
        start = read(directory / 'runtime-start.json')
        end = read(directory / 'runtime-end.json')
        maps = start['normal']['mapped_native_SHA']
        require(maps == end['normal']['mapped_native_SHA'], 'Actual before/after native maps differ')
        require(len(maps) == (155 if role == 'source' else 712), 'Full normal mapped package count')
        require(all(digest(local(path)) == checksum for path, checksum in maps.items()), 'Mapped native file SHA')
        require(all(digest(local(path)) == checksum for path, checksum in grant['core_loaded_stack_sha256'].items()), 'Core file SHA')
        witness = torch.load(directory / 'complete-input-output-witness.pt', map_location='cpu', weights_only=True, mmap=True)
        require(witness.keys() == inputs.keys() | {'actual'}, 'Complete input witness keys')
        for name, tensor in inputs.items():
            require(tensor.dtype == witness[name].dtype and tensor.shape == witness[name].shape, 'Full input metadata')
            require(torch.equal(tensor.view(torch.uint8), witness[name].view(torch.uint8)), 'All complete input bits')
        sentinel = torch.load(directory / 'sentinel-before-replay.pt', map_location='cpu', weights_only=True, mmap=True)
        sentinel_record = read(directory / 'sentinel-before-replay.json')
        require(sentinel.shape == inputs['original'].shape and sentinel.dtype == torch.bfloat16, 'Complete sentinel layout')
        require(torch.isnan(sentinel).all().item(), 'Full sentinel allNaN')
        require(bits_sha(sentinel) == sentinel_record['full_bits_sha256'], 'Sentinel full bits SHA')
        require(digest(directory / 'sentinel-before-replay.pt') == sentinel_record['full_saved_tensor_sha256'], 'Saved full sentinel SHA')
        correctness = validate_output(inputs['original'], witness['actual'], inputs, golden, 'captured_reset_adapter')
        require(correctness['actual_output_uint16_sha256'] == read(directory / 'sentinel-full-output.json')['actual_output_uint16_sha256'], 'Actual output full SHA')
        proof = strict((directory / 'capture.dot').read_text(), read(directory / 'full-window.trace.json'))
        recorded = read(directory / 'strict-replay-proof.json')
        require(all(recorded[key] == value for key, value in proof.items()), 'Independent original strict replay trace gate')
        raw = read(directory / 'raw-Kineto-events.json')
        device = [event for event in raw if 'XPU' in event.get('device_type', '')]
        require(device and all(event['end_ns'] > event['start_ns'] for event in device), 'Real positive raw device timestamps')
        pti = read(directory / 'PTI-device-timestamps.json')
        require(pti['raw_Kineto_device_records'] == device and pti['raw_standalone_PTI_callback_buffers_available'] is False, 'Actual PTI-backed records, not fake standalone callbacks')
        roles[role] = {'reference_and_bits': correctness, 'recomputed_replay_proof': proof,
            'full_native_SHA_count': len(maps), 'raw_Kineto_XPU_events': len(device),
            'complete_original_inputs_and_sentinel_SHA_verified': True}
        del witness, sentinel
    require(roles['source']['reference_and_bits'] == roles['port']['reference_and_bits'], 'Full paired output bits/reference result')
    require(not torch.xpu.is_initialized(), 'Audit initialized XPU')
    files = {str(path.relative_to(ROOT)): digest(path) for path in sorted((ROOT / 'run').rglob('*')) if path.is_file()}
    save(ROOT / 'final-evidence-audit.json', {'actual_exit': 0, 'epoch': time.time(), 'roles': roles,
        'grant_sha256': digest(ROOT / 'main-paired-trace-grant.json'), 'CPU_READY_sha256': digest(ROOT / 'CPU_READY.json'),
        'owned_container_PIDs_absent': pids, 'matching_live_owned_diagnostic_processes': matching,
        'all_run_files_SHA256': files, 'XPU_initialized': False, 'performance_acceptance': False,
        'old_controlled_trace_failure_and_19case_scope_unchanged': True})
    print('Independent complete input/output/sentinel/ref/raw trace/native/core SHA audit actual0; own container drain0; no performance acceptance', flush=True)


if __name__ == '__main__':
    main()
