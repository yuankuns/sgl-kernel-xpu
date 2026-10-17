import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
RUN = Path(__file__).resolve().parent
ARTIFACT = RUN.parent
ACCURACY = ARTIFACT / 'normal-v4-model-config-accuracy-v2'
sys.path.insert(0, str(ACCURACY))
from run_accuracy_r2 import digest, local_path, normal_audit, verify_inputs as verify_accuracy_inputs

GRANT = RUN / 'performance-grant.json'
VERSION = 'normal-v4-model-config-perf-v1'
ORDER = ('port', 'source', 'source', 'port')


def scope():
    from source_accuracy_v1 import scope as original_scope

    original_scope()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def data():
    return json.loads((RUN / 'provenance.json').read_text())


def prepare():
    assert not (RUN / 'provenance.json').exists() and not GRANT.exists()
    original = verify_accuracy_inputs()
    audit = json.loads((ACCURACY / 'final-evidence-audit.json').read_text())
    assert audit['actual_exit'] == 0 and audit['case_count'] == 17
    assert (ACCURACY / 'final-evidence-audit.exit').read_text().strip() == '0'
    assert (ACCURACY / 'evidence-seal.exit').read_text().strip() == '0'
    assert json.loads((ACCURACY / 'drain-check.json').read_text())['drained']
    main_path = ARTIFACT.parent / 'ALL_ELEVEN_FULL_ICPX_V4_RESULTS.json'
    main = json.loads(main_path.read_text())
    assert main['actual_queue_exit'] == 0 and main['host_queue_pid_absent']
    common = json.loads((ACCURACY / 'input-manifest.json').read_text())
    cases = []
    for case in original['cases']:
        fixture = common[case['name']]
        assert digest(local_path(fixture['path'])) == fixture['sha256']
        assert audit['cases'][case['name']]['full_QKV_SOURCE_PORT_bitwise_differences'] == 0
        cases.append({'name': case['name'], 'group': 'actual-config-semantic-synthetic',
            'case': case, 'fixture': fixture, 'expected_SOURCE_output': str(ACCURACY / 'source' / f"{case['name']}.pt"),
            'expected_SOURCE_output_sha256': audit['raw_files'][f"source/{case['name']}.pt"]['sha256'],
            'parameters': original['parameters']})
    previous = ARTIFACT / 'normal-v4-accuracy-only-v1'
    previous_audit = json.loads((previous / 'final-evidence-audit.json').read_text())
    assert previous_audit['actual_exit'] == 0
    historical = ARTIFACT / 'source-accuracy-v1'
    historical_audit = json.loads((historical / 'final-evidence-audit.json').read_text())
    for tokens in (8, 32768):
        path = historical / 'fresh-source' / f'accuracy-t{tokens}.pt'
        checksum = historical_audit['raw_files'][f'fresh-source/accuracy-t{tokens}.pt']['sha256']
        assert digest(path) == checksum
        parameters = {**original['parameters'], 'low': 1.0, 'high': 1.0}
        cases.append({'name': f'legacy-representative-t{tokens}', 'group': 'legacy-aggregate-representative',
            'case': {'tokens': tokens, 'positions': 'arange(T)%4096; preserved legacy synthetic'},
            'fixture': {'path': str(path), 'sha256': checksum},
            'expected_SOURCE_output': str(path), 'expected_SOURCE_output_sha256': checksum, 'parameters': parameters})
    original.update({'version': VERSION, 'cases': cases,
        'accuracy_evidence_sha256': digest(ACCURACY / 'final-evidence-audit.json'),
        'accuracy_seal_sha256': digest(ACCURACY / 'evidence-seal.json'),
        'main_compile_drain_evidence': {'path': str(main_path), 'sha256': digest(main_path), 'queue_exit': 0},
        'driver_sha256': digest(RUN / 'run_perf.py'), 'launcher_sha256': digest(RUN / 'launch.sh'),
        'analysis_driver_sha256': digest(RUN / 'analyze.py'),
        'timing_protocol': {'device': 'stable physical GPU6', 'A': 'optimized original SOURCE', 'B': 'genuine normal full PORT',
            'order': list(ORDER), 'rounds': 3, 'bounded_retest_rounds': 3, 'samples': 21, 'repeats': 32, 'warmup': 5,
            'device_event': True, 'graph': False, 'reset_full_input_before_each_call_outside_event': True,
            'event_span': 'actual model framework adapter + position cast + generic dispatcher + native GPU work; submission gaps may be visible',
            'cooling_initial_seconds': 5, 'cooling_each_case_seconds': 2},
        'eight_card_concurrency_claim': False, 'preopt_correctness_matched_baseline': False})
    original['timing_events_graphs_warmup_allowed'] = {'timing': True, 'events': True, 'graphs': False, 'warmup': True}
    save(RUN / 'provenance.json', original)
    save(RUN / 'fixture-plan.json', cases)


def verify():
    original = verify_accuracy_inputs()
    value = data()
    assert value['version'] == VERSION and value['source_digest'] == original['source_digest']
    for filename, key in (('run_perf.py', 'driver_sha256'), ('launch.sh', 'launcher_sha256'), ('analyze.py', 'analysis_driver_sha256')):
        assert digest(RUN / filename) == value[key]
    assert digest(ACCURACY / 'final-evidence-audit.json') == value['accuracy_evidence_sha256']
    assert digest(ACCURACY / 'evidence-seal.json') == value['accuracy_seal_sha256']
    return value


def invoke(arguments, prefix):
    save(prefix.with_suffix('.command.json'), {'argv': arguments, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'ZE_AFFINITY_MASK': os.environ.get('ZE_AFFINITY_MASK')})
    with prefix.with_suffix('.log').open('w') as output:
        child = subprocess.Popen(arguments, stdout=output, stderr=subprocess.STDOUT)
        save(prefix.with_suffix('.pid.json'), {'pid': child.pid, 'ppid': os.getpid()})
        status = child.wait()
    prefix.with_suffix('.exit').write_text(f'{status}\n')
    assert status == 0, f'Actual exit{status}: {prefix}; preserve failure, no build/math retry'


def cpu_role(role):
    import torch

    assert not torch.xpu.is_initialized()
    maps = normal_audit(role)
    value = verify()
    for case in value['cases']:
        fixture = local_path(case['fixture']['path'])
        assert digest(fixture) == case['fixture']['sha256']
        assert digest(local_path(case['expected_SOURCE_output'])) == case['expected_SOURCE_output_sha256']
        tensors = torch.load(fixture, map_location='cpu', weights_only=True)
        assert tensors['original'].dtype == torch.bfloat16 and tensors['original'].shape == (case['case']['tokens'], 18432)
        assert tensors['positions'].numel() == case['case']['tokens']
        for name in ('query_weight', 'key_weight'):
            assert torch.equal(tensors[name].float(), tensors[name].to(torch.bfloat16).float())
        del tensors
    assert not torch.xpu.is_initialized()
    save(RUN / f'cpu-normal-{role}.json', {'actual_exit': 0, 'pid': os.getpid(), 'XPU_initialized': False, 'maps': maps, 'case_count': 19})


def cpu_preflight():
    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    for role in ('source', 'port'):
        invoke([sys.executable, str(RUN / 'run_perf.py'), '--mode', 'cpu-role', '--role', role], RUN / f'cpu-normal-{role}')


def publish_grant():
    value = verify()
    assert not GRANT.exists() and (RUN / 'cpu-preflight.exit').read_text().strip() == '0'
    for role in ('source', 'port'):
        assert not json.loads((RUN / f'cpu-normal-{role}.json').read_text())['XPU_initialized']
    save(GRANT, {'version': VERSION, 'grant_kind': 'main-authorized stable GPU6 device-event BAAB performance',
        'authorization': 'Explicit main/user approval after independent v2 accuracy audit0; not an eight-card concurrent launch',
        'coordinator_release': True, 'gpu': 6, 'cpus': '32-47', 'start_epoch': time.time() + 120,
        'package_manifest_sha256': value['canonical_manifest_sha256'], 'compiler_policy_sha256': value['compiler_policy_sha256'],
        'source_digest': value['source_digest'], 'SOURCE_manifest_sha256': value['source_provider']['source_manifest_sha256'],
        'SOURCE_native_sha256': value['source_provider']['source_qk_library_sha256'], 'PORT_native_SHA': value['port_native_SHA'],
        'fixture_plan_sha256': digest(RUN / 'fixture-plan.json'), 'all_shared_fixtures': {case['name']: case['fixture'] for case in value['cases']},
        'driver_sha256': value['driver_sha256'], 'launcher_sha256': value['launcher_sha256'],
        'analysis_driver_sha256': value['analysis_driver_sha256'], 'scope': value['scope'], 'protocol': value['timing_protocol'],
        'legacy_representatives_separate': True, 'failed_preopt3508_correctness_matched_baseline_allowed': False,
        'eight_card_concurrency_claim_allowed': False, 'W4_lease_transfer_authorized': False})
    print(GRANT.read_text(), flush=True)


def verify_grant():
    value = verify()
    grant = json.loads(GRANT.read_text())
    assert grant['version'] == VERSION and grant['gpu'] == 6 and grant['coordinator_release']
    for key, expected in {'driver_sha256': value['driver_sha256'], 'launcher_sha256': value['launcher_sha256'],
        'analysis_driver_sha256': value['analysis_driver_sha256'], 'package_manifest_sha256': value['canonical_manifest_sha256'],
        'source_digest': value['source_digest'], 'compiler_policy_sha256': value['compiler_policy_sha256'],
        'SOURCE_manifest_sha256': value['source_provider']['source_manifest_sha256'],
        'SOURCE_native_sha256': value['source_provider']['source_qk_library_sha256'], 'PORT_native_SHA': value['port_native_SHA'],
        'fixture_plan_sha256': digest(RUN / 'fixture-plan.json'), 'protocol': value['timing_protocol'],
        'all_shared_fixtures': {case['name']: case['fixture'] for case in value['cases']}}.items():
        assert grant[key] == expected
    assert not grant['W4_lease_transfer_authorized'] and not grant['eight_card_concurrency_claim_allowed']
    assert set(os.sched_getaffinity(0)) == set(range(32, 48)) and os.environ['ZE_AFFINITY_MASK'] == '6'
    return grant


def phase(role, phase_id):
    verify_grant()
    import torch

    maps_before = normal_audit(role)
    import sglang.kernels.ops.attention.fused_qknorm_rope as adapter

    assert hasattr(torch, 'xpu') and torch.xpu.is_available()
    assert tuple(torch.ops.sgl_kernel.query_device.default(torch.xpu.current_device())) == (3, 5)
    value = data()
    selection = json.loads((RUN / 'phase-plan.json').read_text())[phase_id]
    assert selection['role'] == role
    starts = [torch.xpu.Event(enable_timing=True) for repeat in range(32)]
    ends = [torch.xpu.Event(enable_timing=True) for repeat in range(32)]
    for name in selection['cases']:
        case = next(item for item in value['cases'] if item['name'] == name)
        path = local_path(case['fixture']['path'])
        assert digest(path) == case['fixture']['sha256']
        tensors = torch.load(path, map_location='cpu', weights_only=True)
        original = tensors['original']
        initial = original.to('xpu:0')
        qkv = initial.clone()
        query_weight = tensors['query_weight'].to(device='xpu:0', dtype=torch.bfloat16)
        key_weight = tensors['key_weight'].to(device='xpu:0', dtype=torch.bfloat16)
        positions = tensors['positions'].to('xpu:0')
        parameters = case['parameters']

        def actual_call():
            adapter.fused_qk_norm_rope(qkv, parameters['num_heads_q'], parameters['num_heads_k'], parameters['tail_heads'],
                parameters['head_dim'], parameters['norm_eps'], query_weight, key_weight, parameters['rope_theta'],
                parameters['is_neox'], positions, factor=parameters['factor'], low=parameters['low'], high=parameters['high'],
                attention_factor=parameters['attention_factor'], rotary_dim=parameters['rotary_dim'])

        torch.xpu.synchronize()
        time.sleep(2)
        actual_call()
        torch.xpu.synchronize()
        expected = torch.load(local_path(case['expected_SOURCE_output']), map_location='cpu', weights_only=True)['actual']
        sanity = qkv.cpu()
        assert torch.equal(sanity.view(torch.uint16), expected.view(torch.uint16)), 'Current input correctness sanity mismatch; no timing baseline allowed'
        assert torch.equal(sanity[:, 9216:].view(torch.uint16), original[:, 9216:].view(torch.uint16))
        del sanity, expected
        for warmup in range(5):
            qkv.copy_(initial)
            actual_call()
        torch.xpu.synchronize()
        raw = []
        raw_repeats = []
        for sample in range(21):
            for repeat in range(32):
                qkv.copy_(initial)
                starts[repeat].record()
                actual_call()
                ends[repeat].record()
            ends[-1].synchronize()
            elapsed = [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
            assert all(math.isfinite(item) and item > 0 for item in elapsed), 'Invalid actual device-event timing'
            raw_repeats.append(elapsed)
            raw.append(statistics.mean(elapsed))
        median = statistics.median(raw)
        result = {'case': case['name'], 'group': case['group'], 'role': role, 'phase': phase_id,
            'raw_us': raw, 'raw_per_repeat_us': raw_repeats, 'median_us': median,
            'MAD_us': statistics.median(abs(item - median) for item in raw),
            'samples': 21, 'repeats': 32, 'warmup': 5, 'device_event': True, 'graph': False,
            'full_input_reset_each_call_outside_event': True, 'current_full_output_SOURCE_bitwise_sanity': True,
            'current_tail_bitwise_sanity': True, 'fixture_sha256': case['fixture']['sha256'],
            'actual_operator_path': value['actual_entry_path'], 'before_after_optimization_claim': False}
        save(RUN / 'phases' / phase_id / f'{name}.json', result)
        print(json.dumps({key: item for key, item in result.items() if not key.startswith('raw_')}), flush=True)
        del tensors, original, qkv, initial, positions, query_weight, key_weight
    save(RUN / 'phases' / phase_id / 'runtime-audit.json', {'pid': os.getpid(), 'role': role,
        'before': maps_before, 'after': normal_audit(role), 'case_count': len(selection['cases'])})


def execute_rounds(first_round, cases, retest):
    plan_path = RUN / 'phase-plan.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    for round_index in range(first_round, first_round + 3):
        ordered = list(reversed(cases)) if round_index % 2 else list(cases)
        for slot, role in enumerate(ORDER):
            phase_id = f'r{round_index:02d}-s{slot}-{role}'
            plan[phase_id] = {'round': round_index, 'slot': slot, 'role': role, 'cases': ordered, 'bounded_retest': retest}
            save(plan_path, plan)
            prefix = RUN / 'phases' / phase_id / 'supervisor'
            prefix.parent.mkdir(parents=True, exist_ok=True)
            invoke([sys.executable, str(RUN / 'run_perf.py'), '--mode', 'phase', '--role', role, '--phase-id', phase_id], prefix)


def suite():
    grant = verify_grant()
    assert grant['start_epoch'] > time.time(), 'Fresh +120 performance epoch required'
    save(RUN / 'gpu-supervisor.json', {'pid': os.getpid(), 'ppid': os.getppid(), 'physical_gpu': 6, 'cpus': sorted(os.sched_getaffinity(0))})
    time.sleep(max(0, grant['start_epoch'] - time.time()))
    execute_rounds(0, [case['name'] for case in data()['cases']], False)
    invoke([sys.executable, str(RUN / 'analyze.py'), '--stage', 'initial'], RUN / 'analysis-initial')
    suspicious = json.loads((RUN / 'analysis-initial.json').read_text())['bounded_retest_cases']
    if suspicious:
        execute_rounds(3, suspicious, True)
    invoke([sys.executable, str(RUN / 'analyze.py'), '--stage', 'final'], RUN / 'analysis-final')


def main():
    scope()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare', 'cpu-preflight', 'cpu-role', 'publish-grant', 'suite', 'phase'), required=True)
    parser.add_argument('--role', choices=('source', 'port'))
    parser.add_argument('--phase-id')
    args = parser.parse_args()
    if args.mode == 'cpu-role':
        assert args.role and not args.phase_id
        cpu_role(args.role)
    elif args.mode == 'phase':
        assert args.role and args.phase_id
        phase(args.role, args.phase_id)
    else:
        assert not args.role and not args.phase_id
        {'prepare': prepare, 'cpu-preflight': cpu_preflight, 'publish-grant': publish_grant, 'suite': suite}[args.mode]()


if __name__ == '__main__':
    main()
