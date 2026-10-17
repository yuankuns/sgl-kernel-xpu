import argparse
import json
from pathlib import Path
import time

from check_official import canonical_preflight, digest
from model_semantic_accuracy import ROOT, WORKLOAD, scope_guard, verify_plan
from settings import ART, STAGE


def preflight():
    scope_guard()
    plan = verify_plan()
    manifest, manifest_file, _, _ = canonical_preflight()
    for label in ('model51-cpu-source-v2', 'model51-cpu-port-v2', 'model-fixture-provenance-cpu-v1',
                  'source-only-gpu5-suite-v1', 'source-only-gpu5-summary-v1', 'normal-v4-gpu5-accuracy-only-v1',
                  'normal-v4-gpu5-accuracy-only-v1-summary'):
        assert (ART / f'{label}.exit').read_text().strip() == '0', label
    gates = {provider: json.loads((ROOT / f'{provider}-CPU-gate.json').read_text()) for provider in ('source', 'port')}
    for provider, gate in gates.items():
        assert gate['CPU_gate'] is True and gate['GPU_initialized'] is False
        assert gate['real_backend_and_registered_wrapper_identity'] is True
        assert gate['plan_sha256'] == digest(ROOT / 'plan.json') and len(gate['cases']) == 51
        assert gate['full_reference_and_output_plan_confirmed'] is True
        assert gate['mapped_native_count'] == (156 if provider == 'source' else 735)
    assert gates['source']['cases'] == gates['port']['cases'], 'shared fixture inputs differ'
    prior = json.loads((ART / 'normal-v4-gpu5-accuracy-only-v1-summary.json').read_text())
    assert prior['exit'] == 0 and prior['GPU_lock_released'] is True and prior['live_runtime_processes'] == []
    assert prior['canonical_manifest_sha256'] == digest(manifest_file)
    return plan, manifest, manifest_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--issue', action='store_true')
    args = parser.parse_args()
    plan, manifest, manifest_file = preflight()
    grant_file = ROOT / 'gpu5-accuracy-only-barrier-v1.json'
    if args.issue:
        assert not grant_file.exists()
        issued = time.time()
        grant = {'phase': 'model-config51-normal-accuracy-only', 'gpu': 5, 'runtime_cpus': '16-31',
                 'coordinator_release': True, 'issued_epoch': issued, 'epoch': issued + 120,
                 'workload': WORKLOAD, 'canonical_version': 'canonical-v4-icpx',
                 'source_digest': manifest['source']['source_digest'], 'canonical_manifest_sha256': digest(manifest_file),
                 'compiler_policy_sha256': digest(STAGE / 'compiler-policy-verified.json'),
                 'plan_sha256': digest(ROOT / 'plan.json'), 'harness_sha256': plan['harness_sha256'],
                 'source_cpu_gate_sha256': digest(ROOT / 'source-CPU-gate.json'),
                 'port_cpu_gate_sha256': digest(ROOT / 'port-CPU-gate.json'),
                 'fresh_source_package_manifest_sha256': digest(ART / 'matched-native-icpx-v1/source/normal-package-manifest.json'),
                 'source_and_port_cases': 51, 'official_performance_wave': False, 'NV_handoff_authorized': False,
                 'no_timing_warmup_events_graphs': True, 'prefill_and_decode_port_route': 'genuine generic fallback; actual backend omits max_seqlen_k, keep0',
                 'authorization': 'main latest user explicitly authorizes this independent future+120 precision grant only after all CPU gates0',
                 'case_GPU_run_authorized_field_boundary': 'inherited False describes CPU-only prototype origin; only this new phase-specific grant authorizes runtime'}
        text = json.dumps(grant, indent=2) + '\n'
        grant_file.write_text(text)
        grant_file.with_name(grant_file.stem + '-issued.json').write_text(text)
        print(json.dumps(grant), flush=True)
    else:
        grant = json.loads(grant_file.read_text())
        assert grant['coordinator_release'] is True and grant['plan_sha256'] == digest(ROOT / 'plan.json')
        print('all CPU gates PASS; immutable shared model51 fixtures; normal SOURCE156/AFTER735 maps; no performance')


if __name__ == '__main__':
    main()
