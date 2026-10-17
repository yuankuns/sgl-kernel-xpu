import hashlib
import json
from pathlib import Path
import subprocess
import time

from canonical_check import ARTIFACT, CANONICAL, READY, TARGET, inventory, local_path, scoped_git_output, verify_reviewed_freeze

evidence = ARTIFACT.parents[2] / 'sgl-kernel-xpu/review-validation-artifacts/qk-before-after-20261017'
provenance = json.loads((evidence / 'provenance.json').read_text())
baseline_binary = local_path(provenance['before']['checkout']) / 'build/src/libsgl-ops-sycl-FusedQKNormRope.so'
assert inventory.digest_file(baseline_binary) == provenance['before']['qk_library_sha256']
baseline_manifest = local_path(provenance['evidence']['before_binary_manifest'])
assert inventory.digest_file(baseline_manifest) == provenance['evidence']['before_binary_manifest_sha256']
archived_baseline = next(line.split()[0] for line in baseline_manifest.read_text().splitlines() if './qk-norm-rope-main-baseline/' in line)
assert archived_baseline == provenance['before']['qk_library_sha256']
runtime = json.loads((evidence / 'runtime.json').read_text())
assert runtime['scope'] == {'batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1, 'dtype': 'bf16'}
assert runtime['gpu_physical_ze_index'] == 7
commands = json.loads((evidence / 'commands.json').read_text())
assert len(commands) == 20
for command in commands:
    log = local_path(command['log'])
    assert inventory.digest_file(log) == command['log_sha256']
    assert (log.with_suffix('.exit')).read_text().strip() == str(command['exit'])
assert (evidence / 'supervisor.exit').read_text().strip() == '0'
accuracy = {}
providers = {}
for label, commit in (('before', 'ab3a26c43a38e6aea7d73cc1349e4b5e210b772b'), ('after', 'feed315712ea91e6becbb983f3a4d56ac0cc3b96')):
    record = provenance[label]
    checkout = local_path(record['checkout'])
    assert record['commit'] == commit
    assert scoped_git_output(checkout, 'rev-parse', 'HEAD').decode().strip() == commit
    for name, expected in record['source_hashes'].items():
        data = scoped_git_output(checkout, 'show', f'{commit}:{name}')
        assert hashlib.sha256(data).hexdigest() == expected
        assert inventory.digest_file(checkout / name) == expected
    manifest_path = evidence / label / 'site-manifest.json'
    assert inventory.digest_file(manifest_path) == record['site_manifest_sha256']
    files = json.loads(manifest_path.read_text())
    expected_files = {str(Path(entry['path']).relative_to(f'{label}/site')): entry['sha256'] for entry in files}
    site = evidence / label / 'site'
    actual_files = inventory.inventory(site)
    assert set(actual_files) == set(expected_files)
    assert all(actual_files[name]['sha256'] == digest for name, digest in expected_files.items())
    assert inventory.digest_file(site / 'sgl_kernel/libsgl-ops-sycl-FusedQKNormRope.so') == record['qk_library_sha256']
    providers[label] = {'commit': commit, 'checkout': str(checkout), 'site': str(site), 'qk_library_sha256': record['qk_library_sha256'], 'site_manifest_sha256': record['site_manifest_sha256'], 'verified_package_files': len(files), 'verified_git_source_inputs': len(record['source_hashes']), 'staging': record['staging']}
    for tokens in (8, 32768):
        log = evidence / label / f'accuracy-t{tokens}.log'
        result = next(json.loads(line) for line in log.read_text().splitlines() if line.startswith('{') and 'violations' in line)
        assert result['tokens'] == tokens and result['dtype'] == 'bf16'
        assert result['heads'] == [128, 16, 144] and result['head_dim'] == 64
        assert result['elements'] == tokens * 144 * 64
        assert result['atol'] == result['rtol'] == 5e-3 and result['tail_equal']
        status = int(log.with_suffix('.exit').read_text())
        assert status == (1 if label == 'before' and tokens == 32768 else 0)
        assert result['violations'] == (3508 if label == 'before' and tokens == 32768 else 0)
        accuracy[f'{label}-t{tokens}'] = {'exit': status, 'log': str(log), 'log_sha256': inventory.digest_file(log), 'result': result}
graph_commands = [command for command in commands if command['name'].startswith('graph-')]
assert len(graph_commands) == 16
for command in graph_commands:
    assert command['exit'] == 0
    label = command['label']
    record = json.loads((evidence / label / f"{command['name']}.json").read_text())
    assert record['tokens'] in (8, 32768) and record['dtype'] == 'bf16'
    assert record['heads'] == [128, 16, 144] and record['head_dim'] == 64
    assert record['repeats'] == 32 and record['samples'] == len(record['raw_us']) == 21
    assert set(record['loaded_libraries'].values()) == {providers[label]['qk_library_sha256']}
    assert all(local_path(path).is_relative_to(evidence / label / 'site') for path in record['loaded_libraries'])
summary = json.loads((evidence / 'summary.json').read_text())
frozen = json.loads(READY.read_text())
reviewed, reviewed_path = verify_reviewed_freeze()
generic = (TARGET / 'src/sycl/FusedQKNormRope.cpp').read_text()
assert '#if SYCL_INTEL_TARGET == 35' in generic and 'at::ScalarType::BFloat16' in generic
assert 'fused_qk_norm_rope_xe35(' in generic
official_original = inventory.git_output(TARGET, 'show', '833c95208dccd2af5628c121222f91186a5b2de7:src/sycl/FusedQKNormRope.cpp')
payload = {'observed_epoch': time.time(), 'comparison': 'real preopt ab3a26c -> optimized feed315, NOT optimized SOURCE -> PORT', 'historical_gpu': 7, 'scope': runtime['scope'], 'providers': providers, 'accuracy': accuracy, 'historical_performance': [{name: value for name, value in record.items() if name not in ('runs', 'paired_speedups')} for record in summary['shapes']], 'historical_graph_runs_verified': 16, 'raw_samples_verified': 336, 'commands_logs_and_exits_verified': 20, 'upstream_base': '833c95208dccd2af5628c121222f91186a5b2de7', 'upstream_original_cpp_sha256': hashlib.sha256(official_original).hexdigest(), 'preopt_cpp_sha256': provenance['before']['source_hashes']['src/sycl/FusedQKNormRope.cpp'], 'preopt_ab3a_is_not_official_833c952': True, 'current_official_site_present': (CANONICAL / 'site').is_dir(), 'current_official_manifest_present': (CANONICAL / 'source-binary-manifest.json').is_file(), 'current_generic_BF16_routes_to_optimized_xe35_not_preopt': True, 'other_preserved_upstream_QK_APIs_validated_as_equivalent_baseline': False, 'preopt_t32768_accuracy_acceptance_eligible': False, 'fresh_gpu6_preopt_to_canonical_port_status': 'PENDING; requires main authorization, canonical package and independent official barrier; do not launch old GPU7 runner', 'estimated_additional_gpu6_minutes': [4, 6], 'estimate_basis': {'historical_20_subprocess_seconds': sum(command['seconds'] for command in commands), 'shape_cooling_seconds': 40, 'initial_wrapper_cooling_seconds': 5, 'extra_import_manifest_audit_budget': '20-60 seconds; package unavailable/wait/build excluded', 'regression_retest_extra_minutes': [1, 2]}, 'ready_source_sha256_unchanged': frozen['source_digest'], 'GPU_launched': False, 'fake_op_created': False}
(ARTIFACT / 'preopt-evidence-audit.json').write_text(json.dumps(payload, indent=2) + '\n')
print(json.dumps({name: value for name, value in payload.items() if name not in ('accuracy', 'historical_performance')}, indent=2))
