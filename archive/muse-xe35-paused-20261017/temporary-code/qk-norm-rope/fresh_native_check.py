import json
from pathlib import Path
import subprocess

from canonical_check import ARTIFACT, inventory, local_path


def verify_provider(role):
    label, reference = {
        'optimized-source': ('source', 'feed315712ea91e6becbb983f3a4d56ac0cc3b96'),
        'preopt': ('preopt', 'ab3a26c43a38e6aea7d73cc1349e4b5e210b772b'),
    }[role]
    run = ARTIFACT / 'native-icpx-cpu-v3'
    directory = run / role
    assert (run / 'host-supervisor.exit').read_text().strip() == '0'
    assert (run / 'container-supervisor.exit').read_text().strip() == '0'
    for stem in ('configure', 'qk-target-build', 'native-nm', 'native-readelf-symbols',
                 'native-readelf-needed', 'normal-package-import'):
        assert (directory / f'{stem}.exit').read_text().strip() == '0', stem
    assert (run / 'torch-cpu-import-probe.exit').read_text().strip() == '0'
    assert 'XPU_INITIALIZED False' in (run / 'torch-cpu-import-probe.log').read_text()
    for stem in ('configure', 'qk-target-build'):
        result = json.loads((directory / f'{stem}.result.json').read_text())
        assert result['actual_exit'] == 0 and result['policy_failure'] is None
        assert result['descendants_drained'] is True
        for kind in ('command', 'log'):
            assert inventory.digest_file(directory / f'{stem}.{kind}') == result[f'{kind}_sha256']
    complete = json.loads((directory / 'native-complete.json').read_text())
    assert complete['ref'] == reference and complete['role'] == role
    assert complete['QK_target_actual_exit'] == complete['normal_package_CPU_import_actual_exit'] == 0
    assert complete['native_ABI_export_verified'] is True
    assert complete['GPU_accuracy_or_performance_executed'] is False
    inputs = json.loads((directory / 'inputs.json').read_text())
    assert inputs['ref'] == reference
    assert inputs['only_changed_path'] == 'cmake/BuildFlags.cmake'
    assert {name for name in inputs['all_tracked_original_files']
            if inputs['all_tracked_original_files'][name] != inputs['build_snapshot_files'][name]} == {'cmake/BuildFlags.cmake'}
    source = directory / 'source'
    actual = {str(path.relative_to(source)): inventory.digest_file(path)
              for path in source.rglob('*') if path.is_file()}
    assert actual == inputs['build_snapshot_files'], 'Fresh original-source snapshot changed'
    assert inventory.digest_file(directory / 'tracked-source.tar') == inputs['archive_sha256']
    pin = 'e87870bb7aacf83bfb3be9da685f148316b8df23'
    dependencies = json.loads((run / 'original-deps.json').read_text())
    comparison = json.loads((run / 'e878-filehash-equivalent-comparison.json').read_text())
    assert complete['original_deps_pin'] == dependencies['head'] == comparison['clean_actual_HEAD'] == pin
    assert comparison['historical_preopt_cutlass_mount_HEAD_or_byte_equivalence_proven'] is False
    dependency_path = local_path(dependencies['path'])
    for relative, expected in dependencies['tracked_files'].items():
        assert inventory.digest_file(dependency_path / relative) == expected, relative
    git = ['git', '-c', f'safe.directory={dependency_path}', '-C', str(dependency_path)]
    assert subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip() == pin
    assert not subprocess.check_output([*git, 'status', '--porcelain'], text=True).strip()
    policy_path = directory / 'compiler-policy-verified.json'
    policy = json.loads(policy_path.read_text())
    assert policy['compiler_id'] == 'IntelLLVM'
    assert policy['cxx'] == policy['sycl'] == policy['host'] == '/opt/gfx-deps/dpcpp/bin/icpx'
    assert policy['c'] == '/opt/gfx-deps/dpcpp/bin/icx'
    assert policy['torch_finder_family'] == 'icx'
    for language, record in policy['real_identity_files'].items():
        identity = local_path(record['path'])
        assert inventory.digest_file(identity) == record['sha256']
        assert f'set(CMAKE_{language}_COMPILER_ID "IntelLLVM")' in identity.read_text()
    assert inventory.digest_file(local_path(policy['compile_recipe'])) == policy['compile_recipe_sha256']
    site = local_path(complete['site'])
    manifest_path = directory / 'site-manifest.json'
    assert inventory.digest_file(manifest_path) == complete['site_manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    native = site / 'sgl_kernel/libsgl-ops-sycl-FusedQKNormRope.so'
    assert inventory.digest_file(native) == complete['fresh_native_sha256']
    actual_site = {str(path.relative_to(site)): inventory.digest_file(path)
                   for path in site.rglob('*') if path.is_file()}
    assert actual_site == manifest, 'Fresh normal baseline package inventory changed'
    result = {
        f'{label}_commit': reference, f'{label}_site': str(site),
        f'{label}_manifest': str(manifest_path),
        f'{label}_manifest_sha256': inventory.digest_file(manifest_path),
        f'{label}_manifest_format': 'relative-sha256-map',
        f'{label}_qk_library_sha256': complete['fresh_native_sha256'],
        f'{label}_native_compiler_proof': str(policy_path),
        f'{label}_native_compiler_proof_sha256': inventory.digest_file(policy_path),
        f'{label}_native_build_scope': complete['scope'],
        f'{label}_original_declared_dependency_pin': pin,
        f'{label}_historical_cutlass_mount_HEAD_proven': False,
        f'{label}_fresh_icpx_GPU_precision_verified': False,
    }
    accuracy = ARTIFACT / 'source-accuracy-v1'
    audit_path = accuracy / 'final-evidence-audit.json'
    if audit_path.exists():
        assert (accuracy / 'final-evidence-audit.exit').read_text().strip() == '0'
        audit = json.loads(audit_path.read_text())
        assert audit['actual_exit'] == 0 and audit['PORT_v4_tested'] is False
        assert audit['timing_graph_events_warmup_executed'] is False
        assert audit['physical_GPU6_lock_released'] is True
        accuracy_role = 'fresh-source' if label == 'source' else 'fresh-preopt'
        phases = json.loads((accuracy / 'phase-results.json').read_text())
        values = [phases[f'{accuracy_role}-t{tokens}'] for tokens in (8, 32768)]
        for value in values:
            assert value['result']['mapped_after']['native_sha256'] == complete['fresh_native_sha256']
            assert value['result']['mapped_after']['manifest_sha256'] == complete['site_manifest_sha256']
            assert value['result']['atol'] == value['result']['rtol'] == 5e-3
        passed = all(value['actual_exit'] == 0 and value['result']['violations'] == 0
                     and value['result']['tail_equal'] for value in values)
        result.update({f'{label}_fresh_icpx_GPU_precision_verified': passed,
                       f'{label}_fresh_icpx_GPU_accuracy_status': 'PASS' if passed else 'FAIL',
                       f'{label}_fresh_icpx_GPU_accuracy_evidence': str(audit_path),
                       f'{label}_fresh_icpx_GPU_accuracy_evidence_sha256': inventory.digest_file(audit_path),
                       f'{label}_fresh_icpx_GPU_accuracy': [
                           {'tokens': value['result']['tokens'], 'violations': value['result']['violations'],
                            'actual_exit': value['actual_exit'], 'tail_bitwise': value['result']['tail_equal']}
                           for value in values],
                       f'{label}_GPU6_SOURCE_accuracy_only_not_PORT_or_perf': True})
    return result
