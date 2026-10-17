import json
import time
from pathlib import Path

from build_fm_metadata_icpx_v5 import ENTRY, PATCH_SHA256, digest, expected_entry


ROOT = Path(__file__).resolve().parent.parent


def main():
    artifact = ROOT / 'fmha'
    stage = artifact / 'canonical-v5-icpx'
    manifest_path = stage / 'source-binary-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    frozen = json.loads((artifact / 'canonical-source-freeze-v5-icpx.json').read_text())
    snapshot = json.loads((stage / 'source-snapshot.json').read_text())
    assert manifest['source']['source_digest'] == snapshot['source_digest'] == frozen['source_digest']
    assert snapshot['files'] == frozen['files']
    policy = json.loads((stage / 'compiler-policy-verified.json').read_text())
    assert policy['compiler_id'] == 'IntelLLVM' and policy['no_GNU_command'] is True
    for name in ('cxx', 'sycl', 'host'):
        assert policy[name] == '/opt/gfx-deps/dpcpp/bin/icpx'
    assert policy['c'] == '/opt/gfx-deps/dpcpp/bin/icx'
    for phase in ('configure', 'default-all-build', 'supervisor'):
        assert (stage / f'{phase}.exit').read_text().strip() == '0'
    assert json.loads((stage / 'source-readonly-postverify.json').read_text())['verified']
    assert manifest['install_actual_exit'] == 0
    build = manifest['build_evidence']
    assert build['exit'] == 0
    assert build['command_text'].strip() == f'cmake --build {manifest["build"]} --parallel 56'
    for filename, recorded in (('command_file', 'command_sha256'), ('exit_file', 'exit_file_sha256'), ('log_file', 'log_sha256')):
        assert digest(Path(build[filename])) == build[recorded]
    assert digest(stage / 'CMakeCache.txt') == manifest['cache_sha256']
    assert digest(stage / 'build.ninja') == manifest['ninja_sha256']
    assert manifest['cutlass_pin'] == '7b56b936b87f799f80dde60c2fd8d11c1128cebb'
    expected = manifest['installed_files'] | manifest['python_snapshot_files']
    site = Path(manifest['site'])
    for relative, record in expected.items():
        assert digest(site / relative) == record['sha256'], relative
    for record in manifest['binaries'].values():
        assert digest(site / record['path']) == record['sha256']
    old = json.loads((artifact / 'canonical-source-freeze-v4-icpx.json').read_text())
    assert old['files'].keys() == frozen['files'].keys()
    changed = [name for name in frozen['files'] if frozen['files'][name] != old['files'][name]]
    assert changed == [ENTRY]
    before = artifact / 'v5-metadata-bridge-preparation/before-entry.cpp'
    current = ROOT.parent / 'pr/muse-fmha-xe35' / ENTRY
    assert digest(before) == old['files'][ENTRY]['sha256']
    assert digest(current) == frozen['files'][ENTRY]['sha256']
    assert expected_entry(before.read_text()) == current.read_text()
    assert digest(artifact / 'MODEL51_XE35_METADATA_BRIDGE_PROPOSAL_v1.patch') == PATCH_SHA256
    report = {
        'component': 'fmha', 'version': 'canonical-v5-icpx', 'epoch': time.time(), 'actual_exit': 0,
        'source_digest': frozen['source_digest'], 'package_manifest_sha256': digest(manifest_path),
        'compiler_policy_sha256': digest(stage / 'compiler-policy-verified.json'),
        'only_source_delta': changed, 'approved_metadata_bridge_patch_sha256': PATCH_SHA256,
        'kernel_math_and_generic_fallback_unchanged': True,
        'default_all_actual_exit': 0, 'install_actual_exit': 0,
        'installed_and_python_files_rehashed': len(expected), 'native_binary_inventory_rehashed': len(manifest['binaries']),
        'actual_CXX_SYCL_host_link': 'icpx', 'actual_C': 'icx', 'GPU_started': False,
        'normal_import_verified': False, 'optimized_model51_route_verified': False,
        'full_precision_performance_acceptance_complete': False,
    }
    with (artifact / 'COORDINATOR_FULL_ICPX_AUDIT_v5.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
