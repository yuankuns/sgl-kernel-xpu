import hashlib
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
COMPONENTS = ('bf16-gemm', 'fmha', 'qk-norm-rope', 'mxfp4-w4a4', 'residual-norm',
              'nvfp4-woq', 'moe-permutation', 'routing-quant', 'memory', 'position', 'logits-vocab')


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    reports = {}
    for component in COMPONENTS:
        version_number = 5 if component in ('fmha', 'nvfp4-woq') else 4
        artifact = ROOT / component
        stage = artifact / f'canonical-v{version_number}-icpx'
        manifest_path = stage / 'source-binary-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        frozen = json.loads((artifact / f'canonical-source-freeze-v{version_number}-icpx.json').read_text())
        snapshot = json.loads((stage / 'source-snapshot.json').read_text())
        assert frozen['source_digest'] == snapshot['source_digest'] == manifest['source']['source_digest']
        assert frozen['files'] == snapshot['files']
        assert manifest['install_actual_exit'] == manifest['build_evidence']['exit'] == 0
        for phase in ('configure', 'default-all-build', 'supervisor'):
            assert (stage / f'{phase}.exit').read_text().strip() == '0'
        assert json.loads((stage / 'source-readonly-postverify.json').read_text())['verified']
        policy_path = stage / 'compiler-policy-verified.json'
        policy = json.loads(policy_path.read_text())
        assert policy['compiler_id'] == 'IntelLLVM' and policy['no_GNU_command'] is True
        assert all(policy[name] == '/opt/gfx-deps/dpcpp/bin/icpx' for name in ('cxx', 'sycl', 'host'))
        assert policy['c'] == '/opt/gfx-deps/dpcpp/bin/icx'
        assert manifest['cutlass_pin'] == '7b56b936b87f799f80dde60c2fd8d11c1128cebb'
        assert digest(stage / 'CMakeCache.txt') == manifest['cache_sha256']
        assert digest(stage / 'build.ninja') == manifest['ninja_sha256']
        expected = manifest['installed_files'] | manifest['python_snapshot_files']
        site = Path(manifest['site'])
        for relative, record in expected.items():
            assert digest(site / relative) == record['sha256'], (component, relative)
        for record in manifest['binaries'].values():
            assert digest(site / record['path']) == record['sha256']
        reports[component] = {'version': f'canonical-v{version_number}-icpx',
                              'package_manifest_sha256': digest(manifest_path),
                              'source_digest': frozen['source_digest'], 'compiler_policy_sha256': digest(policy_path),
                              'files_rehashed': len(expected), 'native_binaries_rehashed': len(manifest['binaries']),
                              'default_all_install_post_supervisor_exit': 0,
                              'runtime_precision_performance_acceptance_implied': False}
    report = {'actual_exit': 0, 'epoch': time.time(), 'components': reports, 'GPU_started': False,
              'current_normal_AFTER_full_build_inventory_verified': True,
              'SOURCE_all_tested_helpers_current_native_verified': False,
              'eight_GPU_parallel_runtime_verified': False, 'full_task_acceptance_complete': False,
              'scope': 'Current genuine full AFTER packages; FM and NV v5, other nine v4. Not SOURCE, dispatch, runtime or performance acceptance.'}
    with (ROOT / 'CURRENT_NORMAL_ICPX_PACKAGE_AUDIT_v5.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'components': len(reports),
                      'files_rehashed': sum(record['files_rehashed'] for record in reports.values()),
                      'native_binaries_rehashed': sum(record['native_binaries_rehashed'] for record in reports.values()),
                      'GPU_started': False}))


if __name__ == '__main__':
    main()
