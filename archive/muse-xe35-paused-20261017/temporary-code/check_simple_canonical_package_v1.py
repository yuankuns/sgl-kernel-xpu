import argparse
import json
import os
import shlex
from pathlib import Path

from canonical_tools_loader_v1 import digest_file


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('component', choices=('memory', 'position', 'logits-vocab'))
    parser.add_argument('label', choices=('before', 'after'))
    parser.add_argument('--build-version', choices=('canonical-v4-icpx',), default='canonical-v4-icpx')
    parser.add_argument('--validation-version', default='canonical-validation-v4-icpx')
    arguments = parser.parse_args()
    artifact = ROOT / arguments.component
    official_root = artifact / arguments.build_version
    policy = json.loads((official_root / 'compiler-policy-verified.json').read_text())
    if policy.get('compiler_id') != 'IntelLLVM' or policy.get('no_GNU_command') is not True:
        raise RuntimeError('Genuine icpx compiler policy required')
    for key in ('cxx', 'sycl', 'host'):
        if policy.get(key) != '/opt/gfx-deps/dpcpp/bin/icpx':
            raise RuntimeError(f'Wrong compiler policy: {key}')
    official = json.loads((official_root / 'source-binary-manifest.json').read_text())
    if official['build_evidence']['exit'] != 0 or official['install_actual_exit'] != 0:
        raise RuntimeError('Current default-all build or fresh installation did not succeed')
    command = shlex.split(official['build_evidence']['command_text'])
    if command != ['cmake', '--build', official['build'], '--parallel', '56']:
        raise RuntimeError('Not the actual authorized default all build command')
    for suffix, hash_key in (('exit_file', 'exit_file_sha256'), ('command_file', 'command_sha256'), ('log_file', 'log_sha256')):
        if digest_file(Path(official['build_evidence'][suffix])) != official['build_evidence'][hash_key]:
            raise RuntimeError(f'Changed full build evidence: {suffix}')
    if arguments.label == 'after':
        stage = official_root / 'site'
        expected = official['installed_files'] | official['python_snapshot_files']
    else:
        comparator = json.loads((artifact / arguments.validation_version / 'source-comparator-manifest.json').read_text())
        if digest_file(Path(comparator['source_native_manifest'])) != comparator['source_native_manifest_sha256']:
            raise RuntimeError('Changed fresh SOURCE tested-native provenance')
        if digest_file(Path(comparator['source_compiler_policy'])) != comparator['source_compiler_policy_sha256']:
            raise RuntimeError('Changed fresh SOURCE icpx compiler policy')
        if digest_file(official_root / 'source-binary-manifest.json') != comparator['after_manifest_sha256']:
            raise RuntimeError('Source comparator belongs to a different official package')
        stage = Path(comparator['stage'])
        expected = comparator['files']
    import sgl_kernel
    import torch

    if Path(sgl_kernel.__file__).resolve() != (stage / 'sgl_kernel/__init__.py').resolve():
        raise RuntimeError('Wrong normal package import')
    if digest_file(Path(sgl_kernel.__file__)) != expected['sgl_kernel/__init__.py']['sha256']:
        raise RuntimeError('Normal package initialization differs from source manifest')
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        raise RuntimeError('Xe35 architecture required')
    affinity = os.environ.get('ZE_AFFINITY_MASK')
    if affinity not in {'4', '5', '6', '7', '8', '9', '10', '11'}:
        raise RuntimeError(f'Unexpected GPU allocation: {affinity}')
    names = {
        'memory': ('paged_alloc_extend', 'paged_alloc_extend_and_write', 'paged_alloc_decode', 'write_req_to_token_pool'),
        'position': ('compute_position', 'compute_position_with_hint'),
        'logits-vocab': ('softcap_logits', 'vocab_mask', 'vocab_parallel_embedding'),
    }[arguments.component]
    schemas = {name + '_xe35': str(getattr(torch.ops.sgl_kernel, name + '_xe35').default._schema) for name in names}
    mapped = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libsgl-ops-sycl-' in line or '/common_ops' in line})
    libraries = {}
    for filename in mapped:
        path = Path(filename).resolve()
        relative = path.relative_to(stage.resolve()).as_posix()
        digest = digest_file(path)
        if expected.get(relative, {}).get('sha256') != digest:
            raise RuntimeError(f'Mapped binary differs from stage: {path}')
        libraries[relative] = digest
    expected_native = {record['path'] for record in official['binaries'].values()}
    required_native = {official['common_ops']}
    required_native.update(record['path'] for name, record in official['binaries'].items() if name in official['common_ops_dt_needed'])
    if not required_native.issubset(libraries) or not set(libraries).issubset(expected_native):
        raise RuntimeError(f'Mapped/current-graph native mismatch: missing={sorted(required_native - set(libraries))}, extra={sorted(set(libraries) - expected_native)}')
    print(json.dumps({'component': arguments.component, 'label': arguments.label, 'gpu': affinity, 'package': sgl_kernel.__file__, 'torch': torch.__version__, 'schemas': schemas, 'mapped_libraries': libraries, 'default_all_verified': True, 'normal_package_import_verified': True, 'all_runtime_mapped_dso_verified': True, 'runtime_accuracy_performance_pending': True}))


if __name__ == '__main__':
    main()
