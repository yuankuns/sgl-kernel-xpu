import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('component', choices=('memory', 'position', 'logits-vocab'))
    parser.add_argument('label', choices=('before', 'after'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite CPU evidence')
    assert set(os.sched_getaffinity(0)) == set(range(112, 128))
    official = ROOT / args.component / 'canonical-v4-icpx'
    manifest_path = official / 'source-binary-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    policy = json.loads((official / 'compiler-policy-verified.json').read_text())
    assert policy['compiler_id'] == 'IntelLLVM' and policy['no_GNU_command'] is True
    assert all(policy[name] == '/opt/gfx-deps/dpcpp/bin/icpx' for name in ('cxx', 'sycl', 'host'))
    assert manifest['build_evidence']['exit'] == manifest['install_actual_exit'] == 0
    comparator_path = ROOT / args.component / 'canonical-validation-v4-icpx/source-comparator-manifest.json'
    comparator = json.loads(comparator_path.read_text())
    assert comparator['after_manifest_sha256'] == digest(manifest_path)
    assert comparator['source_native_manifest_sha256'] == digest(Path(comparator['source_native_manifest']))
    stage = official / 'site' if args.label == 'after' else Path(comparator['stage'])
    files = manifest['installed_files'] | manifest['python_snapshot_files'] if args.label == 'after' else comparator['files']
    for relative, record in files.items():
        assert digest(stage / relative) == record['sha256'], relative
    import sgl_kernel
    import torch

    assert not torch.xpu.is_initialized()
    assert Path(sgl_kernel.__file__).resolve() == (stage / 'sgl_kernel/__init__.py').resolve()
    names = {
        'memory': ('paged_alloc_extend', 'paged_alloc_extend_and_write', 'paged_alloc_decode', 'write_req_to_token_pool'),
        'position': ('compute_position', 'compute_position_with_hint'),
        'logits-vocab': ('softcap_logits', 'vocab_mask', 'vocab_parallel_embedding'),
    }[args.component]
    schemas = {name: str(getattr(torch.ops.sgl_kernel, name + '_xe35').default._schema) for name in names}
    mapped = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
              if 'libsgl-ops-sycl-' in line or '/common_ops' in line}
    libraries = {}
    for filename in mapped:
        path = Path(filename).resolve()
        relative = path.relative_to(stage.resolve()).as_posix()
        assert digest(path) == files[relative]['sha256'], relative
        libraries[relative] = digest(path)
    required = {manifest['common_ops']}
    required.update(record['path'] for name, record in manifest['binaries'].items() if name in manifest['common_ops_dt_needed'])
    assert required <= set(libraries)
    assert not torch.xpu.is_initialized()
    report = {'actual_exit': 0, 'CPU_only': True, 'GPU_initialized': False, 'component': args.component, 'label': args.label,
              'actual_CPU_affinity': sorted(os.sched_getaffinity(0)), 'normal_package': str(stage),
              'package_manifest_sha256': digest(manifest_path), 'source_comparator_manifest_sha256': digest(comparator_path),
              'source_native_manifest_sha256': comparator['source_native_manifest_sha256'],
              'compiler_policy_sha256': digest(official / 'compiler-policy-verified.json'),
              'full_stage_files_rehashed': len(files), 'actual_mapped_native_libraries': libraries, 'actual_schemas': schemas,
              'hardware_identity_or_runtime_performance_verified': False}
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'component': args.component, 'label': args.label,
                      'files': len(files), 'native_mapped': len(libraries), 'GPU_initialized': False}))


if __name__ == '__main__':
    main()
