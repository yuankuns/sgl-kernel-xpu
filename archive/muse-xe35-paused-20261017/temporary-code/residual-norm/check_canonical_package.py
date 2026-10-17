import argparse
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import runpy
import shlex
import time


ROOT = Path(__file__).resolve().parent
V1_DIGEST = 'a0cb1c6078fab4ed1f6096bdbe83d53ed41a2a0646635f1efe7e63474858ed91'
DEFAULT_FEATURES = {'USE_MOE': 'ON', 'USE_FMHA': 'ON', 'USE_GDN': 'ON', 'USE_MLA': 'ON',
                    'USE_QSA': 'ON', 'USE_MLA_SPARSE_FUSED': 'OFF', 'USE_SYCL_JIT': 'OFF'}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def mapped(path):
    value = str(path)
    if str(ROOT).startswith('/root/') and value.startswith('/home/syk/'):
        value = '/root/' + value.removeprefix('/home/syk/')
    return Path(value)


def scope():
    for name in ('BATCHES', 'BATCH', 'SERVING_BATCH', 'BATCH_SIZE', 'SGLANG_BATCH_SIZE',
                 'SGLANG_MUSE_BATCH_SIZE', 'MUSE_BATCH_SIZE', 'SERVING_BATCH_SIZE', 'MUSE_BS', 'BS'):
        if os.environ.get(name, '8') != '8':
            raise RuntimeError(f'{name} must be 8 before GPU work')
    for name, expected in (('ISL', '4096'), ('OSL', '16'), ('TP', '1'), ('EP', '1')):
        if os.environ.get(name, expected) != expected:
            raise RuntimeError(f'{name} must be {expected} before GPU work')


def provenance(version):
    freeze = json.loads((ROOT / f'canonical-source-freeze-{version.removeprefix("canonical-")}.json').read_text())
    expected_digest = freeze['source_digest']
    assert freeze['files']
    if version == 'canonical-v1':
        assert expected_digest == V1_DIGEST and len(freeze['files']) == 646
    assert hashlib.sha256(json.dumps(freeze['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest() == expected_digest
    source = mapped(freeze['source'])
    for name, record in freeze['files'].items():
        path = source / name
        assert not path.is_symlink() and digest(path) == record['sha256'], name
        assert path.stat().st_size == record['size']
        assert path.stat().st_mode & 0o777 == record['mode']
    manifest_path = ROOT / version / 'source-binary-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['source']['source_digest'] == expected_digest
    assert manifest['source']['files'] == freeze['files']
    assert manifest['source']['head'] == freeze['head'] == '833c95208dccd2af5628c121222f91186a5b2de7'
    assert manifest['default_features'] == DEFAULT_FEATURES
    for name, value in DEFAULT_FEATURES.items():
        assert manifest['flags'][name]['value'] == value
    assert manifest['flags']['DPCPP_SYCL_TARGET']['value'] == 'cri'
    assert manifest['cutlass_pin'] == '7b56b936b87f799f80dde60c2fd8d11c1128cebb'
    evidence = manifest['build_evidence']
    assert evidence['exit'] == manifest['install_actual_exit'] == 0
    assert shlex.split(evidence['command_text']) == ['cmake', '--build', manifest['build'], '--parallel', '56']
    for field, hash_field in [('exit_file', 'exit_file_sha256'), ('command_file', 'command_sha256'), ('log_file', 'log_sha256')]:
        assert digest(mapped(evidence[field])) == evidence[hash_field], field
    assert mapped(evidence['exit_file']).read_text().strip() == '0'
    stage = mapped(manifest['site'])
    assert stage.resolve() == (ROOT / version / 'site').resolve()
    expected = manifest['installed_files'] | manifest['python_snapshot_files']
    for name, record in expected.items():
        assert digest(stage / name) == record['sha256'], name
    return source, stage, manifest, expected, digest(manifest_path)


def package(stage, manifest, expected):
    import torch
    if not torch.xpu.is_available():
        raise RuntimeError('XPU required before architecture query')
    import sgl_kernel
    from sgl_kernel.utils import is_xe3_arch
    assert is_xe3_arch()
    assert Path(sgl_kernel.__file__).resolve() == (stage / 'sgl_kernel/__init__.py').resolve()
    assert os.environ.get('ZE_AFFINITY_MASK') == '7'
    libraries = {}
    for filename in sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                            if 'libsgl-ops-sycl-' in line or '/common_ops' in line}):
        path = Path(filename).resolve()
        relative = path.relative_to(stage.resolve()).as_posix()
        assert digest(path) == expected[relative]['sha256'], filename
        libraries[relative] = expected[relative]['sha256']
    native = {record['path'] for record in manifest['binaries'].values()}
    required = {manifest['common_ops']} | {record['path'] for name, record in manifest['binaries'].items()
                                                   if name in manifest['common_ops_dt_needed']}
    assert required.issubset(libraries) and set(libraries).issubset(native)
    operator = torch.ops.sgl_kernel.fused_post_norm_residual_pre_norm_xe35
    assert 'fused_post_norm_residual_pre_norm_xe35' in str(operator.default._schema)
    assert callable(sgl_kernel.fused_post_norm_residual_pre_norm_xe35)
    print(json.dumps({'normal_package': str(sgl_kernel.__file__), 'new_op_schema': str(operator.default._schema),
                      'mapped_libraries': libraries, 'default_all_verified': True,
                      'accuracy_verified': False, 'performance_approved': False}), flush=True)
    return torch, sgl_kernel.fused_post_norm_residual_pre_norm_xe35


def main():
    scope()
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-version', choices=['canonical-v1', 'canonical-v2'], default='canonical-v2')
    parser.add_argument('--mode', choices=['package', 'accuracy', 'perf'], default='package')
    parser.add_argument('--output-dir', type=Path)
    arguments = parser.parse_args()
    source, stage, manifest, expected, manifest_hash = provenance(arguments.build_version)
    torch, invoke = package(stage, manifest, expected)
    if arguments.mode != 'package':
        if arguments.mode == 'accuracy' and arguments.output_dir is None:
            raise RuntimeError('Full24 accuracy requires output directory')
        namespace = runpy.run_path(str(source / 'benchmark/bench_fused_norm_residual.py'))
        run_case = namespace['run_case_xe35']
        run_case.__globals__['torch'] = torch
        options = argparse.Namespace(provider='port', check=arguments.mode == 'accuracy',
                                     accuracy_only=arguments.mode == 'accuracy', output_dir=arguments.output_dir,
                                     iterations=256, samples=31, warmup=15, cpu_submit_trace=True)
        for rows, post_mode, round_pre, widen in itertools.product([8, 32768], ['none', 'weight', 'input_scale'], [False, True], [False, True]):
            run_case(options, invoke, rows, post_mode, round_pre, widen)
            torch.xpu.synchronize()
            gc.collect()
            torch.xpu.empty_cache()
            time.sleep(2)
    package(stage, manifest, expected)
    assert digest(ROOT / arguments.build_version / 'source-binary-manifest.json') == manifest_hash


if __name__ == '__main__':
    main()
