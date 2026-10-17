import argparse
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import time


ROOT = Path(__file__).resolve().parent
V1_DIGEST = 'a0cb1c6078fab4ed1f6096bdbe83d53ed41a2a0646635f1efe7e63474858ed91'
DEFAULT_FEATURES = {'USE_MOE': 'ON', 'USE_FMHA': 'ON', 'USE_GDN': 'ON', 'USE_MLA': 'ON',
                    'USE_QSA': 'ON', 'USE_MLA_SPARSE_FUSED': 'OFF', 'USE_SYCL_JIT': 'OFF'}
CXX = '/opt/gfx-deps/dpcpp/bin/icpx'
CC = '/opt/gfx-deps/dpcpp/bin/icx'
REQUIRED_FLAGS = ['-fhonor-nans', '-fhonor-infinities', '-fno-associative-math',
                  '-fno-approx-func', '-no-ftz', '-fsycl-targets=spir64_gen']


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
    elif str(ROOT).startswith('/home/syk/') and value.startswith('/root/'):
        value = '/home/syk/' + value.removeprefix('/root/')
    return Path(value)


def scope():
    for name in ('BATCHES', 'BATCH', 'SERVING_BATCH', 'BATCH_SIZE', 'SGLANG_BATCH_SIZE',
                 'SGLANG_MUSE_BATCH_SIZE', 'MUSE_BATCH_SIZE', 'SERVING_BATCH_SIZE', 'MUSE_BS', 'BS'):
        if os.environ.get(name, '8') != '8':
            raise RuntimeError(f'{name} must be 8 before GPU work')
    for name, expected in (('ISL', '4096'), ('OSL', '16'), ('TP', '1'), ('EP', '1')):
        if os.environ.get(name, expected) != expected:
            raise RuntimeError(f'{name} must be {expected} before GPU work')


def before_provenance():
    job = ROOT / 'source-before-icpx-bounded-v1'
    record = json.loads((job / 'native-package-manifest.json').read_text())
    assert record['source_ref'] == 'ec29f7a6a14c4f1fac42fb93526236ca03ab894b'
    assert record['cutlass']['pin'] == 'e87870bb7aacf83bfb3be9da685f148316b8df23' and record['cutlass']['clean']
    assert record['native_compiler_id'] == 'IntelLLVM'
    assert all(record['native_compilers'][name] == CXX for name in ['cxx', 'sycl', 'host', 'AOT_link'])
    assert record['native_compilers']['c'] == CC
    original = json.loads((ROOT / 'source-compiler-provenance.json').read_text())
    assert record['original_helpers_sha256'] == original['original_helpers_sha256']
    assert record['source_kernel_sha256'] == original['source_kernel_sha256']
    assert record['registration_DSO_sha256'] == original['common_ops_sha256']
    assert record['historical_GNU_registration_DSO_rebuilt'] is False
    for name in ['supervisor', 'native-build', 'normal-import']:
        assert (job / f'{name}.exit').read_text().strip() == '0'
    site = mapped(record['prefix']) / 'lib/python3.12/site-packages/sgl_kernel'
    expected = json.loads((job / 'site-sha.json').read_text())['post']
    actual = {path.relative_to(site).as_posix(): digest(path) for path in site.rglob('*') if path.is_file()}
    assert actual == expected
    assert actual['libsgl-ops-sycl-FusedNormResidual.so'] == record['tested_native_sha256']
    assert record['tested_native_sha256'] != original['source_native_sha256']
    assert mapped(record['python']).is_file()
    return record


def compare_fresh_accuracy(version):
    import torch
    directory = ROOT / f'canonical-validation-{version.removeprefix("canonical-")}'
    namespace = runpy.run_path(str(ROOT / 'compare_accuracy.py'))
    source = namespace['cases_xe35'](directory / 'source_accuracy.log')
    port = namespace['cases_xe35'](directory / 'port_accuracy.log')
    assert len(source) == len(port) == 24 and source.keys() == port.keys()
    for key in source:
        exact = all(source[key]['accuracy'][name] == port[key]['accuracy'][name]
                    for name in ['norm_sha256', 'stream_sha256'])
        if not exact:
            name = f'{key[0]}_{key[1]}_{int(key[2])}_{int(key[3])}.pt'
            source_outputs = torch.load(directory / 'source-outputs' / name, map_location='cpu')
            port_outputs = torch.load(directory / 'port-outputs' / name, map_location='cpu')
            torch.testing.assert_close(port_outputs['norm'].float(), source_outputs['norm'].float(), rtol=5e-3, atol=5e-3)
            torch.testing.assert_close(port_outputs['stream'], source_outputs['stream'], rtol=1e-5, atol=1e-5)
        print(json.dumps({'case': key, 'bit_exact_fresh_source_to_port': exact,
                          'source': source[key]['accuracy'], 'port': port[key]['accuracy']}), flush=True)


def compiler_policy(version, manifest, probe=False):
    evidence = ROOT / version
    policy_path = evidence / 'compiler-policy-verified.json'
    policy = json.loads(policy_path.read_text())
    assert policy['compiler_id'] == 'IntelLLVM'
    assert policy['cxx'] == policy['sycl'] == policy['host'] == CXX
    assert policy['c'] == CC and policy['no_GNU_command'] is True
    assert policy['torch_SYCL_family'] == 'icx'
    assert set(REQUIRED_FLAGS).issubset(policy['required_flags'])
    for name, expected in {'CMAKE_CXX_COMPILER': CXX, 'CMAKE_C_COMPILER': CC,
                           'SYCL_COMPILER': CXX, 'SYCL_HOST_COMPILER': CXX}.items():
        assert manifest['flags'][name]['value'] == expected, name
    assert digest(evidence / 'CMakeCache.txt') == manifest['cache_sha256']
    assert digest(evidence / 'build.ninja') == manifest['ninja_sha256']
    configure = shlex.split((evidence / 'configure.command').read_text())
    for name, expected in {'CMAKE_CXX_COMPILER': CXX, 'CMAKE_C_COMPILER': CC,
                           'SYCL_COMPILER': CXX, 'SYCL_HOST_COMPILER': CXX}.items():
        assert f'-D{name}={expected}' in configure, name
    assert (evidence / 'configure.exit').read_text().strip() == '0'
    assert '-DXPU_SYCL_COMPILER=icx' in configure
    ninja = (evidence / 'build.ninja').read_text()
    for forbidden in ['/usr/bin/c++', '/usr/bin/g++', '/usr/bin/gcc', 'fsycl-host-compiler=gcc']:
        assert forbidden not in ninja and forbidden not in ' '.join(configure), forbidden
    assert not re.search(r'-D__INTEL_LLVM_COMPILER(?:=|\s|$)', ninja + '\n' + ' '.join(configure))
    build = mapped(manifest['build'])
    pool = {'canonical-v3-icpx': 'canonical-icpx-pool', 'canonical-v4-icpx': 'canonical-icpx-pool-v4'}[version]
    assert build.resolve() == (ROOT.parent / pool / 'build').resolve()
    identities = list((build / 'CMakeFiles').glob('*/CMakeCXXCompiler.cmake'))
    assert len(identities) == 1
    assert 'set(CMAKE_CXX_COMPILER_ID "IntelLLVM")' in identities[0].read_text()
    if probe and str(ROOT).startswith('/root/'):
        for driver in [CXX, CC]:
            output = subprocess.check_output([driver, '--version'], text=True)
            assert 'Intel' in output and 'oneAPI' in output, (driver, output)
        macros = subprocess.run([CXX, '-dM', '-E', '-x', 'c++', '-'], input='', text=True,
                                capture_output=True, check=True).stdout
        assert '#define __INTEL_LLVM_COMPILER ' in macros
        print(json.dumps({'real_driver_probe': True, 'compiler_policy_sha256': digest(policy_path),
                          'compiler_identity_sha256': digest(identities[0]), 'cxx': CXX, 'c': CC,
                          'cxx_binary_sha256': digest(Path(CXX)), 'c_binary_sha256': digest(Path(CC))}), flush=True)
    return digest(policy_path)


def provenance(version, driver_probe=False):
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
    evidence_root = ROOT / version
    assert (evidence_root / 'supervisor.exit').read_text().strip() == '0'
    post = json.loads((evidence_root / 'source-readonly-postverify.json').read_text())
    assert post['verified'] is True and mapped(post['source']).resolve() == source.resolve()
    snapshot = json.loads((evidence_root / 'source-snapshot.json').read_text())
    assert snapshot['source_digest'] == expected_digest and snapshot['files'] == freeze['files']
    assert digest(evidence_root / 'source-snapshot.json') == manifest['source_manifest_file_sha256']
    assert digest(evidence_root / 'install.log') == manifest['install_log_sha256']
    completed = json.loads((evidence_root / 'build-stage-complete.json').read_text())
    assert completed['default_all_exit'] == 0 and completed['fresh_current_graph_install'] is True
    assert completed['source_digest'] == expected_digest
    assert post['epoch'] >= completed['epoch']
    compiler_policy(version, manifest, probe=driver_probe)
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
    parser.add_argument('--build-version', choices=['canonical-v3-icpx', 'canonical-v4-icpx'], default='canonical-v4-icpx')
    parser.add_argument('--mode', choices=['before-provenance', 'provenance', 'package', 'accuracy', 'compare', 'perf'], default='package')
    parser.add_argument('--output-dir', type=Path)
    arguments = parser.parse_args()
    if arguments.mode == 'before-provenance':
        record = before_provenance()
        print(json.dumps({'fresh_before_CPU_gates': True, 'native_sha256': record['tested_native_sha256'],
                          'GNU_registration_rebuilt': False, 'GPU_started': False}), flush=True)
        return
    source, stage, manifest, expected, manifest_hash = provenance(arguments.build_version, driver_probe=arguments.mode == 'provenance')
    policy_hash = digest(ROOT / arguments.build_version / 'compiler-policy-verified.json')
    if arguments.mode == 'provenance':
        before_provenance()
        print(json.dumps({'mode': 'CPU_only_provenance', 'build_version': arguments.build_version,
                          'manifest_sha256': manifest_hash, 'GPU_started': False}), flush=True)
        return
    if arguments.mode == 'compare':
        before_provenance()
        compare_fresh_accuracy(arguments.build_version)
        return
    assert (ROOT / f'{arguments.build_version}_CPU_preflight.exit').read_text().strip() == '0'
    preflight = [json.loads(line) for line in (ROOT / f'{arguments.build_version}_CPU_preflight.log').read_text().splitlines()
                 if line.startswith('{')]
    assert any(record.get('real_driver_probe') is True and record['compiler_policy_sha256'] == policy_hash
               and record['cxx_binary_sha256'] == digest(Path(CXX)) and record['c_binary_sha256'] == digest(Path(CC))
               for record in preflight)
    assert any(record.get('mode') == 'CPU_only_provenance' and record['manifest_sha256'] == manifest_hash
               for record in preflight)
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
    compiler_policy(arguments.build_version, manifest)
    assert digest(ROOT / arguments.build_version / 'compiler-policy-verified.json') == policy_hash
    assert digest(ROOT / arguments.build_version / 'source-binary-manifest.json') == manifest_hash


if __name__ == '__main__':
    main()
