import argparse
import hashlib
import json
import pathlib
import re
import subprocess


BASE = '833c95208dccd2af5628c121222f91186a5b2de7'
PIN = '7b56b936b87f799f80dde60c2fd8d11c1128cebb'
FEATURES = {'USE_MOE': 'ON', 'USE_FMHA': 'ON', 'USE_GDN': 'ON', 'USE_MLA': 'ON', 'USE_QSA': 'ON', 'USE_MLA_SPARSE_FUSED': 'OFF', 'USE_SYCL_JIT': 'OFF'}
OPS = {
    'mxfp4-w4a4': ['moe_grouped_mm_nt_mxfp4_w4a4_xe35', 'mxfp4_moe_scale_to_mn_major_xe35'],
    'routing-quant': ['muse_spark_moe_routing_xe35', 'sgl_per_token_group_quant_fp4_xe35', 'sgl_padded_gather_per_token_group_quant_fp4_xe35', 'topk_sigmoid_xe35', 'mxfp4_moe_scale_to_mn_major_xe35'],
}


def verify_icpx_policy(evidence, build, version):
    version_tag = version.removeprefix('canonical-').removesuffix('-icpx')
    pool = 'canonical-icpx-pool' if version_tag == 'v3' else 'canonical-icpx-pool-' + version_tag
    expected_build = pathlib.Path('/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017') / pool / 'build'
    if build != expected_build:
        raise RuntimeError('Not the separate real icpx canonical build pool')
    cache = {}
    for line in (evidence / 'CMakeCache.txt').read_text().splitlines():
        if line and not line.startswith(('#', '//')) and '=' in line and ':' in line.split('=', 1)[0]:
            name, value = line.split('=', 1)
            cache[name.split(':', 1)[0]] = value
    compiler = '/opt/gfx-deps/dpcpp/bin/icpx'
    expected = {'CMAKE_CXX_COMPILER': compiler, 'CMAKE_C_COMPILER': '/opt/gfx-deps/dpcpp/bin/icx', 'SYCL_COMPILER': compiler, 'SYCL_HOST_COMPILER': compiler, 'XPU_SYCL_COMPILER': 'icx'}
    for name, value in expected.items():
        if cache.get(name) != value:
            raise RuntimeError(f'Archived real compiler path/family mismatch: {name}')
    policy_path = evidence / 'compiler-policy-verified.json'
    policy = json.loads(policy_path.read_text())
    expected = {'compiler_id': 'IntelLLVM', 'cxx': compiler, 'c': '/opt/gfx-deps/dpcpp/bin/icx', 'sycl': compiler, 'host': compiler, 'torch_SYCL_family': 'icx', 'no_GNU_command': True}
    for name, value in expected.items():
        if policy.get(name) != value:
            raise RuntimeError(f'Actual configure compiler-policy proof mismatch: {name}')
    required_flags = {'-fhonor-nans', '-fhonor-infinities', '-fno-associative-math', '-fno-approx-func', '-no-ftz', '-fsycl-targets=spir64_gen'}
    if not required_flags.issubset(policy['required_flags']):
        raise RuntimeError('Production precision flags missing from main compiler-policy proof')
    return {'compiler_id': 'IntelLLVM', 'actual_cxx_sycl_host': compiler, 'actual_c': '/opt/gfx-deps/dpcpp/bin/icx', 'torch_family_token': 'icx', 'compiler_policy_sha256': digest(policy_path), 'proof': 'this stage archived compiler cache and main genuine compiler-ID/generated-command verification; not the mutable successor build graph'}


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def container_path(path):
    return pathlib.Path(str(path).replace('/home/syk/', '/root/', 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', choices=tuple(OPS), required=True)
    parser.add_argument('--version', default='canonical-v4-icpx')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'canonical-v[1-9][0-9]*(?:-icpx)?', args.version):
        parser.error('Expected canonical-vN or canonical-vN-icpx version')
    artifact = pathlib.Path('/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017') / args.module
    evidence = artifact / args.version
    site = (evidence / 'site').resolve(strict=True)
    manifest_path = evidence / 'source-binary-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if pathlib.Path(manifest['site']).resolve() != site or manifest['source']['head'] != BASE or manifest['cutlass_pin'] != PIN:
        raise RuntimeError('Canonical package source/base/site/pin mismatch')
    if manifest['default_features'] != FEATURES or manifest['build_evidence']['exit'] != 0 or manifest['install_actual_exit'] != 0:
        raise RuntimeError('Default-all build and current install have not succeeded')
    for filename in ('default-all-build.exit', 'configure.exit', 'supervisor.exit'):
        if (evidence / filename).read_text().strip() != '0':
            raise RuntimeError(f'Canonical true terminal gate failed: {filename}')
    build = pathlib.Path(manifest['build']).resolve(strict=True)
    archived_cache = evidence / 'CMakeCache.txt'
    archived_graph = evidence / 'build.ninja'
    if digest(archived_cache) != manifest['cache_sha256'] or digest(archived_graph) != manifest['ninja_sha256']:
        raise RuntimeError('This stage archived graph differs from its real current-install graph')
    compiler_policy = verify_icpx_policy(evidence, build, args.version) if args.version.endswith('-icpx') else {'kind': 'historical compiler evidence; not eligible for new icpx GPU acceptance'}
    postverify = json.loads((evidence / 'source-readonly-postverify.json').read_text())
    if postverify['verified'] is not True:
        raise RuntimeError('Main complete source-readonly postverify has not succeeded')
    native_targets = set()
    for line in archived_graph.read_text().splitlines():
        if not line.startswith('build '):
            continue
        outputs, inputs = line[6:].split(': ', 1)
        if inputs.split()[0] == 'phony':
            continue
        native_targets.update(pathlib.PurePosixPath(name).name for name in outputs.split(' | ', 1)[0].split() if name.endswith('.so'))
    native_names = set(manifest['binaries'])
    actual_native = {path.name for path in site.rglob('*.so')}
    if native_names != native_targets or native_names != set(manifest['current_native_targets']) or native_names != actual_native:
        raise RuntimeError('Current graph, manifest and actual installed native set differ')
    files = manifest['installed_files'] | manifest['python_snapshot_files']
    actual_files = {path.relative_to(site).as_posix() for path in site.rglob('*') if path.is_file()}
    if actual_files != set(files):
        raise RuntimeError('Fresh normal site has missing or extraneous files')
    for name, record in files.items():
        path = site / name
        if path.is_symlink() or digest(path) != record['sha256'] or path.stat().st_size != record['size']:
            raise RuntimeError(f'Current installed file hash mismatch: {name}')
    frozen = json.loads((artifact / 'port-source-provenance.json').read_text())['source_hashes']
    snapshot_files = manifest['source']['files']
    target = pathlib.Path('/root/worktrees/sgl-kernel-xpu-up/pr') / ('muse-' + args.module + '-xe35')
    complete_freeze = json.loads((artifact / ('canonical-source-freeze-' + args.version.removeprefix('canonical-') + '.json')).read_text())
    if snapshot_files != complete_freeze['files'] or manifest['source']['source_digest'] != complete_freeze['source_digest']:
        raise RuntimeError('Canonical complete source inventory differs from the main frozen PR source')
    for name, record in complete_freeze['files'].items():
        path = target / name
        if path.is_symlink() or digest(path) != record['sha256'] or path.stat().st_size != record['size'] or path.stat().st_mode & 0o777 != record['mode']:
            raise RuntimeError(f'Actual PR source changed after main freeze: {name}')
    for name, expected in frozen.items():
        if snapshot_files[name]['sha256'] != expected or digest(target / name) != expected:
            raise RuntimeError(f'Frozen PR source and canonical snapshot mismatch: {name}')
    for field, hash_field in [('command_file', 'command_sha256'), ('log_file', 'log_sha256'), ('exit_file', 'exit_file_sha256')]:
        if digest(container_path(manifest['build_evidence'][field])) != manifest['build_evidence'][hash_field]:
            raise RuntimeError(f'Full build evidence hash mismatch: {field}')
    command = manifest['build_evidence']['command_text']
    if '--build' not in command or str(build) not in command or '--target' in command and not re.search(r'--target\s+all(?:\s|$)', command):
        raise RuntimeError('Evidence is not a current default-all build command')
    import torch
    import sgl_kernel

    package = pathlib.Path(sgl_kernel.__file__).resolve().parent
    if package != site / 'sgl_kernel':
        raise RuntimeError(f'Not importing canonical normal package: {package}')
    if torch.xpu.is_initialized():
        raise RuntimeError('CPU package checker unexpectedly initialized XPU')
    for name in OPS[args.module]:
        if not hasattr(sgl_kernel, name) or not torch._C._dispatch_has_kernel_for_dispatch_key('sgl_kernel::' + name, 'XPU'):
            raise RuntimeError(f'Missing actual new Python/registered XPU API: {name}')
    maps = pathlib.Path('/proc/self/maps').read_text()
    mapped = sorted({pathlib.Path(line.split()[-1]).resolve() for line in maps.splitlines() if '/' in line.split()[-1] and ('libsgl-ops-' in line or '/common_ops.' in line or '/libmuse-' in line)})
    expected_paths = {site / record['path'] for record in manifest['binaries'].values()}
    if not mapped or any(path not in expected_paths for path in mapped):
        raise RuntimeError('Mapped native library is outside the current canonical install graph')
    if (site / manifest['common_ops']).resolve() not in mapped:
        raise RuntimeError('Actual current common_ops not mapped')
    symbols = ''.join(subprocess.check_output(['nm', '-D', '--defined-only', str(path)], text=True) for path in mapped)
    for name in OPS[args.module]:
        if name not in symbols:
            raise RuntimeError(f'New exported native API not present in mapped graph: {name}')
    report = {'kind': 'AFTER normal canonical default-all incremental package; no diagnostic ABI forwards/native replacement', 'module': args.module, 'version': args.version, 'package': str(package), 'manifest_sha256': digest(manifest_path), 'current_native_count': len(native_names), 'mapped_native_hashes': {str(path): digest(path) for path in mapped}, 'new_ops': OPS[args.module], 'source_digest': manifest['source']['source_digest'], 'default_all_and_fresh_current_install_verified': True, 'normal_package_import_verified': True, 'xpu_initialized': False, 'accuracy': 'not run by CPU checker', 'performance': 'not run by CPU checker', 'independent_clean_build': False}
    report['compiler_policy'] = compiler_policy
    output = pathlib.Path(args.output)
    output.write_text(json.dumps(report, indent=2) + '\n')
    output.chmod(0o644)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
