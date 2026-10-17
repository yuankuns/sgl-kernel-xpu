import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import time

from overlay_snapshot import OFFICIAL_BASE, digest_file, file_record, git_output, manifest_digest, separate_paths, verify_inventory, write_json


DEFAULT_FEATURES = {'USE_MOE': 'ON', 'USE_FMHA': 'ON', 'USE_GDN': 'ON', 'USE_MLA': 'ON', 'USE_QSA': 'ON', 'USE_MLA_SPARSE_FUSED': 'OFF', 'USE_SYCL_JIT': 'OFF'}
CUTLASS_PIN = '7b56b936b87f799f80dde60c2fd8d11c1128cebb'


def mapped_path(value, mappings):
    for original, replacement in mappings:
        if value == original or value.startswith(original + '/'):
            return replacement + value[len(original):]
    return value


def read_cache(path):
    return {key: {'type': kind, 'value': value} for key, kind, value in re.findall(r'^([^/#\n:]+):([^=\n]*)=(.*)$', path.read_text(), re.MULTILINE)}


def current_native_targets(ninja):
    outputs = set()
    for line in ninja.read_text().splitlines():
        if not line.startswith('build '):
            continue
        outputs_text, inputs = line[6:].split(': ', 1)
        if inputs.split()[0] == 'phony':
            continue
        for output in outputs_text.split(' | ', 1)[0].split():
            if output.endswith('.so'):
                outputs.add(pathlib.PurePosixPath(output).name)
    return outputs


def stage(arguments):
    build = pathlib.Path(arguments.build_dir).resolve(strict=True)
    snapshot = pathlib.Path(arguments.snapshot_dir).resolve(strict=True)
    site_input = pathlib.Path(arguments.site).absolute()
    if site_input.exists() or site_input.is_symlink():
        raise RuntimeError('Site must be a completely new path, not an existing or overlaid package')
    site = site_input.resolve()
    manifest = pathlib.Path(arguments.manifest).resolve()
    install_log = pathlib.Path(arguments.install_log).resolve()
    mappings = [entry.split('=', 1) for entry in arguments.path_map]
    if any(len(entry) != 2 or not entry[0].startswith('/') or not entry[1].startswith('/') for entry in mappings):
        raise RuntimeError('Use explicit absolute --path-map OLD=NEW entries')
    separate_paths(snapshot, build)
    for directory in [snapshot, build]:
        separate_paths(directory, site)
    for output in [manifest, install_log]:
        if output.exists() or output == site or site in output.parents or output == build or build in output.parents or output == snapshot or snapshot in output.parents:
            raise RuntimeError('Use new versioned evidence paths outside build/source/site')
    if os.environ.get('DESTDIR'):
        raise RuntimeError('DESTDIR is unsupported; refusing ambiguous staged install paths')
    source_manifest_path = pathlib.Path(arguments.snapshot_manifest).resolve(strict=True)
    source_manifest = json.loads(source_manifest_path.read_text())
    if not source_manifest.get('verified') or source_manifest['source_digest'] != manifest_digest(source_manifest['files']):
        raise RuntimeError('Invalid source snapshot manifest')
    if source_manifest.get('head') != OFFICIAL_BASE or source_manifest.get('official_base') != OFFICIAL_BASE:
        raise RuntimeError('Expected the official 833c952 base snapshot')
    if pathlib.Path(mapped_path(source_manifest['destination'], mappings)).resolve() != snapshot:
        raise RuntimeError('Snapshot manifest path mismatch; provide explicit host/container --path-map if needed')
    origin = pathlib.Path(mapped_path(source_manifest['source'], mappings)).resolve()
    for output in [build, snapshot, site, manifest, install_log]:
        separate_paths(origin, output)
    verify_inventory(snapshot, source_manifest['files'])
    exit_path = pathlib.Path(arguments.build_exit_file).resolve(strict=True)
    if exit_path.read_text().strip() != '0':
        raise RuntimeError('Default all build must have a real successful terminal exit before installation')
    build_command = pathlib.Path(arguments.build_command_file).resolve(strict=True)
    build_log = pathlib.Path(arguments.build_log).resolve(strict=True)
    cache_path = build / 'CMakeCache.txt'
    ninja_path = build / 'build.ninja'
    cache = read_cache(cache_path)
    if pathlib.Path(cache['CMAKE_HOME_DIRECTORY']['value']).resolve() != snapshot:
        raise RuntimeError('Configured CMake source is not this canonical snapshot')
    if pathlib.Path(cache['CMAKE_CACHEFILE_DIR']['value']).resolve() != build:
        raise RuntimeError('CMake build directory identity mismatch')
    if cache.get('DPCPP_SYCL_TARGET', {}).get('value') != 'cri':
        raise RuntimeError('Expected CRI architecture')
    if cache.get('CMAKE_BUILD_TYPE', {}).get('value') != arguments.config:
        raise RuntimeError('Requested install config differs from the configured build type')
    for key, value in DEFAULT_FEATURES.items():
        if cache.get(key, {}).get('value') != value:
            raise RuntimeError(f'Not a default-feature package: {key} != {value}')
    for key in ['CMAKE_PROJECT_INCLUDE', 'CMAKE_PROJECT_INCLUDE_BEFORE', 'CMAKE_PROJECT_TOP_LEVEL_INCLUDES']:
        if cache.get(key, {}).get('value'):
            raise RuntimeError(f'Unreviewed project include hook: {key}')
    cutlass = pathlib.Path(cache['FETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL']['value']).resolve(strict=True)
    pin = git_output(cutlass, 'rev-parse', 'HEAD').decode().strip()
    if pin != CUTLASS_PIN or git_output(cutlass, 'status', '--porcelain').strip():
        raise RuntimeError('Public exact CUTLASS pin must match and be clean')
    targets = current_native_targets(ninja_path)
    if not any(name.startswith('common_ops.') for name in targets):
        raise RuntimeError('Current native target graph has no common_ops')
    before = {'cache_sha256': digest_file(cache_path), 'ninja_sha256': digest_file(ninja_path), 'source_manifest_sha256': digest_file(source_manifest_path)}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    install_log.parent.mkdir(parents=True, exist_ok=True)
    command = [arguments.cmake, '--install', str(build), '--prefix', str(site), '--config', arguments.config]
    with install_log.open('x') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        stream.write(f'\nCANONICAL_INSTALL_ACTUAL_EXIT={result.returncode}\n')
    if result.returncode:
        raise RuntimeError(f'cmake install failed with real exit {result.returncode}; log {install_log}')
    install_manifest = build / 'install_manifest.txt'
    install_text = install_manifest.read_text()
    installed = {}
    for line in install_text.splitlines():
        if not line:
            continue
        path = pathlib.Path(line)
        if not path.is_absolute() or not path.is_file() or site not in path.resolve().parents:
            raise RuntimeError(f'Install manifest escapes fresh site or has a missing file: {line}')
        installed[path.relative_to(site).as_posix()] = file_record(path)
    if not installed:
        raise RuntimeError('Current CMake install manifest is empty')
    native = {pathlib.PurePosixPath(name).name: name for name in installed if name.endswith('.so')}
    if set(native) != targets:
        raise RuntimeError(f'Current target/install native mismatch: missing={sorted(targets - set(native))}, extra={sorted(set(native) - targets)}')
    python_files = {name: record for name, record in source_manifest['files'].items() if name.startswith('python/sgl_kernel/')}
    if 'python/sgl_kernel/__init__.py' not in python_files:
        raise RuntimeError('Snapshot has no actual sgl_kernel Python package')
    copied_python = {}
    for name, record in python_files.items():
        relative = pathlib.PurePosixPath(name).relative_to('python').as_posix()
        target = site / relative
        if target.exists() and file_record(target) != record:
            raise RuntimeError(f'Python source would overwrite a different installed file: {relative}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snapshot / name, target)
        os.chmod(target, record['mode'])
        if file_record(target) != record:
            raise RuntimeError(f'Copied Python differs from current snapshot: {relative}')
        copied_python[relative] = record
    verify_inventory(snapshot, source_manifest['files'])
    verify_inventory(site, installed | copied_python)
    if before != {'cache_sha256': digest_file(cache_path), 'ninja_sha256': digest_file(ninja_path), 'source_manifest_sha256': digest_file(source_manifest_path)}:
        raise RuntimeError('Build configuration or snapshot evidence changed during installation')
    common = [name for name in native if name.startswith('common_ops.')]
    if len(common) != 1:
        raise RuntimeError('Expected exactly one current common_ops binary')
    dynamic = subprocess.check_output([arguments.readelf, '-d', str(site / native[common[0]])], text=True)
    needed = re.findall(r'\(NEEDED\).*\[([^\]]+)\]', dynamic)
    missing_needed = [name for name in needed if name.startswith('libsgl-ops-') and name not in native]
    if missing_needed:
        raise RuntimeError(f'common_ops needs unstaged native libraries: {missing_needed}')
    payload = {'schema': 1, 'created_epoch': time.time(), 'site': str(site), 'build': str(build), 'snapshot': str(snapshot), 'source': source_manifest, 'source_manifest_file_sha256': before['source_manifest_sha256'], 'flags': cache, 'cache_sha256': before['cache_sha256'], 'ninja_sha256': before['ninja_sha256'], 'cutlass_pin': pin, 'default_features': DEFAULT_FEATURES, 'current_native_targets': sorted(targets), 'cmake_install_command': command, 'install_actual_exit': result.returncode, 'install_log_sha256': digest_file(install_log), 'install_manifest_text': install_text, 'install_manifest_sha256': digest_file(install_manifest), 'installed_files': installed, 'python_snapshot_files': copied_python, 'binaries': {name: {'path': native[name], **installed[native[name]]} for name in sorted(native)}, 'common_ops': native[common[0]], 'common_ops_dt_needed': needed, 'build_evidence': {'exit': 0, 'exit_file': str(exit_path), 'exit_file_sha256': digest_file(exit_path), 'command_file': str(build_command), 'command_text': build_command.read_text(), 'command_sha256': digest_file(build_command), 'log_file': str(build_log), 'log_sha256': digest_file(build_log)}, 'formal_checker': {'default_all_command_verified': False, 'normal_package_import_verified': False, 'all_runtime_mapped_dso_verified': False, 'accuracy_verified': False, 'performance_verified': False}, 'acceptance_complete': False}
    write_json(manifest, payload)
    print(json.dumps({'site': str(site), 'manifest': str(manifest), 'native_count': len(native), 'python_count': len(copied_python), 'formal_checker_pending': True}))


def main():
    parser = argparse.ArgumentParser(description='Fresh current-graph cmake install plus actual snapshot Python; never configure/build/import Torch')
    for name in ['build-dir', 'snapshot-dir', 'snapshot-manifest', 'site', 'manifest', 'install-log', 'build-exit-file', 'build-command-file', 'build-log']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--path-map', action='append', default=[])
    parser.add_argument('--cmake', default='cmake')
    parser.add_argument('--readelf', default='readelf')
    parser.add_argument('--config', default='Release')
    stage(parser.parse_args())


if __name__ == '__main__':
    main()
