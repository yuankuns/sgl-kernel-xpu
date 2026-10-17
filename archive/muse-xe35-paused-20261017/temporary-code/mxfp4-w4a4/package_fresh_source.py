import argparse
import hashlib
import json
import pathlib
import subprocess
import zipfile

from wheel.wheelfile import WheelFile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(artifact):
    source = pathlib.Path('/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014')
    dependency = pathlib.Path('/root/worktrees/cutlass_internal/muse-w4a4-e87870b')
    git_dir = (source / '.git').read_text().strip().removeprefix('gitdir: ').replace('/home/syk/', '/root/')
    command = ['git', '--git-dir=' + git_dir, '--work-tree=' + str(source)]
    files = [source / 'src/sycl/xe35/moe_grouped_mm_mxfp4_w4a4.cpp']
    files += sorted((source / 'src/sycl/kernels/moe/xe35').rglob('*.hpp'))
    files += sorted((source / 'src/sycl/kernels/moe/xe35').rglob('*.h'))
    hashes = {}
    for path in files:
        name = str(path.relative_to(source))
        committed = subprocess.check_output(command + ['show', 'HEAD:' + name])
        actual = digest(path)
        if actual != hashlib.sha256(committed).hexdigest():
            raise RuntimeError('Original source body changed: ' + name)
        hashes[name] = actual
    compiler = pathlib.Path('/opt/gfx-deps/dpcpp/bin/icx')
    return {
        'kind': 'Fresh original W4 native only in complete real original package; diagnostic, not official full build',
        'container': 'codex-muse-bf16-review-20261015',
        'source_head': subprocess.check_output(command + ['rev-parse', 'HEAD'], text=True).strip(),
        'original_body_sha256_matches_git_HEAD': hashes,
        'dependency_head': subprocess.check_output(['git', '-c', 'safe.directory=' + str(dependency), '-C', str(dependency), 'rev-parse', 'HEAD'], text=True).strip(),
        'dependency_dirty': subprocess.check_output(['git', '-c', 'safe.directory=' + str(dependency), '-C', str(dependency), 'status', '--porcelain'], text=True).strip(),
        'dependency_headers_sha256': {str(path.relative_to(dependency)): digest(path) for path in sorted((dependency / 'include').rglob('*')) if path.is_file()},
        'compiler': {'path': str(compiler), 'sha256': digest(compiler.resolve()), 'version': subprocess.check_output([str(compiler), '--version'], text=True)},
        'resources': {'cpus': '64-79', 'lane': 4, 'jobs': 2, 'inner_device_link': 1, 'file_limit_gib': 8, 'features_disabled': False},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot-only', action='store_true')
    args = parser.parse_args()
    artifact = pathlib.Path(__file__).resolve().parent
    report = snapshot(artifact)
    before = artifact / 'fresh-source-input-provenance.json'
    if args.snapshot_only:
        before.write_text(json.dumps(report, indent=2) + '\n')
        print('Original body and dependency/header/compiler SHA256 recorded; no GPU work.')
        return
    original = json.loads(before.read_text())
    for key in ('source_head', 'original_body_sha256_matches_git_HEAD', 'dependency_head', 'dependency_dirty', 'dependency_headers_sha256', 'compiler'):
        if report[key] != original[key]:
            raise RuntimeError('Source compilation input changed: ' + key)
    if report['dependency_dirty']:
        raise RuntimeError('Original private source pin not clean')
    source_wheel = next((artifact / 'source-wheels').glob('*.whl'))
    source_provenance = json.loads((artifact / 'source-package-provenance.json').read_text())
    if digest(source_wheel) != source_provenance['wheel_sha256']:
        raise RuntimeError('Complete original source wheel hash mismatch')
    native_name = 'libsgl-ops-sycl-moe_grouped_mm_mxfp4_w4a4.so'
    native = artifact / 'fresh-source-build/src' / native_name
    wheel = artifact / 'fresh-source-wheels' / source_wheel.name
    wheel.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(source_wheel) as original_package, WheelFile(wheel, 'w') as package:
        for name in original_package.namelist():
            if name.endswith('/RECORD') or pathlib.Path(name).name == native_name:
                continue
            package.writestr(name, original_package.read(name))
        package.write(native, 'sgl_kernel/' + native_name)
    settings = next((artifact / 'fresh-source-build/src').glob('CMakeFiles/sgl-ops-sycl-moe_grouped_mm_mxfp4_w4a4.dir/sycl/xe35/*Release.cmake'))
    report.update({'fresh_native_sha256': digest(native), 'original_source_wheel_sha256': digest(source_wheel), 'fresh_diagnostic_wheel_sha256': digest(wheel), 'fresh_compile_settings_path': str(settings), 'fresh_compile_settings_sha256': digest(settings), 'preservation': 'Every original Python/common_ops and other native file retained byte-for-byte; only original W4 native freshly compiled without body/API changes. No fake registrations, forwards or load_library needed.'})
    (artifact / 'fresh-source-package-provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('kind', 'fresh_native_sha256', 'fresh_diagnostic_wheel_sha256', 'preservation')}, indent=2))


if __name__ == '__main__':
    main()
