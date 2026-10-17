import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import zipfile

from check_canonical_package import digest


REVISIONS = {
    'mxfp4-w4a4': '54216962c83b2d489702fd38c58d71116f3a6969',
    'routing-quant': '4cb5e929f84ebcebf98ec6702a26f1ee93190828',
}
OPS = {
    'mxfp4-w4a4': ['moe_grouped_mm_nt_xe35_mxfp4_w4a4'],
    'routing-quant': ['muse_spark_moe_routing', 'sgl_per_token_group_quant_fp4', 'sgl_padded_gather_per_token_group_quant_fp4', 'topk_sigmoid'],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', choices=tuple(REVISIONS), required=True)
    parser.add_argument('--version', default='canonical-v4-icpx')
    args = parser.parse_args()
    if not re.fullmatch(r'canonical-v[1-9][0-9]*(?:-icpx)?', args.version):
        parser.error('Expected canonical-vN or canonical-vN-icpx version')
    artifact = pathlib.Path('/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017') / args.module
    if pathlib.Path(sys.prefix) != artifact / 'source-venv':
        raise RuntimeError('Baseline must use the real original source environment')
    if (artifact / 'source-package.exit').read_text().strip() != '0':
        raise RuntimeError('Original complete source package did not succeed')
    source = pathlib.Path('/root/worktrees/sgl-kernel-xpu') / ('muse-spark-mxfp4-w4a4-main-20261014' if args.module == 'mxfp4-w4a4' else 'muse-spark-mxfp4-routing-quant-main-20261014')
    head = subprocess.check_output(['git', '-c', 'safe.directory=' + str(source), '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if head != REVISIONS[args.module]:
        raise RuntimeError('Baseline original source revision changed')
    provenance = json.loads((artifact / 'source-package-provenance.json').read_text())
    wheel = artifact / 'source-wheels/sgl_kernel-0.11.0-cp39-abi3-linux_x86_64.whl'
    if digest(wheel) != provenance['wheel_sha256']:
        raise RuntimeError('Original source wheel hash changed')
    for name, expected in provenance['source_native_hashes'].items():
        if digest(source / name) != expected:
            raise RuntimeError(f'Original source native input changed: {name}')
    import torch
    import sgl_kernel

    package = pathlib.Path(sgl_kernel.__file__).resolve().parent
    if not package.is_relative_to(artifact / 'source-venv') or torch.xpu.is_initialized():
        raise RuntimeError('Not a CPU-only import of the original source package')
    verified = {}
    with zipfile.ZipFile(wheel) as archived:
        for entry in archived.infolist():
            if not entry.filename.startswith('sgl_kernel/') or entry.is_dir():
                continue
            path = package.parent / entry.filename
            checksum = hashlib.sha256()
            with archived.open(entry) as stream:
                while chunk := stream.read(1024 * 1024):
                    checksum.update(chunk)
            expected = checksum.hexdigest()
            if path.is_symlink() or not path.is_file() or digest(path) != expected:
                raise RuntimeError(f'Original installed package differs from source wheel: {entry.filename}')
            verified[str(path)] = expected
    actual = {str(path) for path in package.rglob('*') if path.is_file() and path.suffix in ('.so', '.py')}
    if actual != set(verified):
        raise RuntimeError('Original package has missing or extraneous Python/native files')
    for name, expected in provenance['source_python_hashes'].items():
        if digest(source / 'python' / name) != expected:
            raise RuntimeError(f'Original source Python changed: {name}')
    for name in OPS[args.module]:
        if not hasattr(sgl_kernel, name) or not torch._C._dispatch_has_kernel_for_dispatch_key('sgl_kernel::' + name, 'XPU'):
            raise RuntimeError(f'Original baseline ABI is missing: {name}')
    maps = pathlib.Path('/proc/self/maps').read_text()
    mapped = sorted({pathlib.Path(line.split()[-1]).resolve() for line in maps.splitlines() if '/' in line.split()[-1] and ('libsgl-ops-' in line or '/common_ops.' in line or '/libmuse-' in line)})
    if not mapped or not any(path.name.startswith('common_ops.') for path in mapped):
        raise RuntimeError('Original source common_ops is not actually mapped')
    for path in mapped:
        if str(path) not in verified or digest(path) != verified[str(path)]:
            raise RuntimeError(f'Mapped baseline DSO is not the archived original source: {path}')
    before_path = artifact / 'source-before-provenance-v3-icpx.json'
    before = json.loads(before_path.read_text())
    if before['source_revision'] != head or before['source_wheel_sha256'] != provenance['wheel_sha256']:
        raise RuntimeError('Original BEFORE build-command provenance revision/wheel mismatch')
    for native in before['native_builds']:
        if digest(pathlib.Path(native['before_installed_native'])) != native['native_sha256']:
            raise RuntimeError('Original BEFORE native replaced after compiler/flags audit')
        for settings in native['generated_compile_settings']:
            if digest(pathlib.Path(settings['path'])) != settings['sha256']:
                raise RuntimeError('Original source generated compiler/flags evidence changed')
        if digest(pathlib.Path(native['ninja_command_output'])) != native['ninja_command_output_sha256']:
            raise RuntimeError('Original source raw Ninja command evidence changed')
    report = {'module': args.module, 'version': args.version, 'source_revision': head, 'package': str(package), 'source_wheel_sha256': digest(wheel), 'verified_package_file_count': len(verified), 'mapped_native_sha256': {str(path): verified[str(path)] for path in mapped}, 'kind': 'real complete original source package; no diagnostic ABI forward/native replacement', 'historical_source_provenance_limits': provenance.get('limits', provenance.get('reuse')), 'xpu_initialized': False, 'accuracy': 'not run', 'performance': 'not run'}
    report['before_build_command_provenance_sha256'] = digest(before_path)
    report['before_compiler_policy'] = 'Original native SYCL icx (quant monitor forwards icx), GNU host; from genuine archived commands, not Torch family token; unchanged baseline, not claimed matched-icpx source rebuild'
    output = artifact / args.version / 'source-baseline-check.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    output.chmod(0o644)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
