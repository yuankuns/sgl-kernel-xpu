import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path('/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017')
HISTORICAL = Path('/root/worktrees/sgl-kernel-xpu/review-validation-artifacts/accuracy-20261016')
PACKAGES = {
    'memory': HISTORICAL / 'memory-scope-correction/install/sgl_kernel',
    'position': HISTORICAL / 'fmha-logits-position/scope-correction/muse_position/installed-package-before-final',
    'logits-vocab': HISTORICAL / 'fmha-logits-position/scope-correction/muse_logits_vocab/installed-package-before-final',
}


def main():
    component = sys.argv[1]
    artifact = ROOT / component
    original = PACKAGES[component]
    if not list(original.glob('common_ops*.so')):
        raise RuntimeError(f'missing real source package extension: {original}')
    spec = importlib.util.spec_from_file_location('simple_metadata', ROOT / 'port_simple.py')
    metadata = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metadata)
    metadata.SOURCE = Path('/root/sgl-kernel-xpu')
    info = metadata.SPECS[component]
    manifests = {}
    for label, build_name in [('before', 'source-baseline-build'), ('after', 'native-diagnostic-build')]:
        build = artifact / build_name
        stage = artifact / 'diagnostic' / label / 'site' / 'sgl_kernel'
        shutil.copytree(original, stage, dirs_exist_ok=True)
        libraries = list(build.glob('libsgl-ops-sycl-*.so'))
        if not libraries:
            raise RuntimeError(f'no newly compiled test libraries: {build}')
        for library in libraries:
            if not (stage / library.name).is_file():
                raise RuntimeError(f'library absent from original normal package: {library.name}')
            shutil.copy2(library, stage / library.name)
        manifests[label] = {
            'registration_package': str(original),
            'registration_extension_sha256': {
                filename.name: hashlib.sha256(filename.read_bytes()).hexdigest()
                for filename in stage.glob('common_ops*.so')
            },
            'fresh_native_libraries': {
                library.name: hashlib.sha256((stage / library.name).read_bytes()).hexdigest()
                for library in libraries
            },
            'package_files': {
                str(filename.relative_to(stage)): hashlib.sha256(filename.read_bytes()).hexdigest()
                for filename in stage.rglob('*') if filename.is_file() and '__pycache__' not in filename.parts
            },
        }
    port = Path('/root/worktrees/sgl-kernel-xpu-up/pr') / ('muse-' + component + '-xe35')
    report = {
        'scope': 'source-to-port kernel diagnostic with real original production package; not official-package integration acceptance',
        'source_revision': metadata.git_text('rev-parse', info['branch']).strip(),
        'upstream_base': subprocess.check_output(
            ['git', '-c', 'safe.directory=/root/sgl-kernel-xpu-up', '-C', '/root/sgl-kernel-xpu-up',
             'rev-parse', 'pr/muse-' + component + '-xe35'], text=True
        ).strip(),
        'port_sources_sha256': {
            filename: hashlib.sha256((port / filename).read_bytes()).hexdigest()
            for filename in info['files'].values() if filename.endswith('.cpp')
        },
        'baseline_sources_sha256': {
            str(filename.relative_to(artifact)): hashlib.sha256(filename.read_bytes()).hexdigest()
            for folder in (artifact / 'source-baseline', artifact / 'native-diagnostic')
            for filename in folder.rglob('*') if filename.is_file()
        },
        'packages': manifests,
    }
    (artifact / 'diagnostic-source-binary-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(component, 'real source package staged with fresh native kernels for both sides', flush=True)


if __name__ == '__main__':
    main()
