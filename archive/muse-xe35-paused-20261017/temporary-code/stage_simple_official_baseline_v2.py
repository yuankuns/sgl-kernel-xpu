import hashlib
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    component = sys.argv[1]
    if component not in ('memory', 'position', 'logits-vocab'):
        raise SystemExit('Unknown component')
    artifact = ROOT / component
    official = json.loads((artifact / 'source-binary-manifest.json').read_text())
    source = json.loads((artifact / 'diagnostic-source-binary-manifest.json').read_text())
    expected = source['packages']['before']['fresh_native_libraries']
    libraries = list((artifact / 'source-baseline-build').glob('libsgl-ops-sycl-*.so'))
    actual = {library.name: hashlib.sha256(library.read_bytes()).hexdigest() for library in libraries}
    if actual != expected:
        raise RuntimeError('Baseline libraries differ from the tested original-source native binaries')
    stage = artifact / 'before/site'
    shutil.copytree(artifact / 'after/site', stage, dirs_exist_ok=True)
    for library in libraries:
        destination = stage / 'sgl_kernel' / library.name
        if not destination.is_file():
            raise RuntimeError(f'Official package lacks matching native target: {destination}')
        shutil.copy2(library, destination)
    registration = {
        str(filename.relative_to(stage)): hashlib.sha256(filename.read_bytes()).hexdigest()
        for filename in stage.rglob('common_ops*.so')
    }
    if not registration:
        raise RuntimeError('No actual official registration extension in baseline stage')
    for filename, digest in registration.items():
        if official['stage'][filename] != digest:
            raise RuntimeError('Baseline registration differs from actual official registration')
    manifest = {
        'source_revision': source['source_revision'], 'official_base': official['head'],
        'baseline_method': 'tested original optimized native kernels and native C++ symbol forwards through real new official package registration; source comparator only',
        'original_native_libraries': actual, 'registration': registration,
        'stage': {str(filename.relative_to(stage)): hashlib.sha256(filename.read_bytes()).hexdigest() for filename in stage.rglob('*') if filename.is_file()},
    }
    (artifact / 'baseline-source-binary-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'component': component, 'native_libraries': actual, 'registration': registration}))


if __name__ == '__main__':
    main()
