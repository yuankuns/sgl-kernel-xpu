import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


ART = Path(__file__).resolve().parent
ROOT = ART / 'matched-native-icpx-v1'
ARCHIVE = Path('/root/worktrees/sgl-kernel-xpu/review-validation-artifacts/fmha-before-after-20261017')
TARGETS = ('sgl-ops-sycl-flash_attentionXe35',
           'sgl-ops-sycl-flash_attentionXe35_decode_64_bf16',
           'sgl-ops-sycl-flash_attentionXe35_prefill_64_bf16')


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=('source', 'preopt'), required=True)
    args = parser.parse_args()
    role = args.role
    folder = ROOT / role
    side = 'after' if role == 'source' else 'before'
    assert (folder / 'fresh-native-binaries.json').is_file()
    fresh_record = json.loads((folder / 'fresh-native-binaries.json').read_text())
    terminal = json.loads((folder / fresh_record.get('terminal_file', 'native-build.terminal.json')).read_text())
    assert terminal['exit'] == 0 and terminal['remaining_known_live'] == []
    assert json.loads((folder / 'inputs-prebuild.json').read_text()) == json.loads((folder / 'inputs-postbuild.json').read_text())
    expected = fresh_record['libraries']
    symbols = {}
    dependencies = {}
    recipes = {}
    for target in TARGETS:
        library = f'lib{target}.so'
        fresh = folder / 'build/src' / library
        assert digest(fresh) == expected[library]
        exports = subprocess.check_output(['nm', '-D', '--defined-only', '-C', str(fresh)], text=True)
        original = subprocess.check_output(['nm', '-D', '--defined-only', '-C', str(ARCHIVE / f'site-{side}/sgl_kernel' / library)], text=True)
        symbols[library] = {'fresh_nm_defined': exports, 'original_nm_defined': original,
                            'readelf_dynamic': subprocess.check_output(['readelf', '-d', str(fresh)], text=True)}
        if target == TARGETS[0]:
            original_entry = [line.split(' ', 2)[-1] for line in original.splitlines() if re.search(r'\bmha_fwd\(', line)]
            fresh_entry = [line.split(' ', 2)[-1] for line in exports.splitlines() if re.search(r'\bmha_fwd\(', line)]
            assert len(original_entry) == 1 and fresh_entry == original_entry, 'real original native ABI mismatch; no glue permitted'
        recipes[target] = []
        directory = folder / 'build/src/CMakeFiles' / f'{target}.dir'
        depfiles = list(directory.rglob('*.o.depend'))
        assert depfiles, target
        for depfile in depfiles:
            text = depfile.read_text()
            assert 'SET(SYCL_DEPEND' in text
            paths = re.findall(r'^\s*"([^"\n]+)"\s*$', text, re.MULTILINE)
            assert paths
            for name in paths:
                path = Path(name)
                if not path.is_absolute():
                    path = directory / 'sycl' / path
                path = path.resolve(strict=True)
                dependencies[str(path)] = digest(path)
            recipes[target].append({'path': str(depfile), 'sha256': digest(depfile), 'actual_dependency_recipe': text})
    write(folder / 'actual-native-dependencies.json', {'files_sha256': dependencies, 'compiler_dependency_recipes': recipes,
          'scope': 'actual -MD/-MF compiler input closure, plus full tracked source and full e878 pre/post inventories'})
    write(folder / 'native-abi-symbols.json', symbols)
    site = folder / 'site'
    assert not site.exists()
    records = json.loads((ARCHIVE / f'binary-manifest-{side}.json').read_text())
    package_pre = {}
    package_post = {}
    for record in records:
        original = Path(record['path'])
        relative = original.relative_to(ARCHIVE / f'site-{side}')
        assert digest(original) == record['sha256']
        destination = site / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        package_pre[str(relative)] = record['sha256']
    for library, checksum in expected.items():
        shutil.copy2(folder / 'build/src' / library, site / 'sgl_kernel' / library)
        assert digest(site / 'sgl_kernel' / library) == checksum
    for name in package_pre:
        package_post[name] = digest(site / name)
    changed = [name for name in package_pre if package_pre[name] != package_post[name]]
    assert set(changed) == {f'sgl_kernel/{name}' for name in expected}
    write(folder / 'normal-package-manifest.json', {'role': role, 'source_head': json.loads((folder / 'source-pre.json').read_text())['head'],
          'site': str(site), 'original_manifest_sha256': digest(ARCHIVE / f'binary-manifest-{side}.json'),
          'original_package_files_sha256': package_pre, 'fresh_package_files_sha256': package_post,
          'fresh_tested_native_libraries': expected, 'only_replaced_files': changed,
          'native_compiler': 'icpx', 'other_package_components': 'original archived normal package, historical GNU host/icx SYCL; not rebuilt',
          'default_all_baseline_claimed': False, 'diagnostic_abi_bridge': False, 'gpu_run': False,
          'remaining_confound': 'e878/profiling ON and historical non-target package components versus official public7b56/profiling OFF'})
    print(json.dumps({'role': role, 'site': str(site), 'package_files': len(package_post), 'only_changed': changed, 'GPU': False}), flush=True)


if __name__ == '__main__':
    main()
