import hashlib
import json
import re
from pathlib import Path
import subprocess


ROOT = Path('/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-fmha-xe35')
ART = Path('/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha')
SOURCE = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-fmha-main-20261014')
OLD = Path('/home/syk/worktrees/sgl-kernel-xpu/review-validation-artifacts/fmha-before-after-20261017')
DEP = ART.parent / 'deps/sycl-tla-7b56'


def git_xe35(root, *args):
    return subprocess.check_output(['git', '-c', f'safe.directory={root}', '-C', str(root), *args], text=True).strip()


def hash_xe35(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_algorithm_xe35(text):
    text = re.sub(r'^\s*#include.*$', '', text, flags=re.M)
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    text = text.replace('_xe35', '').replace('SGL_KERNEL_EXPORT', '')
    return re.sub(r'\s+', '', text)


def main_xe35():
    assert git_xe35(ROOT, 'rev-parse', 'HEAD') == '833c95208dccd2af5628c121222f91186a5b2de7'
    assert git_xe35(SOURCE, 'rev-parse', 'HEAD') == '08955d90fad52a40779705f259569db93e570a22'
    assert git_xe35(DEP, 'rev-parse', 'HEAD') == '7b56b936b87f799f80dde60c2fd8d11c1128cebb'
    assert not git_xe35(DEP, 'status', '--porcelain')
    include_roots = [ROOT / 'src', ROOT / 'include', DEP / 'include', DEP / 'tools/util/include']
    checked_includes = []
    for path in (ROOT / 'src/sycl/xe35/fmha').rglob('*'):
        if not path.is_file():
            continue
        assert '_xe35' not in path.name and 'Xe35' not in path.name, path
        for target in re.findall(r'^#include "([^\"]+)"', path.read_text(), re.M):
            candidates = [path.parent / target, *(base / target for base in include_roots)]
            assert any(candidate.is_file() for candidate in candidates), (path, target)
            checked_includes.append({'file': str(path.relative_to(ROOT)), 'target': target})
    source_sycl = SOURCE / 'src/sycl'
    private = ROOT / 'src/sycl/xe35/fmha'
    pairs = [(source_sycl / 'flash_attentionXe35.cpp', private / 'flash_attention.cpp'),
             (source_sycl / 'flash_attentionXe35_common.hpp', private / 'common.hpp')]
    source_kernels = source_sycl / 'kernels/flash_attention_v2/xe3'
    pairs += [(path, private / path.relative_to(source_kernels)) for path in source_kernels.rglob('*.hpp')]
    pairs += [(source_sycl / f'flash_attentionXe35_{kind}_kernel.cpp.in', private / f'fmha_{kind}_kernel.cpp.in')
              for kind in ('decode', 'prefill', 'decode_nopage', 'prefill_nopage')]
    equivalent_algorithms = []
    for before, after in pairs:
        text = before.read_text()
        if before.name == 'flash_attentionXe35_common.hpp':
            text = '#pragma once' + text.split('#pragma once', 2)[1]
        normalized = canonical_algorithm_xe35(text)
        assert normalized == canonical_algorithm_xe35(after.read_text()), after
        equivalent_algorithms.append({'source': str(before), 'port': str(after.relative_to(ROOT)),
                                      'canonical_sha256': hashlib.sha256(normalized.encode()).hexdigest()})
    protected = ['src/sycl/flash_attention.cpp']
    protected += git_xe35(ROOT, 'ls-files', 'src/*mla*', 'src/sycl/*mla*', 'src/sycl/kernels/flash_attention_v2').splitlines()
    for relative in protected:
        if '/xe35/' in relative:
            continue
        expected = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'HEAD:{relative}'])
        assert hashlib.sha256(expected).hexdigest() == hash_xe35(ROOT / relative), relative
    prior_source = json.loads((OLD / 'source-manifest.json').read_text())
    record = next(item for item in prior_source['sources'] if item['side'] == 'after')
    for filename, expected in record['files_sha256'].items():
        path = Path(filename)
        if not path.is_absolute():
            path = SOURCE / path
        path = Path(str(path).replace('/root/', '/home/syk/', 1))
        assert hash_xe35(path) == expected, path
    source_manifest = json.loads((OLD / 'binary-manifest-after.json').read_text())
    freshly_built = []
    required = {'libsgl-ops-sycl-flash_attentionXe35.so', 'libsgl-ops-sycl-flash_attentionXe35_decode_64_bf16.so', 'libsgl-ops-sycl-flash_attentionXe35_prefill_64_bf16.so'}
    for item in source_manifest:
        path = Path(item['path'].replace('/root/', '/home/syk/', 1))
        if path.name in required:
            actual = hash_xe35(path)
            assert actual == item['sha256'] == hash_xe35(OLD / 'build-after/src' / path.name)
            freshly_built.append({'path': str(path), 'sha256': actual, 'fresh_build_match': True})
    assert len(freshly_built) == 3
    changed = git_xe35(ROOT, 'diff', '--name-only').splitlines()
    added = git_xe35(ROOT, 'ls-files', '--others', '--exclude-standard', 'src', 'benchmark', 'tests').splitlines()
    files = sorted(set(changed + added))
    assert all(relative.startswith(('src/', 'benchmark/', 'tests/')) for relative in files), files
    assert all('build-xe35' not in relative and '/build/' not in relative for relative in files), files
    manifest = {'official_checkout': '/home/syk/sgl-kernel-xpu-up',
                'worktree': str(ROOT), 'branch': git_xe35(ROOT, 'branch', '--show-current'),
                'upstream': git_xe35(ROOT, 'rev-parse', '--abbrev-ref', '@{upstream}'),
                'official_base': git_xe35(ROOT, 'rev-parse', 'HEAD'), 'source': str(SOURCE),
                'source_head': git_xe35(SOURCE, 'rev-parse', 'HEAD'),
                'source_base': 'ab3a26c43a38e6aea7d73cc1349e4b5e210b772b',
                'public_cutlass_pin': git_xe35(DEP, 'rev-parse', 'HEAD'),
                'source_cutlass': 'historical freshly-built e878; not used as port dependency',
                'source_reused_fresh_binaries': freshly_built,
                'generic_and_mla_protected_files': len(protected),
                'private_include_audit': checked_includes,
                'algorithm_equivalence': equivalent_algorithms,
                'algorithm_equivalence_scope': '18 active files; ignores includes/comments/whitespace/architecture identifier suffixes/export annotations; not a substitute for runtime accuracy or dependency equivalence',
                'source_write_set_scope': 'Only src/, benchmark/, tests/; explicitly excludes build-xe35/, generated build sources, lock files, site packages and artifacts; project .gitignore unchanged',
                'files': {relative: hash_xe35(ROOT / relative) for relative in files}}
    (ART / 'source-port-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (ART / 'changed-files.txt').write_text('\n'.join(files) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main_xe35()
