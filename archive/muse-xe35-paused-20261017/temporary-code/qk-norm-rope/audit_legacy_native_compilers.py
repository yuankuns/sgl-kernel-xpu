import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ARTIFACT = Path(__file__).resolve().parent
PERSONAL = Path('/home/syk/worktrees/sgl-kernel-xpu')
EVIDENCE = PERSONAL / 'review-validation-artifacts'
HISTORY = EVIDENCE / 'qk-before-after-20261017'
LIBRARY = 'libsgl-ops-sycl-FusedQKNormRope.so'
RECIPE = Path('src/CMakeFiles/sgl-ops-sycl-FusedQKNormRope.dir/sycl') / (
    'sgl-ops-sycl-FusedQKNormRope_gen_FusedQKNormRope.cpp.o.Release.cmake'
)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def command(arguments):
    process = subprocess.run(arguments, text=True, capture_output=True, check=False)
    assert process.returncode == 0, (arguments, process.stderr)
    return process.stdout.strip()


def record(path):
    return {'path': str(path), 'sha256': digest(path)}


def inspect_build(build, tested_hash):
    cache = build / 'CMakeCache.txt'
    recipe = build / RECIPE
    cache_values = {}
    keys = {
        'CMAKE_CXX_COMPILER', 'CMAKE_C_COMPILER', 'SYCL_COMPILER',
        'SYCL_HOST_COMPILER', 'XPU_SYCL_COMPILER', 'DPCPP_SYCL_TARGET',
        'FETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL',
    }
    for line in cache.read_text().splitlines():
        match = re.match(r'([^:]+):[^=]+=(.*)', line)
        if match and match[1] in keys:
            cache_values[match[1]] = match[2]
    compiler = next((build / 'CMakeFiles').glob('*/CMakeCXXCompiler.cmake'))
    compiler_lines = [
        line for line in compiler.read_text().splitlines()
        if re.match(r'set\(CMAKE_CXX_COMPILER(?:_ID|_VERSION)? ', line)
    ]
    recipe_lines = [
        line for line in recipe.read_text().splitlines()
        if re.match(r'set\((?:source_file|SYCL_(?:host_compiler|executable|compile_flags|compile_definitions)|CMAKE_HOST_FLAGS) ', line)
    ]
    result = {
        'library': record(build / 'src' / LIBRARY),
        'matches_tested_native': digest(build / 'src' / LIBRARY) == tested_hash,
        'cache': record(cache), 'cache_values': cache_values,
        'compiler_identity': record(compiler), 'compiler_identity_lines': compiler_lines,
        'generated_compile_recipe': record(recipe), 'recipe_lines': recipe_lines,
        'already_all_icpx': False,
        'recipe_is_not_verbose_executed_argv_transcript': True,
    }
    ninja = build / 'build.ninja'
    if ninja.exists():
        text = ninja.read_text()
        device = next(
            line for line in text.splitlines()
            if line.startswith('  COMMAND = ') and '-fsycl-link ' in line
            and 'FusedQKNormRope_sycl_device_obj.o' in line
        )
        host = re.search(
            r'^build src/libsgl-ops-sycl-FusedQKNormRope\.so:.*?\n\n',
            text, flags=re.MULTILINE | re.DOTALL,
        ).group()
        rules = build / 'CMakeFiles/rules.ninja'
        rule = re.search(
            r'^rule CXX_SHARED_LIBRARY_LINKER__sgl-ops-sycl-FusedQKNormRope_Release\n.*?\n\n',
            rules.read_text(), flags=re.MULTILINE | re.DOTALL,
        ).group()
        result.update({
            'ninja': record(ninja), 'rules': record(rules),
            'device_link_recipe': device,
            'device_link_recipe_sha256': hashlib.sha256(device.encode()).hexdigest(),
            'host_link_edge': host, 'host_link_rule': rule,
        })
    symbols = command(['nm', '-D', '--defined-only', '-C', str(build / 'src' / LIBRARY)])
    result['exported_original_native_symbols'] = [
        line for line in symbols.splitlines() if 'fused_qk_norm_rope(' in line
    ]
    assert result['exported_original_native_symbols']
    assert cache_values['CMAKE_CXX_COMPILER'] == '/usr/bin/c++'
    assert cache_values['SYCL_COMPILER'] == '/opt/gfx-deps/dpcpp/bin/icx'
    assert cache_values['SYCL_HOST_COMPILER'] == '/usr/bin/c++'
    assert any('"GNU"' in line for line in compiler_lines)
    return result


def main():
    provenance = json.loads((HISTORY / 'provenance.json').read_text())
    providers = {}
    for label, key, build in (
        ('preopt', 'before', PERSONAL / 'qk-norm-rope-main-baseline/build'),
        ('optimized_source', 'after', EVIDENCE / 'accuracy-20261016/muse_qk_norm_rope/clean-default-build'),
    ):
        provider = provenance[key]
        checkout = Path(provider['checkout'])
        checked = {}
        for relative, expected in provider['source_hashes'].items():
            contents = subprocess.check_output([
                'git', '-C', str(checkout), 'show', f"{provider['commit']}:{relative}",
            ])
            actual = hashlib.sha256(contents).hexdigest()
            assert actual == expected, relative
            checked[relative] = actual
        site_library = Path(provider['site']) / 'sgl_kernel' / LIBRARY
        assert digest(site_library) == provider['qk_library_sha256']
        native = inspect_build(build, provider['qk_library_sha256'])
        assert native['matches_tested_native']
        providers[label] = {
            'ref': provider['commit'], 'checkout': str(checkout),
            'source_inputs_git_verified': checked,
            'tested_site_library': record(site_library), 'matching_real_build': native,
        }
    source_current = inspect_build(
        PERSONAL / 'muse-spark-qk-norm-rope-main-20261014/build',
        provenance['after']['qk_library_sha256'],
    )
    assert not source_current['matches_tested_native']
    pin = ARTIFACT.parent / 'deps/sycl-tla-7b56'
    pin_head = command(['git', '-C', str(pin), 'rev-parse', 'HEAD'])
    pin_status = command(['git', '-C', str(pin), 'status', '--porcelain'])
    assert pin_head == '7b56b936b87f799f80dde60c2fd8d11c1128cebb' and not pin_status
    result = {
        'audit_time': datetime.now(timezone.utc).isoformat(),
        'scope': 'CPU-only existing-file/hash/git/nm audit; no configure/build/import/GPU',
        'provenance': record(HISTORY / 'provenance.json'), 'providers': providers,
        'current_source_build_not_tested_provider': source_current,
        'future_public_cutlass': {'path': str(pin), 'head': pin_head, 'clean': True},
        'historical_optimized_cutlass': 'internal muse-w4a4-e87870b path; public 7b56 NOT used in tested old SOURCE',
        'v3_default_full_status': 'main reports actual1 HCHead.cpp:70; drained; no official AFTER',
        'preopt_T32768_historical_strict_violations': 3508,
        'preopt_T32768_historical_actual_exit': 1,
        'strict_reference_unchanged': {'atol': 0.005, 'rtol': 0.005, 'tail_bitwise': True},
        'build_grant': False, 'gpu_grant': False, 'future_version_unassigned': True,
    }
    (ARTIFACT / 'legacy-native-compiler-audit.json').write_text(
        json.dumps(result, indent=2) + '\n'
    )
    for label, provider in providers.items():
        native = provider['matching_real_build']
        print(label, 'tested_library_sha256=' + native['library']['sha256'])
        print(label, 'generated_compile_recipe_sha256=' + native['generated_compile_recipe']['sha256'])
        print(label, 'CXX=GNU /usr/bin/c++; SYCL=icx; host=/usr/bin/c++; not all-icpx')
    print('CPU_AUDIT_ACTUAL_EXIT=0; public pin clean; optimized current build is NOT tested provider')


if __name__ == '__main__':
    main()
