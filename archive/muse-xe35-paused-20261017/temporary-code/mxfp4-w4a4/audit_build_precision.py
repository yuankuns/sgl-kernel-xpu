import hashlib
import json
import pathlib
import re
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = pathlib.Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014')
    artifact = pathlib.Path(__file__).resolve().parent
    target = artifact.parent.parent / 'pr/muse-mxfp4-w4a4-xe35'
    dependencies = {
        'source': pathlib.Path('/home/syk/worktrees/cutlass_internal/muse-w4a4-e87870b'),
        'port': artifact.parent / 'deps/sycl-tla-7b56',
    }
    headers = ('include/cutlass/bfloat16.h', 'include/cutlass/numeric_conversion.h', 'include/cute/arch/mma_xe.hpp', 'include/cute/atom/mma_traits_xe.hpp', 'include/cutlass/gemm/dispatch_policy.hpp')
    report = {'acceptance': 'W4 source-to-port .005/.005 FAIL; root cause pending', 'dependency': {}, 'generated_compile_settings': {}, 'input_hash_limit': 'Previous GPU round did not persist raw input hashes. Same seed and AA/BB equality are not substituted for raw input hash proof.', 'historical_compiler_limit': 'Current compiler version/path in retained generated settings is evidence, not an immutable hash of the historical compiler binary.'}
    for implementation, dependency in dependencies.items():
        report['dependency'][implementation] = {
            'path': str(dependency),
            'head': subprocess.check_output(['git', '-C', str(dependency), 'rev-parse', 'HEAD'], text=True).strip(),
            'dirty': subprocess.check_output(['git', '-C', str(dependency), 'status', '--porcelain'], text=True).strip(),
            'headers_sha256': {header: digest(dependency / header) for header in headers},
        }
        build = pathlib.Path('/home/syk/worktrees/sgl-kernel-xpu/review-validation-artifacts/accuracy-20261016/muse_mxfp4_w4a4/full-default-clean-build') if implementation == 'source' else target / 'build'
        pattern = 'src/CMakeFiles/sgl-ops-sycl-moe_grouped_mm_mxfp4_w4a4.dir/sycl/xe35/*Release.cmake'
        settings = next(build.glob(pattern))
        text = settings.read_text()
        keys = ('SYCL_executable', 'SYCL_compile_flags', 'SYCL_compile_definitions', 'SYCL_host_compiler', 'CMAKE_HOST_FLAGS')
        report['generated_compile_settings'][implementation] = {
            'path': str(settings), 'sha256': digest(settings),
            'settings': {key: re.search(r'^set\(' + key + r' (.*)\)', text, re.MULTILINE).group(1) for key in keys},
        }
    report['bf16_conversion_headers_identical'] = all(report['dependency']['source']['headers_sha256'][header] == report['dependency']['port']['headers_sha256'][header] for header in headers[:2])
    report['compiled_dependency_header_limit'] = 'Current clean dependency files are hashed. Historical compiled dependency headers were not included in original source manifest; current file equality is not historical compiled-header equality proof.'
    report['source_build_command'] = pathlib.Path('/home/syk/worktrees/sgl-kernel-xpu/review-validation-artifacts/accuracy-20261016/muse_mxfp4_w4a4/full-default-build.command').read_text()
    report['investigation'] = [
        'Both generated device flags retain fhonor-nans/fhonor-infinities/fno-associative-math/fno-approx-func/no-ftz; no observed fast-math precision change.',
        'Source uses private e87870bb7aacf83bfb3be9da685f148316b8df23; port remains official 7b56b936b87f799f80dde60c2fd8d11c1128cebb.',
        'BF16 and numeric_conversion headers are byte-identical; source profiling macros and port default feature/host visibility macros differ.',
        'Compare exact tensor/scale hashes plus selected FP64 tie-point references in a newly authorized GPU8 run after routing naturally terminates.',
        'Audit native blockscaled MMA/scale traits and actual device-link flags before any Xe35-local semantic repair. No global/BMG/public dependency modifications.',
    ]
    output = artifact / 'diagnostic-build-precision-audit.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
