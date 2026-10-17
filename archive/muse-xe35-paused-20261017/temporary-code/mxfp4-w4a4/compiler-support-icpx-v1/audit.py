import hashlib
import json
import pathlib
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sidecar = pathlib.Path(__file__).resolve().parent
    source = sidecar / 'source'
    original = pathlib.Path('/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-mxfp4-w4a4-xe35')
    relative = 'cmake/BuildFlags.cmake'
    before = (sidecar / 'BuildFlags-original.cmake').read_text()
    expected = before.replace('if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")', 'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")', 1).replace('Currently only support GCC compiler on Linux as CXX compiler.', 'Currently only support GNU or IntelLLVM compiler on Linux as CXX compiler.', 1)
    assert (source / relative).read_text() == expected
    finder = 'cmake/Modules/FindSYCLToolkit.cmake'
    assert (source / finder).read_bytes() == (original / finder).read_bytes()
    freeze = json.loads((sidecar.parent / 'canonical-source-freeze-v2.json').read_text())
    assert digest(sidecar / 'BuildFlags-original.cmake') == freeze['files'][relative]['sha256']
    changed = [name for name, record in freeze['files'].items() if digest(source / name) != record['sha256']]
    assert changed == [relative], changed
    build = sidecar / 'build-intelllvm-v2'
    compiler = '/opt/gfx-deps/dpcpp/bin/icpx'
    ids = {}
    for language in ('C', 'CXX'):
        record = next((build / 'CMakeFiles').glob('*/CMake' + language + 'Compiler.cmake')).read_text()
        ids[language] = re.search(r'set\(CMAKE_' + language + r'_COMPILER_ID "([^"]+)"\)', record).group(1)
        assert ids[language] == 'IntelLLVM'
    precision = ['-fhonor-nans', '-fhonor-infinities', '-fno-associative-math', '-fno-approx-func', '-no-ftz', '-fno-sycl-instrument-device-code', '-fsycl-fp64-conv-emu']
    compile_flags = ['-fsycl', '-ftemplate-backtrace-limit=0', '-fno-sycl-unnamed-lambda', '-std=c++20', '-sycl-std=2020', '-fsycl-targets=spir64_gen'] + precision
    scripts = sorted(build.rglob('*.Release.cmake'))
    compile_evidence = []
    for path in scripts:
        text = path.read_text()
        flags = re.search(r'^set\(SYCL_compile_flags (.*)\) # list$', text, re.MULTILINE).group(1).split(';')
        for flag in compile_flags:
            assert flag in flags, (path, flag)
        assert f'set(SYCL_executable "{compiler}")' in text, path
        assert f'set(SYCL_host_compiler "{compiler}")' in text, path
        compile_evidence.append({'script': str(path.relative_to(sidecar)), 'sha256': digest(path), 'flags': flags, 'sycl_compiler': compiler, 'host_compiler': compiler})
    commands = (sidecar / 'ninja-commands-v2.log').read_text().splitlines()
    links = [line for line in commands if '-fsycl-link' in line]
    link_flags = ['-fsycl', '-fsycl-max-parallel-link-jobs=1', '--offload-compress', '-fsycl-targets=spir64_gen', '-Xspirv-translator', '+SPV_INTEL_split_barrier', '+SPV_INTEL_2d_block_io', '+SPV_INTEL_subgroup_matrix_multiply_accumulate', '-device\\ cri', '-cl-poison-unsupported-fp64-kernels', '-cl-intel-enable-auto-large-GRF-mode', '-cl-fp32-correctly-rounded-divide-sqrt', '-cl-intel-greater-than-4GB-buffer-required']
    for command in links:
        assert compiler in command
        for flag in link_flags:
            assert flag in command, flag
    grf = [line for line in links if '-cl-intel-512-GRF-per-thread' in line]
    assert len(grf) == 1 and 'moe_grouped_mm_mxfp4_w4a4' in grf[0]
    hosts = [line for line in commands if ' -c ' in line]
    assert hosts and all(compiler in command for command in hosts)
    assert len(links) == len(scripts) == 712
    for name in ('configure-v2.exit', 'wrapper-v2.exit', 'ninja-inspect-v2.exit'):
        assert (sidecar / name).read_text().strip() == '0'
    (sidecar / 'generated-sycl-compile-flags.json').write_text(json.dumps(compile_evidence, indent=2) + '\n')
    (sidecar / 'ninja-host-commands.txt').write_text('\n'.join(hosts) + '\n')
    (sidecar / 'ninja-device-link-commands.txt').write_text('\n'.join(links) + '\n')
    (sidecar / 'ninja-w4-grf512-command.txt').write_text(grf[0] + '\n')
    torch_finder = (sidecar / 'torch-FindSYCLToolkit-original.cmake').read_text()
    assert 'elseif("${XPU_SYCL_COMPILER}" MATCHES "icx")' in torch_finder
    assert 'MATCHES "icpx"' not in torch_finder
    cache = (build / 'CMakeCache.txt').read_text()
    assert 'XPU_SYCL_COMPILER:STRING=icx' in cache
    report = {'kind': 'compiler-support proposal CPU configure/generated-command audit; not full build/package/GPU acceptance', 'configure_handle': 50824, 'configure_actual_exit': 0, 'v1_failed_handle': 55946, 'v1_actual_exit': 1, 'modified_source_files': changed, 'finder_proposal_removed_as_unnecessary': True, 'buildflags_production_body_byte_preserved': True, 'compiler_ids': ids, 'actual_cxx_sycl_host_compiler': compiler, 'actual_c_compiler': '/opt/gfx-deps/dpcpp/bin/icx', 'every_sycl_compile_script_checked': len(scripts), 'every_device_link_command_checked': len(links), 'host_compile_command_count': len(hosts), 'precision_flags': precision, 'grf512_w4_only': True, 'default_features_not_disabled': True, 'public_cutlass_pin': '7b56b936b87f799f80dde60c2fd8d11c1128cebb', 'successful_torch_family_token': 'icx', 'torch_family_icpx_supported_by_current_finder': False, 'main_driver_required_action': 'omit XPU_SYCL_COMPILER or retain actual oneAPI-family icx token; actual SYCL/CXX/host paths stay absolute icpx. An explicit family token icpx is not covered by this successful configure and current Torch finder does not recognize it.', 'live_handle': None, 'full_or_gpu_launched': False, 'frozen_pr_writes': 0}
    (sidecar / 'cpu-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
