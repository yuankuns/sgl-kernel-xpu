import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import verify_source_baseline

root = Path(__file__).resolve().parent
source = verify_source_baseline.SOURCE
build = source / 'build'
revision = verify_source_baseline.REVISION
targets = ['sgl-ops-sycl-nvfp4_woq', 'sgl-ops-sycl-blockwise_moe_nvfp4_woq', 'sgl-ops-sycl-mxfp4_moe_scale_layout', 'sgl-ops-sycl-blockwise_moe_mxfp8']
subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', revision, '--'], check=True)
baseline = verify_source_baseline.verify()
cache_path = build / 'CMakeCache.txt'
cache = dict(re.findall(r'^([^#/:\n][^:\n]*):[^=\n]+=(.*)$', cache_path.read_text(), re.MULTILINE))
ninja_path = build / 'build.ninja'
rules_path = build / 'CMakeFiles/rules.ninja'
ninja = ninja_path.read_text()
rules = rules_path.read_text()
ninja_log_path = build / '.ninja_log'
ninja_log = ninja_log_path.read_text()
compiler_files = list((build / 'CMakeFiles').glob('*/CMakeCXXCompiler.cmake'))
assert len(compiler_files) == 1
compiler_id = re.search(r'set\(CMAKE_CXX_COMPILER_ID "([^"]+)"\)', compiler_files[0].read_text()).group(1)
input_files = [cache_path, ninja_path, rules_path, ninja_log_path, compiler_files[0], source / 'src/BuildOnLinux.cmake', source / 'cmake/BuildFlags.cmake']
required_flags = ['-fhonor-nans', '-fhonor-infinities', '-fno-associative-math', '-fno-approx-func', '-no-ftz', '-fsycl-targets=spir64_gen']
records = {}
for target in targets:
    generated = list((build / 'src/CMakeFiles' / (target + '.dir')).rglob('*.Release.cmake'))
    assert len(generated) == 1, target
    input_files.extend(generated)
    text = generated[0].read_text()
    variables = {}
    for name in ['source_file', 'SYCL_host_compiler', 'SYCL_executable', 'SYCL_compile_flags', 'SYCL_include_dirs', 'SYCL_compile_definitions', 'CMAKE_HOST_FLAGS']:
        match = re.search(r'^set\(' + name + r' (.*)\)(?: #.*)?$', text, re.MULTILINE)
        assert match, (target, name)
        variables[name] = match.group(1)
    assert all(flag in variables['SYCL_compile_flags'] for flag in required_flags)
    generate_block = text[text.index('SYCL_execute_process(\n  "Generating'):]
    generate_block = generate_block[:generate_block.index('\n  )') + len('\n  )')]
    custom_commands = [line.strip() for line in ninja.splitlines() if line.strip().startswith('COMMAND =') and target + '.dir/' in line]
    aot = [line for line in custom_commands if '-fsycl-link ' in line]
    assert len(aot) == 1, target
    if target == 'sgl-ops-sycl-blockwise_moe_nvfp4_woq':
        assert '-cl-intel-512-GRF-per-thread' in aot[0]
    library = build / 'src' / ('lib' + target + '.so')
    rule_name = 'CXX_SHARED_LIBRARY_LINKER__' + target + '_Release'
    rule_match = re.search(r'^rule ' + re.escape(rule_name) + r'\n.*?(?=\n\n)', rules, re.MULTILINE | re.DOTALL)
    library_match = re.search(r'^build src/' + re.escape(library.name) + r':.*?(?=\n\n)', ninja, re.MULTILINE | re.DOTALL)
    assert rule_match and library_match
    comment = subprocess.run(['readelf', '-p', '.comment', str(library)], text=True, capture_output=True)
    records[target] = {'generated_sycl_file': str(generated[0]), 'configured_compile_variables_verbatim': variables, 'actual_generated_compile_invocation_template': generate_block, 'actual_ninja_custom_commands_verbatim': custom_commands, 'actual_host_link_rule_verbatim': rule_match.group(0), 'actual_library_link_edge_flags_verbatim': library_match.group(0), 'ninja_completed_output_records': [line for line in ninja_log.splitlines() if target in line], 'library': str(library), 'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(), 'elf_comment_actual_exit': comment.returncode, 'elf_comment': comment.stdout.strip(), 'elf_comment_stderr': comment.stderr.strip()}

dependency = Path(cache['FETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL'].replace('/root/', '/home/syk/', 1))
dependency_head = subprocess.check_output(['git', '-C', str(dependency), 'rev-parse', 'HEAD'], text=True).strip()
dependency_status = subprocess.check_output(['git', '-C', str(dependency), 'status', '--porcelain'], text=True).strip()
assert dependency_head == 'e87870bb7aacf83bfb3be9da685f148316b8df23' and not dependency_status
history_path = source / 'review_logs/nvfp4_woq_20261015/commands.md'
history = history_path.read_text()
start = history.index('cmake -S /workspace')
stop = history.index('CUTLASS=/cutlass', start)
input_files.extend([history_path, source / 'review_logs/nvfp4_woq_20261015/build-baseline.log'])
compiler_keys = ['CMAKE_CXX_COMPILER', 'CMAKE_C_COMPILER', 'SYCL_COMPILER', 'SYCL_HOST_COMPILER', 'XPU_SYCL_COMPILER', 'CMAKE_BUILD_TYPE', 'DPCPP_SYCL_TARGET', 'FETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL']
failure_dir = root.parent / 'bf16-gemm/canonical-v3-icpx'
failure_log = failure_dir / 'default-all-build.log'
failure = {'default_all_actual_exit': int((failure_dir / 'default-all-build.exit').read_text()), 'supervisor_actual_exit': int((failure_dir / 'supervisor.exit').read_text()), 'raw_log': str(failure_log), 'raw_log_sha256': hashlib.sha256(failure_log.read_bytes()).hexdigest(), 'error_lines': [line for line in failure_log.read_text().splitlines() if "error: 'sycl::reqd_work_group_size' attribute requires exactly 3 arguments" in line], 'all_compilers_drained': 'coordinator reported; this CPU audit does not claim a new full build', 'next_formal_version': None}
assert failure['default_all_actual_exit'] == failure['supervisor_actual_exit'] == 1
print(json.dumps({'mode': 'CPU-read-only-source-compiler-audit', 'source_revision': revision, 'tracked_source_diff_zero': True, 'compiler_id': compiler_id, 'compiler_cache': {name: cache[name] for name in compiler_keys}, 'source_is_GNU_host_plus_icx_SYCL': compiler_id == 'GNU' and cache['SYCL_COMPILER'].endswith('/icx') and cache['SYCL_HOST_COMPILER'] == '/usr/bin/c++', 'original_native_ABI_verified': baseline['original_native_abi_verified'], 'source_cutlass': {'path': str(dependency), 'HEAD': dependency_head, 'clean': not dependency_status}, 'original_NV_GRF512_present_in_actual_AOT_command': True, 'targets': records, 'historical_documented_build_command': history[start:stop].strip(), 'command_provenance_limits': 'Current build metadata and completed Ninja records plus identical native SHA/ELF comments corroborate mixed compiler provenance. Generated compile invocation is recorded as an unexpanded CMake template, not fabricated verbose argv. Historical commands.md describes an earlier 2b9ac2b review; no dedicated .command/exit proves the final d983 library shell invocation. No independent source full-package build acceptance claimed.', 'input_files_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_files}, 'canonical_v3_failure': failure, 'source_or_PR_modified': False, 'compiler_launched': False, 'configure_launched': False, 'full_or_GPU_launched': False, 'fresh_source_icpx_build_granted': False}, indent=2))
