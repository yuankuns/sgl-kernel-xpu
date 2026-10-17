import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent
HISTORICAL = Path('/home/syk/worktrees/sgl-kernel-xpu/review-validation-artifacts/accuracy-20261016/residual-final-full-release')
SOURCE = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-residual-norm-main-20261014')
CUTLASS = Path('/home/syk/worktrees/cutlass_internal/muse-w4a4-e87870b')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


cache_path = HISTORICAL / 'build/CMakeCache.txt'
cache = {name: value for name, value in re.findall(r'^([^/#\n:]+):[^=\n]*=(.*)$', cache_path.read_text(), re.MULTILINE)}
generated = HISTORICAL / 'build/src/CMakeFiles/sgl-ops-sycl-FusedNormResidual.dir/sycl/sgl-ops-sycl-FusedNormResidual_gen_FusedNormResidual.cpp.o.Release.cmake'
settings = [line for line in generated.read_text().splitlines() if line.startswith(('set(SYCL_host_compiler ', 'set(SYCL_executable ', 'set(SYCL_compile_flags ', 'set(SYCL_compile_definitions ', 'set(CMAKE_HOST_FLAGS '))]
commands = [line for line in (HISTORICAL / 'full-default-commands.txt').read_text().splitlines()
            if 'FusedNormResidual' in line and 'common_ops' not in line]
assert len(commands) == 3, len(commands)
kernel = HISTORICAL / 'frozen-source/src/sycl/FusedNormResidual.cpp'
expected_kernel = subprocess.check_output(['git', '--no-optional-locks', '-C', str(SOURCE), 'show', 'ec29f7a6a14c4f1fac42fb93526236ca03ab894b:src/sycl/FusedNormResidual.cpp'])
assert kernel.read_bytes() == expected_kernel
helpers = {}
for name in ['src/sycl/MemoryAccess.h', 'src/sycl/SYCLHelpers.h', 'src/sycl/Utils.h']:
    content = subprocess.check_output(['git', '--no-optional-locks', '-C', str(SOURCE), 'show', f'ec29f7a:{name}'])
    assert content == (HISTORICAL / 'frozen-source' / name).read_bytes()
    helpers[name] = hashlib.sha256(content).hexdigest()
dependency_file = generated.with_name('sgl-ops-sycl-FusedNormResidual_gen_FusedNormResidual.cpp.o.depend')
dependencies = re.findall(r'"([^"]+)"', dependency_file.read_text())
cutlass_headers = [name for name in dependencies if 'cutlass' in name.lower()]
pin = subprocess.check_output(['git', '--no-optional-locks', '-C', str(CUTLASS), 'rev-parse', 'HEAD'], text=True).strip()
assert pin == 'e87870bb7aacf83bfb3be9da685f148316b8df23'
assert not subprocess.check_output(['git', '--no-optional-locks', '-C', str(CUTLASS), 'status', '--porcelain'], text=True).strip()
native_relative = 'sgl_kernel/libsgl-ops-sycl-FusedNormResidual.so'
native = HISTORICAL / 'stage' / native_relative
isolated = ROOT / 'source-full-venv/lib/python3.12/site-packages' / native_relative
assert sha(native) == sha(isolated)
compiler_identity = list((HISTORICAL / 'build/CMakeFiles').glob('*/CMakeCXXCompiler.cmake'))
assert len(compiler_identity) == 1
assert 'set(CMAKE_CXX_COMPILER_ID "GNU")' in compiler_identity[0].read_text()
record = {'audit': 'CPU-only read-only source provenance; no configure/build/import/GPU',
          'source_ref': 'ec29f7a6a14c4f1fac42fb93526236ca03ab894b', 'source_diff_base': 'ab3a26c',
          'historical_package_root': str(HISTORICAL), 'compiler_id': 'GNU',
          'compiler_cache': {name: cache.get(name) for name in ['CMAKE_CXX_COMPILER', 'CMAKE_C_COMPILER',
              'SYCL_COMPILER', 'SYCL_HOST_COMPILER', 'XPU_SYCL_COMPILER', 'DPCPP_SYCL_TARGET',
              'FETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL']},
          'actual_generated_SYCL_settings': settings, 'actual_recorded_native_commands': commands,
          'historical_cutlass_pin': 'e87870bb7aacf83bfb3be9da685f148316b8df23',
          'historical_cutlass_pin_provenance': str(HISTORICAL / 'build.sh'),
          'original_helpers_sha256': helpers,
          'original_cutlass_current_checkout': {'path': str(CUTLASS), 'head': pin, 'clean': True},
          'historical_residual_TU_dependencies': {'path': str(dependency_file), 'sha256': sha(dependency_file),
              'count': len(dependencies), 'cutlass_headers': cutlass_headers,
              'limit': 'historical GNU/icx dependency record only; fresh icpx dependency closure must be audited again; retain original e878 pin'},
          'source_kernel_sha256': sha(kernel), 'source_native_sha256': sha(native),
          'isolated_native_matches_historical': True,
          'common_ops_sha256': sha(HISTORICAL / 'stage/sgl_kernel/common_ops.cpython-312-x86_64-linux-gnu.so'),
          'evidence_sha256': {str(path): sha(path) for path in [cache_path, generated, compiler_identity[0],
              HISTORICAL / 'build.sh', HISTORICAL / 'full-default-commands.txt', HISTORICAL / 'full-build.exit']},
          'historical_full_exit': int((HISTORICAL / 'full-build.exit').read_text()),
          'comparison_limit': 'historical native is GNUhost/icx; fresh SOURCE must preserve original helpers/e878, AFTER is later full icpx normal package/public7b56; dependency pin difference is disclosed, not silently normalized'}
(ROOT / 'source-compiler-provenance.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({name: value for name, value in record.items() if name not in ['actual_recorded_native_commands', 'evidence_sha256']}, indent=2))
