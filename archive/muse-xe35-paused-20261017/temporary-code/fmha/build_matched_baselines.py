import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ART = Path(__file__).resolve().parent
ROOT = ART / 'matched-native-icpx-v1'
TARGETS = ('sgl-ops-sycl-flash_attentionXe35',
           'sgl-ops-sycl-flash_attentionXe35_decode_64_bf16',
           'sgl-ops-sycl-flash_attentionXe35_prefill_64_bf16')
ICPX = '/opt/gfx-deps/dpcpp/bin/icpx'
ICX = '/opt/gfx-deps/dpcpp/bin/icx'
RESOURCE_RETRY = False


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def mounted(path):
    return Path(path.replace('/home/syk/', '/root/', 1))


def audit_inputs(role, phase):
    folder = ROOT / role
    record = json.loads((folder / 'source-pre.json').read_text())
    original = mounted(record['original'])
    git = ['git', '-c', f'safe.directory={original}', '-C', str(original)]
    assert subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip() == record['head']
    assert not subprocess.check_output([*git, 'status', '--porcelain', '--untracked-files=no'], text=True)
    original_hashes = {name: digest(original / name) for name in record['tracked_files']}
    assert original_hashes == record['tracked_files']
    snapshot_hashes = {name: digest(folder / 'source' / name) for name in record['tracked_files']}
    changes = [name for name in snapshot_hashes if snapshot_hashes[name] != original_hashes[name]]
    assert changes == ['cmake/BuildFlags.cmake'], changes
    expected = (original / changes[0]).read_text().replace(
        'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")',
        'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")').replace(
        'only support GCC compiler on Linux as CXX compiler.',
        'only support GCC or IntelLLVM compiler on Linux as CXX compiler.')
    assert (folder / 'source' / changes[0]).read_text() == expected
    dependency_expected = json.loads((ROOT / 'e878-full-files-pre.json').read_text())
    dependency_hashes = {name: digest(ROOT / 'deps/e878' / name) for name in dependency_expected}
    assert dependency_hashes == dependency_expected
    assert {str(path.relative_to(ROOT / 'deps/e878')) for path in (ROOT / 'deps/e878').rglob('*') if path.is_file()} == set(dependency_expected)
    save(folder / f'inputs-{phase}.json', {'source_head': record['head'], 'original_tracked_files': original_hashes,
         'snapshot_tracked_files': snapshot_hashes, 'allowed_delta': changes,
         'dependency_files_sha256': dependency_hashes, 'dependency_label': record['dependency_historical_label']})


def run_phase(role, phase, command):
    prefix = ROOT / role / phase
    if RESOURCE_RETRY and phase == 'configure':
        assert prefix.with_suffix('.exit').read_text().strip() == '0'
        print(f'Resource-only retry: reuse actual exit0 {role} configure, same source/flags graph', flush=True)
        return
    if RESOURCE_RETRY:
        prefix = ROOT / role / f'{phase}-resource-v4'
    if role == 'source' and phase == 'configure' and prefix.with_suffix('.started.json').exists():
        assert prefix.with_suffix('.exit').read_text().strip() == '0'
        recovery = json.loads((ROOT / role / 'configure.drain-recovery-v3.json').read_text())
        assert recovery['remaining_live'] == [] and recovery['configure_exit'] == 0
        print('Reuse actual exit0 SOURCE configure; recorded v2 supervisor serialization failure is not hidden', flush=True)
        return
    assert not prefix.with_suffix('.started.json').exists(), prefix
    options = ['--compiler-limit-gib', '16', '--tree-limit-gib', '24', '--min-available-gib', '96', '--outer-jobs', '1'] if RESOURCE_RETRY else []
    result = subprocess.run([sys.executable, str(ART / 'bounded_baseline_command.py'),
        '--prefix', str(prefix), '--cwd', str(ROOT / role), *options, '--', *command])
    if result.returncode:
        raise SystemExit(result.returncode)


def audit_graph(role):
    build = ROOT / role / 'build'
    identity = next((build / 'CMakeFiles').glob('*/CMakeCXXCompiler.cmake')).read_text()
    assert 'set(CMAKE_CXX_COMPILER_ID "IntelLLVM")' in identity
    assert f'set(CMAKE_CXX_COMPILER "{ICPX}")' in identity
    c_identity = next((build / 'CMakeFiles').glob('*/CMakeCCompiler.cmake')).read_text()
    assert f'set(CMAKE_C_COMPILER "{ICX}")' in c_identity
    ninja = (build / 'build.ninja').read_text()
    rules = (build / 'CMakeFiles/rules.ninja').read_text()
    native = {}
    strict = ('-fhonor-nans', '-fhonor-infinities', '-fno-associative-math', '-fno-approx-func', '-no-ftz')
    for target in TARGETS:
        generated = list((build / 'src/CMakeFiles' / f'{target}.dir/sycl').glob('*.Release.cmake'))
        assert len(generated) == 1, target
        text = generated[0].read_text()
        assert f'set(SYCL_executable "{ICPX}")' in text
        assert f'set(SYCL_host_compiler "{ICPX}")' in text
        assert all(flag in text for flag in strict)
        assert all(flag in text for flag in ('CUTLASS_SYCL_PROFILING_ENABLED', 'COMPAT_PROFILING_ENABLED', 'SYCL_INTEL_TARGET=35'))
        edges = [block for block in ninja.split('\n\n') if block.startswith('build ') and
                 f'{target}_sycl_device_obj.o:' in block and ' -fsycl-link ' in block]
        assert len(edges) == 1 and ICPX + ' -fPIC -fsycl' in edges[0]
        assert '-fsycl-max-parallel-link-jobs=1 ' in edges[0]
        assert all(flag in edges[0].replace('\\ ', ' ') for flag in ('-device cri', '-cl-intel-enable-auto-large-GRF-mode', '-cl-fp32-correctly-rounded-divide-sqrt'))
        link = next(block for block in ninja.split('\n\n') if block.startswith(f'build src/lib{target}.so: '))
        rule_name = link.splitlines()[0].split(': ', 1)[1].split()[0]
        rule = next(block for block in rules.split('\n\n') if block.startswith(f'rule {rule_name}\n'))
        assert ICPX in rule and '/usr/bin/c++' not in rule
        native[target] = {'generated_command': str(generated[0]), 'generated_sha256': digest(generated[0]),
                          'native_recipe': text, 'device_link_edge': edges[0], 'shared_link_edge': link, 'shared_link_rule': rule}
    record = json.loads((ROOT / role / 'source-pre.json').read_text())
    for name, details in record['original_generated_instances'].items():
        assert digest(build / 'src/sycl' / name) == details['sha256'], name
    save(ROOT / role / 'compiler-graph-verified.json', {'role': role, 'cxx_identity': identity,
         'c_identity': c_identity, 'targets': native, 'build_ninja_sha256': digest(build / 'build.ninja'),
         'rules_ninja_sha256': digest(build / 'CMakeFiles/rules.ninja'), 'generated_instances_match_original': True,
         'scope': 'three original BF16 D64 native targets only; not a default-all build'})


def main():
    global RESOURCE_RETRY
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=('source', 'preopt'))
    parser.add_argument('--resource-retry-v4', action='store_true')
    args = parser.parse_args()
    roles = (args.role,) if args.role else ('source', 'preopt')
    RESOURCE_RETRY = args.resource_retry_v4
    if RESOURCE_RETRY:
        assert args.role is None
        assert json.loads((ROOT / 'source/native-build.drain-verified.json').read_text())['remaining_live'] == []
        assert json.loads((ROOT / 'preopt/manual-budget-transition-drain.json').read_text())['remaining_live'] == []
        assert int((ROOT / 'source/native-build.exit').read_text()) == 137
        assert int((ROOT / 'preopt/native-build.exit').read_text()) == 137
        assert os.environ['CMAKE_BUILD_PARALLEL_LEVEL'] == '1'
    assert sorted(os.sched_getaffinity(0)) == list(range(16, 32))
    assert os.environ['MAX_JOBS'] == '1'
    assert os.environ['LD_LIBRARY_PATH'].split(':')[0] == '/opt/gfx-deps/profiling_tools/lib'
    lock = (ROOT / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    label = 'resource-v4' if RESOURCE_RETRY else ('preopt-v1' if args.role == 'preopt' else 'v3')
    save(ROOT / f'pipeline-{label}.started.json', {'pid': os.getpid(), 'epoch': time.time(),
         'cpu_affinity': sorted(os.sched_getaffinity(0)), 'outer_jobs': 1 if RESOURCE_RETRY else 2, 'MAX_JOBS': '1',
         'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH'], 'gpu_started': False})
    import torch
    assert not torch.xpu.is_initialized()
    save(ROOT / f'torch-cpu-preflight-{label}.json', {'torch': torch.__version__, 'torch_file': torch.__file__,
         'gpu_initialized': torch.xpu.is_initialized(), 'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH']})
    print('Torch CPU import passed; XPU initialized=False', flush=True)
    for role in roles:
        audit_inputs(role, 'prebuild')
        source, build = ROOT / role / 'source', ROOT / role / 'build'
        command = ['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja',
            '-DCMAKE_BUILD_TYPE=Release', '-DDPCPP_SYCL_TARGET=cri',
            f'-DCMAKE_CXX_COMPILER={ICPX}', f'-DCMAKE_C_COMPILER={ICX}',
            f'-DSYCL_COMPILER={ICPX}', f'-DSYCL_HOST_COMPILER={ICPX}',
            '-DSGL_ENABLE_SYCL_PROFILING=ON', '-DCUTLASS_SYCL_PROFILING_ENABLED=ON',
            f'-DPYTHON3_EXECUTABLE={sys.executable}', '-DSKBUILD_SABI_COMPONENT=Development.SABIModule',
            '-DSKBUILD_SABI_VERSION=3.9',
            f'-DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL={ROOT / "deps/e878"}']
        try:
            run_phase(role, 'configure', command)
            audit_graph(role)
            run_phase(role, 'native-build', ['cmake', '--build', str(build), '--parallel', '1' if RESOURCE_RETRY else '2', '--verbose', '--target', *TARGETS])
            hashes = {f'lib{target}.so': digest(build / 'src' / f'lib{target}.so') for target in TARGETS}
            save(ROOT / role / 'fresh-native-binaries.json', {'libraries': hashes, 'completed_ninja_log_sha256': digest(build / '.ninja_log'),
                 'terminal_file': 'native-build-resource-v4.terminal.json' if RESOURCE_RETRY else 'native-build.terminal.json',
                 'build_boundary': 'resource-only same-graph incremental three-target build, not clean/default-all' if RESOURCE_RETRY else 'three-target build'})
        finally:
            audit_inputs(role, 'postbuild')
    save(ROOT / f'pipeline-{label}-complete.json', {'exit': 0, 'roles_serial': list(roles), 'gpu_started': False})
    if all((ROOT / role / 'fresh-native-binaries.json').is_file() for role in ('source', 'preopt')):
        save(ROOT / 'pipeline-complete.json', {'exit': 0, 'roles_serial': ['source', 'preopt'], 'gpu_started': False})


if __name__ == '__main__':
    main()
