import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True
DIRECTORY = Path(__file__).resolve().parent
ARTIFACT = DIRECTORY.parent
sys.path.insert(0, str(ARTIFACT.parent / 'memory/fresh-source-icpx-v1'))
import sidecar
sys.path.insert(0, str(ARTIFACT))
import verify_source_baseline

REVISION = 'd983e877d6bf97015027bd3b1379ff047009a5dd'
CUTLASS_REVISION = 'e87870bb7aacf83bfb3be9da685f148316b8df23'
NAMES = ('nvfp4_woq', 'blockwise_moe_nvfp4_woq', 'mxfp4_moe_scale_layout', 'blockwise_moe_mxfp8')
SOURCE = verify_source_baseline.SOURCE
CUTLASS = sidecar.local_path('/home/syk/worktrees/cutlass_internal/muse-w4a4-e87870b')


def verify_original():
    verify_source_baseline.verify()
    if sidecar.git(SOURCE, 'diff', REVISION, '--').strip():
        raise RuntimeError('Authoritative source worktree changed')
    if sidecar.git(CUTLASS, 'rev-parse', 'HEAD').decode().strip() != CUTLASS_REVISION or sidecar.git(CUTLASS, 'status', '--porcelain').strip():
        raise RuntimeError('Original clean e878 source dependency required')


def configure_policy(attempt):
    build = attempt / 'build'
    cache = dict(re.findall(r'^([^#/:\n][^:\n]*):[^=\n]+=(.*)$', (build / 'CMakeCache.txt').read_text(), re.MULTILINE))
    expected = {'CMAKE_CXX_COMPILER': sidecar.CXX, 'CMAKE_C_COMPILER': sidecar.CC, 'SYCL_COMPILER': sidecar.CXX, 'SYCL_HOST_COMPILER': sidecar.CXX, 'XPU_SYCL_COMPILER': 'icx'}
    if any(cache.get(name) != value for name, value in expected.items()):
        raise RuntimeError('True icpx CXX/SYCL/host and C icx required')
    identities = list((build / 'CMakeFiles').glob('*/CMakeCXXCompiler.cmake'))
    if len(identities) != 1 or 'set(CMAKE_CXX_COMPILER_ID "IntelLLVM")' not in identities[0].read_text():
        raise RuntimeError('Actual IntelLLVM compiler identity required')
    rules = (build / 'CMakeFiles/rules.ninja').read_text()
    ninja = (build / 'build.ninja').read_text()
    if 'common_ops' in ninja:
        raise RuntimeError('No common_ops/full target in finite SOURCE graph')
    for forbidden in ('/usr/bin/c++', '/usr/bin/g++', '/usr/bin/gcc', 'fsycl-host-compiler=gcc'):
        if forbidden in ninja + rules:
            raise RuntimeError('Unexpected GNU compiler command')
    targets = {}
    for name in NAMES:
        target = 'sgl-ops-sycl-' + name
        old = list((SOURCE / 'build/src/CMakeFiles' / (target + '.dir')).rglob('*.Release.cmake'))
        fresh = list((build / 'CMakeFiles' / (target + '.dir')).rglob('*.Release.cmake'))
        if len(old) != 1 or len(fresh) != 1:
            raise RuntimeError('Expected one original SOURCE translation unit')
        old_text = old[0].read_text()
        fresh_text = fresh[0].read_text()
        for variable in ('SYCL_compile_flags', 'SYCL_compile_definitions', 'CMAKE_HOST_FLAGS'):
            if sidecar.setting(old_text, variable) != sidecar.setting(fresh_text, variable):
                raise RuntimeError('Original flags changed: ' + name + '/' + variable)
        if sidecar.setting(fresh_text, 'SYCL_executable') != '"' + sidecar.CXX + '"' or sidecar.setting(fresh_text, 'SYCL_host_compiler') != '"' + sidecar.CXX + '"':
            raise RuntimeError('Actual generated SYCL/host command must be icpx')
        old_ninja = (SOURCE / 'build/build.ninja').read_text()
        original_commands = [line.strip() for line in old_ninja.splitlines() if line.strip().startswith('COMMAND =') and target + '.dir/' in line and '-fsycl-link ' in line]
        commands = [line.strip() for line in ninja.splitlines() if line.strip().startswith('COMMAND =') and target + '.dir/' in line and '-fsycl-link ' in line]
        if len(original_commands) != 1 or len(commands) != 1:
            raise RuntimeError('Original/fresh AOT command missing')
        original = shlex.split(original_commands[0])
        actual = shlex.split(commands[0])
        if sidecar.CXX + ' ' not in commands[0] or actual[actual.index('-Xs') + 1] != original[original.index('-Xs') + 1]:
            raise RuntimeError('Original AOT backend policy changed')
        original_flags = [token for token in original if token.startswith('-fsycl') or token.startswith('-spirv-ext=') or token in ('--offload-compress', '-Xspirv-translator')]
        if any(token not in actual for token in original_flags) or '-fsycl-max-parallel-link-jobs=1' not in actual:
            raise RuntimeError('Original AOT/SPIRV/inner1 flag changed')
        rule = re.search(r'^rule CXX_SHARED_LIBRARY_LINKER__' + re.escape(target) + r'_Release\n.*?(?=\n\n)', rules, re.MULTILINE | re.DOTALL)
        if not rule or sidecar.CXX + ' ' not in rule.group(0):
            raise RuntimeError('Actual final native link driver must be icpx')
        targets[target] = {'original_generated_sha256': sidecar.digest(old[0]), 'actual_generated_sha256': sidecar.digest(fresh[0]), 'actual_compile_metadata': fresh_text, 'actual_AOT_command': commands[0], 'actual_host_link_rule': rule.group(0)}
    return {'compiler_id': 'IntelLLVM', **expected, 'original_source_flags_preserved': True, 'NV_original_GRF512_preserved': True, 'targets': targets}


def stage(attempt):
    baseline = ARTIFACT / 'native-diagnostic/source/site'
    site = attempt / 'site'
    shutil.copytree(baseline, site)
    package = site / 'sgl_kernel'
    registrations = list(package.glob('common_ops*.so'))
    if len(registrations) != 1 or sidecar.digest(registrations[0]) != 'e63d50179a696799f4e6b57bb60d50956282b37537a788ffa05581dc85cd8fea':
        raise RuntimeError('Genuine legacy registration package required')
    native = {}
    for name in NAMES:
        filename = 'libsgl-ops-sycl-' + name + '.so'
        library = attempt / 'build' / filename
        original = SOURCE / 'build/src' / filename
        symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(library)], text=True)
        old_symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(original)], text=True)
        original_exports = {line.split()[-1] for line in old_symbols.splitlines() if len(line.split()) >= 3 and line.split()[-2] == 'T'}
        actual_exports = {line.split()[-1] for line in symbols.splitlines() if len(line.split()) >= 3 and line.split()[-2] == 'T'}
        if not original_exports.issubset(actual_exports):
            raise RuntimeError('Original ABI/exports changed: ' + name)
        dynamic = subprocess.check_output(['readelf', '-d', str(library)], text=True)
        if f'Library soname: [{filename}]' not in dynamic or '_port_payload' in dynamic:
            raise RuntimeError('Original SONAME or native provenance changed')
        shutil.copyfile(library, package / filename)
        if sidecar.digest(package / filename) != sidecar.digest(library):
            raise RuntimeError('Fresh native staging SHA differs')
        native[filename] = {'sha256': sidecar.digest(library), 'original_library_sha256': sidecar.digest(original), 'original_exports_mangled': sorted(original_exports), 'actual_dt_needed': re.findall(r'\(NEEDED\).*\[([^\]]+)\]', dynamic)}
    changed = []
    package_files = {}
    for path in site.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(site)
        package_files[str(relative)] = sidecar.digest(path)
        if sidecar.digest(path) != sidecar.digest(baseline / relative):
            changed.append(path.name)
    if sorted(changed) != sorted(native):
        raise RuntimeError('Legacy package modified beyond four fresh native libraries')
    for name in ('nvfp4.py', 'cutlass_moe.py'):
        original = sidecar.git(SOURCE, 'show', REVISION + ':python/sgl_kernel/' + name)
        if hashlib.sha256(original).hexdigest() != sidecar.digest(package / name):
            raise RuntimeError('Original NV Python API source changed')
    sidecar.save(attempt / 'source-binary-manifest.json', {'mode': 'four fresh original SOURCE icpx tested-native libraries in isolated genuine legacy package; NOT full d983 package / NOT full icpx registration / NOT official AFTER', 'source_revision': REVISION, 'source_cutlass_revision': CUTLASS_REVISION, 'site': str(site), 'registration_revision': '63fae55', 'registration_sha256': sidecar.digest(registrations[0]), 'native': native, 'package_files_sha256': package_files, 'runtime_import_verified': False, 'accuracy_verified': False, 'performance_verified': False, 'GPU_started': False, 'official_full_acceptance': False})


def container_build(run):
    if set(os.sched_getaffinity(0)) != set(range(80, 96)):
        raise RuntimeError('NV CPU80–95 required')
    attempt = DIRECTORY / 'runs' / run
    verify_original()
    sidecar.save(attempt / 'container-supervisor.json', {'pid': os.getpid(), 'proc_start_ticks': sidecar.process_ticks(os.getpid()), 'run': run})
    environment = dict(os.environ, MAX_JOBS='1', CMAKE_BUILD_PARALLEL_LEVEL='2', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH='/opt/gfx-deps/profiling_tools/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
    sidecar.run_command(['/opt/gfx-deps/venv/bin/python', '-c', 'import torch; assert not torch.xpu.is_initialized(); print("TORCH_CPU_IMPORT_PROBE=0 GPU_INITIALIZED=false", torch.__version__)'], attempt / 'torch-cpu-probe', environment, timeout=300, memory_guard=True)
    command = ['cmake', '-S', str(DIRECTORY), '-B', str(attempt / 'build'), '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_CXX_COMPILER={sidecar.CXX}', f'-DCMAKE_C_COMPILER={sidecar.CC}', f'-DSYCL_COMPILER={sidecar.CXX}', f'-DSYCL_HOST_COMPILER={sidecar.CXX}', '-DXPU_SYCL_COMPILER=icx', '-DDPCPP_SYCL_TARGET=cri', '-DPYTHON3_EXECUTABLE=/opt/gfx-deps/venv/bin/python', f'-DSGL_OPS_XPU_ROOT={attempt / "source"}', f'-DORIGINAL_CUTLASS_ROOT={CUTLASS}']
    sidecar.run_command(command, attempt / 'configure', environment, memory_guard=True)
    sidecar.save(attempt / 'compiler-policy-verified.json', configure_policy(attempt))
    command = ['cmake', '--build', str(attempt / 'build'), '--parallel', '2', '--target', *['sgl-ops-sycl-' + name for name in NAMES]]
    sidecar.run_command(command, attempt / 'tested-native-build', environment, memory_guard=True)
    verify_original()
    sidecar.save(attempt / 'source-readonly-postverify.json', sidecar.verify_snapshot({'source_worktree': str(SOURCE), 'source_revision': REVISION}, attempt / 'source'))
    stage(attempt)
    sidecar.save(attempt / 'complete.json', {'actual_native_exit': 0, 'normal_isolated_legacy_package_staged': True, 'GPU_started': False, 'default_all_or_common_ops_built': False, 'full_package_acceptance': False})


def host_build(run):
    verify_original()
    attempt = DIRECTORY / 'runs' / run
    with (DIRECTORY / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt.mkdir(parents=True, exist_ok=False)
        snapshot = attempt / 'source'
        snapshot.mkdir()
        archive = attempt / 'source.tar'
        with archive.open('wb') as output:
            subprocess.run(['git', '-C', str(SOURCE), 'archive', REVISION], stdout=output, check=True)
        with tarfile.open(archive) as contents:
            contents.extractall(snapshot, filter='data')
        gate = snapshot / 'cmake/BuildFlags.cmake'
        patch = '*** Begin Patch\n*** Update File: ' + str(gate) + '\n@@\n-' + sidecar.GATE_OLD + '\n+' + sidecar.GATE_NEW.replace('\n', '\n+') + '\n*** End Patch\n'
        subprocess.run(['apply_patch', patch], check=True)
        sidecar.save(attempt / 'source-snapshot.json', {'source_revision': REVISION, 'archive_sha256': sidecar.digest(archive), **sidecar.verify_snapshot({'source_worktree': str(SOURCE), 'source_revision': REVISION}, snapshot)})
        script = str(Path(__file__).resolve()).replace('/home/syk/', '/root/', 1)
        base = ['/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh', '--workdir', '/root', '--', 'bash', '/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh', '5', '/opt/gfx-deps/venv/bin/python', script]
        environment = dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015')
        try:
            sidecar.run_command([*base, '--container-build', '--run', run], attempt / 'supervisor', environment, timeout=7500)
        except BaseException:
            subprocess.run([*base, '--stop-owned', '--run', run], env=environment, timeout=90, check=False)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--container-build', action='store_true')
    parser.add_argument('--stop-owned', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run):
        raise RuntimeError('Unique safe run name required')
    if args.stop_owned:
        attempt = DIRECTORY / 'runs' / args.run
        results = {}
        for stem in ('torch-cpu-probe', 'configure', 'tested-native-build'):
            path = attempt / (stem + '.owned.json')
            if path.is_file():
                record = json.loads(path.read_text())
                results[stem] = sidecar.ProcessTree(0, record['token'], known=record['identities']).drain()
        sidecar.save(attempt / 'stop-owned-drain.json', results)
        if any(not record['drained'] for record in results.values()):
            raise RuntimeError('NV own process tree did not drain')
    elif args.container_build:
        container_build(args.run)
    else:
        host_build(args.run)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, sidecar.interrupted)
    signal.signal(signal.SIGINT, sidecar.interrupted)
    main()
