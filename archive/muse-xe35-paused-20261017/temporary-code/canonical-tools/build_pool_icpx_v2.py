import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
POOL = ROOT / 'canonical-icpx-pool-v4'
GLOBAL_LOCK = ROOT / 'canonical-pool/full-pool.lock'
VERSION = 'canonical-v4-icpx'
CXX = '/opt/gfx-deps/dpcpp/bin/icpx'
CC = '/opt/gfx-deps/dpcpp/bin/icx'
SNAPSHOT = POOL / 'source'
BUILD = POOL / 'build'
CUTLASS = ROOT / 'deps/sycl-tla-7b56'
WRAPPER = '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh'
JOBS = 56
MIN_AVAILABLE_GIB = 96


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + '\n')


def memory_available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024 ** 2
    raise RuntimeError('MemAvailable is unavailable')


def process_snapshot():
    result = {}
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            result[int(directory.name)] = {
                'ppid': int(fields[1]),
                'starttime': fields[19],
                'name': (directory / 'comm').read_text().strip(),
                'command': command,
            }
        except (OSError, ValueError):
            continue
    return result


def descendants(root_pid, processes):
    selected = {root_pid} if root_pid in processes else set()
    while True:
        additions = {process_id for process_id, item in processes.items() if item['ppid'] in selected} - selected
        if not additions:
            return selected
        selected.update(additions)


def run_command(command, prefix, environment, guarded=False):
    prefix.with_suffix('.command').write_text(subprocess.list2cmdline(command) + '\n')
    save_json(prefix.with_suffix('.started.json'), {'epoch': time.time(), 'available_gib': memory_available_gib(), 'affinity': sorted(os.sched_getaffinity(0))})
    with prefix.with_suffix('.log').open('x') as output, prefix.with_suffix('.resources.jsonl').open('x') as resources:
        process = subprocess.Popen(command, env=environment, stdout=output, stderr=subprocess.STDOUT)
        terminated_for_memory = False
        while process.poll() is None:
            available = memory_available_gib()
            current = process_snapshot()
            selected = descendants(process.pid, current)
            compilers = [process_id for process_id in selected if current[process_id]['name'] in ('icx', 'icpx', 'clang++', 'g++', 'cc1plus')]
            resources.write(json.dumps({'epoch': time.time(), 'root_pid': process.pid, 'available_gib': available, 'descendant_count': len(selected), 'compiler_pids': compilers}) + '\n')
            resources.flush()
            if guarded and available < MIN_AVAILABLE_GIB and not terminated_for_memory:
                terminated_for_memory = True
                for process_id in selected:
                    if current[process_id]['name'] == 'ninja':
                        os.kill(process_id, signal.SIGINT)
                save_json(prefix.with_suffix('.memory-stop.json'), {'epoch': time.time(), 'available_gib': available, 'threshold_gib': MIN_AVAILABLE_GIB, 'action': 'SIGINT owned Ninja; no automatic retry'})
            time.sleep(3)
        status = process.wait()
    prefix.with_suffix('.exit').write_text(str(status) + '\n')
    if status:
        raise RuntimeError(f'Actual command exit {status}: {prefix}')


def container_build(component, version):
    from overlay_snapshot import verify_inventory
    from stage_install import read_cache, stage

    evidence = ROOT / component / version
    source_manifest = evidence / 'source-snapshot.json'
    manifest = json.loads(source_manifest.read_text())
    environment = dict(os.environ)
    environment.update({
        'PYTHONDONTWRITEBYTECODE': '1',
        'MAX_JOBS': '1',
        'CMAKE_BUILD_PARALLEL_LEVEL': str(JOBS),
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'GIT_CONFIG_COUNT': '1',
        'GIT_CONFIG_KEY_0': 'safe.directory',
        'GIT_CONFIG_VALUE_0': str(CUTLASS),
        'LD_LIBRARY_PATH': '/opt/gfx-deps/profiling_tools/lib:' + os.environ.get('LD_LIBRARY_PATH', ''),
    })
    os.environ.update(environment)
    if set(os.sched_getaffinity(0)) != set(range(128)):
        raise RuntimeError('Canonical compiler pool must be restricted to CPUs 0–127')
    available = memory_available_gib()
    if available < JOBS * 8 + MIN_AVAILABLE_GIB:
        raise RuntimeError(f'Insufficient measured headroom for {JOBS} jobs: {available:.2f} GiB')
    verify_inventory(SNAPSHOT, manifest['files'])
    monitor = BUILD / 'mem_monitor'
    if monitor.exists():
        shutil.copytree(monitor, evidence / 'prior-mem-monitor')
    configure = [
        'cmake', '-S', str(SNAPSHOT), '-B', str(BUILD), '-G', 'Ninja',
        '-DDPCPP_SYCL_TARGET=cri', '-DCMAKE_BUILD_TYPE=Release',
        f'-DCMAKE_CXX_COMPILER={CXX}', f'-DCMAKE_C_COMPILER={CC}',
        f'-DSYCL_COMPILER={CXX}', f'-DSYCL_HOST_COMPILER={CXX}',
        '-DXPU_SYCL_COMPILER=icx',
        '-DPYTHON3_EXECUTABLE=/opt/gfx-deps/venv/bin/python',
        '-DSKBUILD_SABI_COMPONENT=Development.SABIModule', '-DSKBUILD_SABI_VERSION=3.10',
        f'-DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL={CUTLASS}',
        '-DSGL_BUILD_MEM_FILE_LIMIT_GIB=8',
    ]
    run_command(configure, evidence / 'configure', environment)
    cache = read_cache(BUILD / 'CMakeCache.txt')
    for key, expected in {'CMAKE_CXX_COMPILER': CXX, 'CMAKE_C_COMPILER': CC, 'SYCL_COMPILER': CXX, 'SYCL_HOST_COMPILER': CXX}.items():
        if cache.get(key, {}).get('value') != expected:
            raise RuntimeError(f'Compiler configuration mismatch: {key}')
    compiler_files = list((BUILD / 'CMakeFiles').glob('*/CMakeCXXCompiler.cmake'))
    if len(compiler_files) != 1 or 'set(CMAKE_CXX_COMPILER_ID "IntelLLVM")' not in compiler_files[0].read_text():
        raise RuntimeError('Expected real IntelLLVM C++ compiler identity')
    ninja_text = (BUILD / 'build.ninja').read_text()
    rules_text = (BUILD / 'CMakeFiles/rules.ninja').read_text()
    generated_commands = list((BUILD / 'src/CMakeFiles').rglob('*.Release.cmake'))
    if not generated_commands:
        raise RuntimeError('Missing generated SYCL compilation commands')
    command_text = ninja_text + rules_text + '\n'.join(path.read_text() for path in generated_commands)
    for forbidden in ['/usr/bin/c++', '/usr/bin/g++', '/usr/bin/gcc', 'fsycl-host-compiler=gcc']:
        if forbidden in command_text:
            raise RuntimeError(f'Unexpected GNU compiler command: {forbidden}')
    required_flags = ['-fhonor-nans', '-fhonor-infinities', '-fno-associative-math', '-fno-approx-func', '-no-ftz', '-fsycl-targets=spir64_gen']
    for flag in required_flags:
        if flag not in command_text:
            raise RuntimeError(f'Missing production precision or SYCL flag: {flag}')
    save_json(evidence / 'compiler-policy-verified.json', {'compiler_id': 'IntelLLVM', 'cxx': CXX, 'c': CC, 'sycl': CXX, 'host': CXX, 'torch_SYCL_family': 'icx', 'required_flags': required_flags, 'no_GNU_command': True})
    run_command(['cmake', '--build', str(BUILD), '--parallel', str(JOBS)], evidence / 'default-all-build', environment, guarded=True)
    verify_inventory(SNAPSHOT, manifest['files'])
    shutil.copy2(BUILD / 'CMakeCache.txt', evidence / 'CMakeCache.txt')
    shutil.copy2(BUILD / 'build.ninja', evidence / 'build.ninja')
    if monitor.exists():
        shutil.copytree(monitor, evidence / 'current-mem-monitor')
    arguments = argparse.Namespace(
        build_dir=str(BUILD), snapshot_dir=str(SNAPSHOT), snapshot_manifest=str(source_manifest),
        site=str(evidence / 'site'), manifest=str(evidence / 'source-binary-manifest.json'),
        install_log=str(evidence / 'install.log'), build_exit_file=str(evidence / 'default-all-build.exit'),
        build_command_file=str(evidence / 'default-all-build.command'), build_log=str(evidence / 'default-all-build.log'),
        path_map=['/home/syk=/root'], cmake='cmake', readelf='readelf', config='Release',
    )
    stage(arguments)
    (evidence / 'source-binary-manifest.json').chmod(0o644)
    save_json(evidence / 'build-stage-complete.json', {
        'epoch': time.time(), 'component': component, 'default_all_exit': 0, 'fresh_current_graph_install': True,
        'build_provenance': 'icpx-only canonical pool: first family clean, subsequent default-all incremental; no historical objects or libraries copied',
        'compiler_policy': 'IntelLLVM C++ and SYCL icpx; C icx; no GCC host',
        'outer_jobs': JOBS, 'inner_jobs': 1, 'cpu_affinity': list(range(128)),
        'source_digest': manifest['source_digest'], 'formal_runtime_acceptance_complete': False,
    })


def host_build(component, version):
    if component not in {'bf16-gemm', 'fmha', 'qk-norm-rope', 'residual-norm', 'mxfp4-w4a4', 'routing-quant', 'nvfp4-woq', 'moe-permutation', 'memory', 'position', 'logits-vocab'}:
        raise RuntimeError('Unknown module')
    POOL.mkdir(exist_ok=True)
    evidence = ROOT / component / version
    evidence.mkdir(exist_ok=False)
    source = ROOT.parent / 'pr' / f'muse-{component}-xe35'
    with GLOBAL_LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = process_snapshot()
        old_ninjas = {process_id: item for process_id, item in current.items() if item['name'] == 'ninja' and str(ROOT) in item['command'] and 'native' not in item['command']}
        for process_id, item in current.items():
            if item['name'] != 'ninja':
                continue
            try:
                cwd = str(Path(f'/proc/{process_id}/cwd').resolve(strict=True))
            except OSError:
                continue
            if 'muse-upstream-artifacts-20261017' in cwd and 'full' in cwd:
                old_ninjas[process_id] = item
        if old_ninjas:
            raise RuntimeError(f'Old full Ninja remains active: {old_ninjas}')
        save_json(evidence / 'preflight.json', {'epoch': time.time(), 'available_gib': memory_available_gib(), 'old_full_ninjas': old_ninjas, 'gpu_work': False, 'cpu_limit': list(range(128)), 'outer_jobs': JOBS, 'inner_jobs': 1})
        from overlay_snapshot import manifest_digest, source_inventory
        frozen = json.loads((ROOT / component / 'canonical-source-freeze-v4-icpx.json').read_text())
        files, selection = source_inventory(source)
        if files != frozen['files'] or selection != frozen['selection'] or manifest_digest(files) != frozen['source_digest']:
            raise RuntimeError('Reviewed icpx source freeze changed; no build')
        previous_pointer = POOL / 'latest-snapshot-manifest.json'
        command = [sys.executable, str(TOOLS / 'overlay_snapshot.py'), '--source', str(source), '--destination', str(SNAPSHOT), '--manifest', str(evidence / 'source-snapshot.json')]
        if previous_pointer.exists():
            command.extend(['--previous-manifest', json.loads(previous_pointer.read_text())['path']])
        subprocess.run(command, check=True)
        save_json(previous_pointer, {'path': str(evidence / 'source-snapshot.json')})
        container_script = str(TOOLS / 'build_pool_icpx_v2.py').replace('/home/syk/', '/root/', 1)
        wrapper = [WRAPPER, '--workdir', '/root/sgl-kernel-xpu-up', '--', 'taskset', '-c', '0-127', 'python', container_script, '--container-build', '--component', component, '--version', version]
        environment = dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015', PYTHONDONTWRITEBYTECODE='1')
        run_command(wrapper, evidence / 'supervisor', environment)
        subprocess.run([*command, '--verify-only'], check=True)
        save_json(evidence / 'source-readonly-postverify.json', {'verified': True, 'epoch': time.time(), 'source': str(source), 'snapshot_manifest': str(evidence / 'source-snapshot.json')})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--component', required=True)
    parser.add_argument('--version', choices=(VERSION,), default=VERSION)
    parser.add_argument('--container-build', action='store_true')
    arguments = parser.parse_args()
    if '/' in arguments.version or arguments.version in {'.', '..'}:
        raise SystemExit('Invalid evidence version')
    if arguments.container_build:
        container_build(arguments.component, arguments.version)
    else:
        host_build(arguments.component, arguments.version)


if __name__ == '__main__':
    main()
