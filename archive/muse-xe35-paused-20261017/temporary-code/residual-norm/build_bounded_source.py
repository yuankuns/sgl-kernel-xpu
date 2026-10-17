import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback


ROOT = Path(__file__).resolve().parent
JOB = ROOT / 'source-before-icpx-bounded-v1'
SOURCE = JOB / 'source'
BUILD = JOB / 'build'
CXX = '/opt/gfx-deps/dpcpp/bin/icpx'
CC = '/opt/gfx-deps/dpcpp/bin/icx'
PIN = 'e87870bb7aacf83bfb3be9da685f148316b8df23'
CUTLASS = Path('/root/worktrees/cutlass_internal/muse-w4a4-e87870b')
TARGET = 'sgl-ops-sycl-FusedNormResidual'
PYTHON = ROOT / 'source-full-venv/bin/python'
GIB = 1024 ** 3


def save(name, value):
    (JOB / name).write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_snapshot():
    result = {}
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            result[int(directory.name)] = {'parent': int(fields[1]), 'start': fields[19],
                'rss': int(fields[21]) * os.sysconf('SC_PAGE_SIZE'), 'state': fields[0],
                'name': (directory / 'comm').read_text().strip(),
                'command': (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')}
        except (OSError, ValueError):
            continue
    return result


def descendants(snapshot, parent):
    result = {parent} if parent in snapshot else set()
    while True:
        additions = {pid for pid, entry in snapshot.items() if entry['parent'] in result} - result
        if not additions:
            return result
        result.update(additions)


def drain(seen):
    for phase_signal in [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT]:
        current = process_snapshot()
        for pid, start in seen.items():
            if pid in current and current[pid]['start'] == start:
                try:
                    os.kill(pid, phase_signal)
                except ProcessLookupError:
                    pass
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        current = process_snapshot()
        active = {pid for pid, start in seen.items() if pid in current and current[pid]['start'] == start}
        if not active:
            return
        time.sleep(0.25)
    for pid in active:
        os.kill(pid, signal.SIGKILL)


def run(name, command):
    save(name + '.command.json', command)
    print(f'PHASE {name}: {command}', flush=True)
    seen = {}
    peak = 0
    compiler_peak = 0
    violation = None
    with (JOB / (name + '.log')).open('x') as log, (JOB / (name + '.resources.jsonl')).open('x') as resources:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            snapshot = process_snapshot()
            selected = descendants(snapshot, process.pid)
            seen.update({pid: snapshot[pid]['start'] for pid in selected})
            tracked = {pid for pid, start in seen.items() if pid in snapshot and snapshot[pid]['start'] == start}
            rss = sum(snapshot[pid]['rss'] for pid in tracked) + process_snapshot().get(os.getpid(), {}).get('rss', 0)
            compiler_rss = {pid: sum(snapshot[child]['rss'] for child in descendants(snapshot, pid))
                            for pid in tracked if snapshot[pid]['name'].startswith(('icpx', 'icx', 'clang', 'ocloc', 'llvm-', 'sycl-', 'ld.'))}
            peak = max(peak, rss)
            compiler_peak = max(compiler_peak, max(compiler_rss.values(), default=0))
            resources.write(json.dumps({'epoch': time.time(), 'root': process.pid, 'descendant_pids': sorted(tracked),
                'tree_rss_bytes': rss, 'compiler_subtree_rss_bytes': compiler_rss}) + '\n')
            resources.flush()
            if rss > 16 * GIB or any(value > 8 * GIB for value in compiler_rss.values()):
                violation = {'tree_rss': rss, 'compiler_subtrees': compiler_rss, 'reason': 'RSS budget exceeded; no retry'}
                save(name + '.resource-stop.json', violation)
                drain(seen)
                break
            time.sleep(0.25)
        raw_status = process.wait()
    snapshot = process_snapshot()
    active = {pid: snapshot[pid] for pid, start in seen.items() if pid in snapshot and snapshot[pid]['start'] == start}
    if active:
        drain(seen)
    status = raw_status if raw_status >= 0 else 128 - raw_status
    (JOB / (name + '.exit')).write_text(str(status) + '\n')
    save(name + '.terminal.json', {'raw_wait_status': raw_status, 'exit': status, 'peak_tree_rss': peak,
         'peak_compiler_subtree_rss': compiler_peak, 'resource_violation': violation, 'descendants_drained': True})
    if status or violation or active:
        raise RuntimeError(f'{name} failure actual exit={status}; violation={violation}; leftovers={active}')


def verify_source():
    manifest = json.loads((JOB / 'snapshot-manifest.json').read_text())
    actual = {path.relative_to(SOURCE).as_posix(): {'sha256': sha(path), 'size': path.stat().st_size,
              'mode': path.stat().st_mode & 0o777} for path in SOURCE.rglob('*') if path.is_file()}
    assert actual == manifest['snapshot_files']
    return manifest['source_digest']


def dependency_proof():
    git = ['git', '--git-dir=/root/cutlass_internal/.git/worktrees/muse-w4a4-e87870b', f'--work-tree={CUTLASS}']
    assert subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip() == PIN
    assert not subprocess.check_output(git + ['status', '--porcelain'], text=True).strip()
    return {'pin': PIN, 'clean': True, 'path': str(CUTLASS)}


def package_files(prefix):
    site = prefix / 'lib/python3.12/site-packages/sgl_kernel'
    return {path.relative_to(site).as_posix(): sha(path) for path in site.rglob('*') if path.is_file()}


def main():
    assert set(os.sched_getaffinity(0)) == set(range(48, 64))
    os.environ.update(MAX_JOBS='1', CMAKE_BUILD_PARALLEL_LEVEL='2', OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                      XDG_CACHE_HOME=str(JOB / 'cache'))
    os.environ['LD_LIBRARY_PATH'] = '/opt/gfx-deps/profiling_tools/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
    for name in ['USE_MOE', 'USE_FMHA', 'USE_GDN', 'USE_MLA', 'USE_QSA', 'USE_MLA_SPARSE_FUSED', 'USE_SYCL_JIT']:
        os.environ.pop(name, None)
    assert not BUILD.exists()
    save('runtime.json', {'container_supervisor_pid': os.getpid(), 'cpu': '48-63', 'outer_jobs': 2,
        'MAX_JOBS': 1, 'compiler_budget_gib': 8, 'tree_budget_gib': 16, 'RSS_accounting': 'recursive PID descendants, not PGID',
        'GPU_work': False, 'target': TARGET})
    save('precheck.json', {'source_digest': verify_source(), 'dependency': dependency_proof()})
    run('drivers', [str(PYTHON), '-B', '-c', f'import subprocess; print(subprocess.check_output(["{CXX}","--version"],text=True)); print(subprocess.check_output(["{CC}","--version"],text=True))'])
    run('configure', ['cmake', '-G', 'Ninja', '-S', str(SOURCE), '-B', str(BUILD),
        f'-DPYTHON3_EXECUTABLE={PYTHON}', f'-DPython3_EXECUTABLE={PYTHON}', '-DSKBUILD_SABI_COMPONENT=Development.SABIModule',
        '-DDPCPP_SYCL_TARGET=cri', '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_CXX_COMPILER={CXX}', f'-DCMAKE_C_COMPILER={CC}',
        f'-DSYCL_COMPILER={CXX}', f'-DSYCL_HOST_COMPILER={CXX}', '-DXPU_SYCL_COMPILER=icx',
        f'-DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL={CUTLASS}'])
    cache = {name: value for name, value in re.findall(r'^([^/#\n:]+):[^=\n]*=(.*)$', (BUILD / 'CMakeCache.txt').read_text(), re.MULTILINE)}
    for name, value in {'CMAKE_CXX_COMPILER': CXX, 'CMAKE_C_COMPILER': CC, 'SYCL_COMPILER': CXX, 'SYCL_HOST_COMPILER': CXX}.items():
        assert cache[name] == value
    identity = list((BUILD / 'CMakeFiles').glob('*/CMakeCXXCompiler.cmake'))
    assert len(identity) == 1 and 'set(CMAKE_CXX_COMPILER_ID "IntelLLVM")' in identity[0].read_text()
    run('closure', ['ninja', '-C', str(BUILD), '-t', 'commands', TARGET])
    commands = (JOB / 'closure.log').read_text()
    assert len(commands.splitlines()) == 3
    assert not any(name in commands for name in ['HCHead', 'common_ops', 'sgl-ops-sycl-Device'])
    assert CXX in commands and '/usr/bin/c++' not in commands and '/usr/bin/g++' not in commands
    generated = BUILD / f'src/CMakeFiles/{TARGET}.dir/sycl/{TARGET}_gen_FusedNormResidual.cpp.o.Release.cmake'
    text = generated.read_text()
    assert f'set(SYCL_host_compiler "{CXX}")' in text and f'set(SYCL_executable "{CXX}")' in text
    old = json.loads((ROOT / 'source-compiler-provenance.json').read_text())
    old_flags = next(line for line in old['actual_generated_SYCL_settings'] if line.startswith('set(SYCL_compile_flags '))
    new_flags = next(line for line in text.splitlines() if line.startswith('set(SYCL_compile_flags '))
    assert old_flags == new_flags, (old_flags, new_flags)
    original_device_link = old['actual_recorded_native_commands'][1]
    current_device_link = commands.splitlines()[1]
    assert original_device_link.split(' -Xs ', 1)[1].split(' -o ', 1)[0] == current_device_link.split(' -Xs ', 1)[1].split(' -o ', 1)[0]
    assert '-fsycl-max-parallel-link-jobs=1' in current_device_link
    save('closure-policy.json', {'target': TARGET, 'steps': 3, 'no_Device_common_ops_HCHead_all': True,
        'compiler_id': 'IntelLLVM', 'cxx_sycl_host_AOT_link': CXX, 'c': CC,
        'original_compile_flags_exact': new_flags, 'Device_rebuild_needed': False,
        'generated_command_sha256': sha(generated), 'cache_sha256': sha(BUILD / 'CMakeCache.txt')})
    run('dryrun', ['ninja', '-C', str(BUILD), '-n', TARGET])
    run('native-build', ['cmake', '--build', str(BUILD), '--target', TARGET, '--parallel', '2'])
    native = BUILD / 'src/libsgl-ops-sycl-FusedNormResidual.so'
    run('exports', ['nm', '-D', '--defined-only', str(native)])
    assert any('fused_post_norm_residual_pre_norm' in line and line.split()[1] == 'T' for line in (JOB / 'exports.log').read_text().splitlines())
    run('dynamic', ['readelf', '-d', str(native)])
    assert 'Library soname: [libsgl-ops-sycl-FusedNormResidual.so]' in (JOB / 'dynamic.log').read_text()
    assert 'libsgl-ops-sycl-Device' not in (JOB / 'dynamic.log').read_text()
    run('undefined', ['nm', '-D', '--undefined-only', str(native)])
    prefix = JOB / 'isolated-source-venv'
    run('copy-normal-package', ['cp', '-a', '--reflink=auto', str(ROOT / 'source-full-venv'), str(prefix)])
    before = package_files(prefix)
    destination = prefix / 'lib/python3.12/site-packages/sgl_kernel/libsgl-ops-sycl-FusedNormResidual.so'
    run('install-tested-native', ['cp', str(native), str(destination)])
    after = package_files(prefix)
    assert [name for name in before if before[name] != after[name]] == ['libsgl-ops-sycl-FusedNormResidual.so']
    save('site-sha.json', {'pre': before, 'post': after, 'only_replaced_native': 'libsgl-ops-sycl-FusedNormResidual.so',
        'registration_common_ops_GNU_rebuilt': False, 'tested_native': 'fresh icpx', 'Device_rebuilt': False})
    run('normal-import', [str(prefix / 'bin/python'), '-B', str(ROOT / 'check_bounded_source_import.py')])
    save('postcheck.json', {'source_digest': verify_source(), 'dependency': dependency_proof(),
        'native_sha256': sha(native), 'site_native_sha256': sha(destination), 'source_ref': 'ec29f7a6a14c4f1fac42fb93526236ca03ab894b',
        'native_target_actual_exit': 0, 'normal_package_import_exit': 0, 'GPU_accuracy_perf_executed': False,
        'official_full_acceptance': False})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        (JOB / 'failure.txt').write_text(traceback.format_exc())
        raise
