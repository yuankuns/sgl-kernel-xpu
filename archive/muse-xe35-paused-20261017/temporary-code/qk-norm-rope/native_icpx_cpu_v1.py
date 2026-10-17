import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import tarfile
import time
import traceback


ARTIFACT = Path(__file__).resolve().parent
RUN = ARTIFACT / 'native-icpx-cpu-v1'
HOME = Path('/root' if str(ARTIFACT).startswith('/root/') else '/home/syk')
PERSONAL = HOME / 'worktrees/sgl-kernel-xpu'
HISTORY = PERSONAL / 'review-validation-artifacts/qk-before-after-20261017'
CUTLASS = HOME / 'worktrees/cutlass_internal/muse-w4a4-e87870b'
PIN = 'e87870bb7aacf83bfb3be9da685f148316b8df23'
TARGET = 'sgl-ops-sycl-FusedQKNormRope'
LIBRARY = f'lib{TARGET}.so'
CXX = '/opt/gfx-deps/dpcpp/bin/icpx'
CC = '/opt/gfx-deps/dpcpp/bin/icx'
ROLES = (
    ('optimized-source', PERSONAL / 'muse-spark-qk-norm-rope-main-20261014',
     'feed315712ea91e6becbb983f3a4d56ac0cc3b96', 'after'),
    ('preopt', PERSONAL / 'qk-norm-rope-main-baseline',
     'ab3a26c43a38e6aea7d73cc1349e4b5e210b772b', 'before'),
)
PRECISION = (
    '-fhonor-nans', '-fhonor-infinities', '-fno-associative-math',
    '-fno-approx-func', '-no-ftz', '-fsycl-fp64-conv-emu',
    '-fsycl-targets=spir64_gen',
)


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2) + '\n')


def checked(arguments):
    return subprocess.check_output(arguments, text=True).strip()


def files(directory):
    return {str(path.relative_to(directory)): sha(path)
            for path in sorted(directory.rglob('*')) if path.is_file()}


def prepare():
    assert not RUN.exists(), 'Fresh versioned run only; no automatic retry'
    assert checked(['git', '-C', str(CUTLASS), 'rev-parse', 'HEAD']) == PIN
    assert not checked(['git', '-C', str(CUTLASS), 'status', '--porcelain'])
    RUN.mkdir()
    tracked = subprocess.check_output(['git', '-C', str(CUTLASS), 'ls-files', '-z'])
    dependencies = {name.decode(): sha(CUTLASS / name.decode())
                    for name in tracked.split(b'\0') if name}
    save(RUN / 'original-deps.json', {'path': str(CUTLASS), 'head': PIN,
         'clean': True, 'tracked_files': dependencies,
         'historical_preopt_cutlass_mount_HEAD_unverified': True})
    for role, checkout, reference, historical in ROLES:
        directory = RUN / role
        directory.mkdir()
        archive = directory / 'tracked-source.tar'
        with archive.open('wb') as output:
            subprocess.run(['git', '-C', str(checkout), 'archive', '--format=tar', reference],
                           stdout=output, check=True)
        source = directory / 'source'
        source.mkdir()
        with tarfile.open(archive) as stream:
            stream.extractall(source, filter='data')
        original = files(source)
        flags = source / 'cmake/BuildFlags.cmake'
        contents = flags.read_bytes()
        gate = b'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")'
        message = b'Currently only support GCC compiler on Linux as CXX compiler.'
        assert contents.count(gate) == contents.count(message) == 1
        flags.write_bytes(contents.replace(gate, b'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")').replace(message, b'Currently only support GNU or IntelLLVM compiler on Linux as CXX compiler.'))
        configured = files(source)
        assert {name for name in original if original[name] != configured[name]} == {'cmake/BuildFlags.cmake'}
        save(directory / 'inputs.json', {'role': role, 'ref': reference,
             'checkout': str(checkout), 'archive_sha256': sha(archive),
             'all_tracked_original_files': original, 'build_snapshot_files': configured,
             'only_changed_path': 'cmake/BuildFlags.cmake',
             'historical_site': str(HISTORY / historical / 'site'),
             'original_dependency_pin': PIN, 'QK_target_only': TARGET,
             'default_full_or_common_ops_or_PORT_built': False})
    save(RUN / 'prepared.json', {'actual_exit': 0, 'created_epoch': time.time(),
         'roles_serial': [role for role, *_ in ROLES], 'outer_jobs': 2,
         'inner_MAX_JOBS': 1, 'cpus': '32-47', 'compiler_GiB': 8,
         'tree_GiB': 16, 'GPU_authorized': False})


def processes():
    result = {}
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            status = (directory / 'status').read_text()
            affinity = re.search(r'^Cpus_allowed_list:\s*(.*)$', status, re.MULTILINE)[1]
            result[int(directory.name)] = {
                'ppid': int(fields[1]), 'starttime': fields[19],
                'rss': int(fields[21]) * os.sysconf('SC_PAGE_SIZE'),
                'command': (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'),
                'name': (directory / 'comm').read_text().strip(), 'affinity': affinity,
            }
        except (FileNotFoundError, ProcessLookupError, PermissionError, TypeError):
            continue
    return result


def descendants(root, snapshot, retained):
    selected = {root} | {pid for pid, start in retained.items()
                         if pid in snapshot and snapshot[pid]['starttime'] == start}
    while True:
        added = {pid for pid, record in snapshot.items() if record['ppid'] in selected}
        if added <= selected:
            break
        selected |= added
    return {pid: snapshot[pid] for pid in selected if pid in snapshot}


def drain(root, retained):
    for attempt in range(70):
        snapshot = processes()
        selected = descendants(root, snapshot, retained)
        for pid, record in selected.items():
            retained[pid] = record['starttime']
        if not selected:
            return
        for pid in sorted(selected, reverse=True):
            try:
                os.kill(pid, signal.SIGTERM if attempt < 25 else signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.2)
    raise RuntimeError('Own command descendant subtree did not drain')


def run(arguments, prefix, environment):
    prefix.with_suffix('.command').write_text(shlex.join(arguments) + '\n')
    retained = {}
    policy_failure = None
    with prefix.with_suffix('.log').open('w') as output, prefix.with_suffix('.resources.jsonl').open('w') as resources:
        process = subprocess.Popen(arguments, stdout=output, stderr=subprocess.STDOUT, env=environment)
        save(prefix.with_suffix('.pid.json'), {'pid': process.pid, 'ppid': os.getpid(),
             'started_epoch': time.time(), 'argv': arguments, 'supervisor_affinity': sorted(os.sched_getaffinity(0))})
        while True:
            snapshot = processes()
            selected = descendants(process.pid, snapshot, retained)
            for pid, record in selected.items():
                retained[pid] = record['starttime']
            tree_rss = sum(record['rss'] for record in selected.values())
            compiler_rss = {}
            for pid, record in selected.items():
                if re.match(r'^(?:icpx|icx|clang.*|ocloc.*)$', record['name']):
                    branch = descendants(pid, selected, {})
                    compiler_rss[pid] = sum(child['rss'] for child in branch.values())
            resources.write(json.dumps({'epoch': time.time(), 'tree_rss': tree_rss,
                            'compiler_descendant_rss': compiler_rss, 'processes': selected}) + '\n')
            resources.flush()
            if tree_rss > 16 * 1024 ** 3 or any(value > 8 * 1024 ** 3 for value in compiler_rss.values()):
                policy_failure = '8GiB compiler-descendant/16GiB tree RSS limit exceeded'
            if any(record['affinity'] != '32-47' for record in selected.values()):
                policy_failure = 'Command descendant escaped lane2 CPU32-47'
            if policy_failure:
                drain(process.pid, retained)
                break
            if process.poll() is not None:
                break
            time.sleep(0.5)
        actual = process.wait()
        remaining = descendants(process.pid, processes(), retained)
        if remaining:
            drain(process.pid, retained)
        prefix.with_suffix('.exit').write_text(f'{actual}\n')
        save(prefix.with_suffix('.result.json'), {'actual_exit': actual,
             'policy_failure': policy_failure, 'descendants_drained': True,
             'command_sha256': sha(prefix.with_suffix('.command')), 'log_sha256': sha(prefix.with_suffix('.log'))})
    if actual != 0 or policy_failure:
        raise RuntimeError(f'{prefix.name} actual_exit={actual}, policy_failure={policy_failure}; STOP without retry/glue')


def verify_commands(directory):
    build = directory / 'build'
    cache = (build / 'CMakeCache.txt').read_text()
    for key, executable in (('CMAKE_CXX_COMPILER', CXX), ('CMAKE_C_COMPILER', CC),
                            ('SYCL_COMPILER', CXX), ('SYCL_HOST_COMPILER', CXX)):
        assert re.search(rf'^{key}:[^=]+={re.escape(executable)}$', cache, re.MULTILINE)
    identities = {}
    for language, executable in (('CXX', CXX), ('C', CC)):
        identity = next((build / 'CMakeFiles').glob(f'*/CMake{language}Compiler.cmake'))
        text = identity.read_text()
        assert f'set(CMAKE_{language}_COMPILER_ID "IntelLLVM")' in text
        assert f'set(CMAKE_{language}_COMPILER "{executable}")' in text
        identities[language] = {'path': str(identity), 'sha256': sha(identity)}
    recipes = list((build / 'src/CMakeFiles').rglob('*FusedQKNormRope*.Release.cmake'))
    assert len(recipes) == 1
    recipe = recipes[0].read_text()
    assert all(flag in recipe for flag in PRECISION)
    assert f'set(SYCL_executable "{CXX}")' in recipe
    assert f'set(SYCL_host_compiler "{CXX}")' in recipe
    assert f'{CUTLASS}/include' in recipe
    ninja = (build / 'build.ninja').read_text()
    rules = (build / 'CMakeFiles/rules.ninja').read_text()
    text = ninja + rules + recipe
    assert not any(compiler in text for compiler in ('/usr/bin/c++', '/usr/bin/g++', '/usr/bin/gcc'))
    device = next(line for line in ninja.splitlines()
                  if line.startswith('  COMMAND = ') and '-fsycl-link ' in line
                  and 'FusedQKNormRope_sycl_device_obj.o' in line)
    for required in (CXX, '-fsycl-targets=spir64_gen', '-fsycl-max-parallel-link-jobs=1',
                     'device\\ cri', '-cl-fp32-correctly-rounded-divide-sqrt',
                     '-cl-poison-unsupported-fp64-kernels', '-cl-intel-enable-auto-large-GRF-mode',
                     '-cl-intel-greater-than-4GB-buffer-required',
                     '+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate'):
        assert required in device, required
    host_rule = re.search(rf'^rule CXX_SHARED_LIBRARY_LINKER__{TARGET}_Release\n.*?\n\n',
                          rules, re.MULTILINE | re.DOTALL)[0]
    assert CXX in host_rule
    save(directory / 'compiler-policy-verified.json', {'compiler_id': 'IntelLLVM',
         'cxx': CXX, 'c': CC, 'sycl': CXX, 'host': CXX, 'torch_finder_family': 'icx',
         'precision_flags': list(PRECISION), 'real_identity_files': identities,
         'compile_recipe': str(recipes[0]), 'compile_recipe_sha256': sha(recipes[0]),
         'cache_sha256': sha(build / 'CMakeCache.txt'), 'ninja_sha256': sha(build / 'build.ninja'),
         'rules_sha256': sha(build / 'CMakeFiles/rules.ninja'),
         'device_link_recipe': device, 'host_link_rule': host_rule})


def container_build():
    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    save(RUN / 'container-supervisor.json', {'pid': os.getpid(), 'ppid': os.getppid(),
         'affinity': sorted(os.sched_getaffinity(0)), 'started_epoch': time.time(),
         'outer_jobs': 2, 'inner_jobs': 1, 'per_compiler_GiB': 8, 'tree_GiB': 16})
    environment = dict(os.environ)
    environment.update({'MAX_JOBS': '1', 'CMAKE_BUILD_PARALLEL_LEVEL': '2',
                        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                        'PYTHONDONTWRITEBYTECODE': '1', 'GIT_CONFIG_COUNT': '1',
                        'GIT_CONFIG_KEY_0': 'safe.directory', 'GIT_CONFIG_VALUE_0': str(CUTLASS)})
    for role, checkout, reference, historical in ROLES:
        directory = RUN / role
        source = directory / 'source'
        build = directory / 'build'
        inputs = json.loads((directory / 'inputs.json').read_text())
        assert files(source) == inputs['build_snapshot_files']
        configure = ['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja',
                     '-DCMAKE_BUILD_TYPE=Release', '-DDPCPP_SYCL_TARGET=cri',
                     f'-DCMAKE_CXX_COMPILER={CXX}', f'-DCMAKE_C_COMPILER={CC}',
                     f'-DSYCL_COMPILER={CXX}', f'-DSYCL_HOST_COMPILER={CXX}',
                     '-DXPU_SYCL_COMPILER=icx', '-DPYTHON3_EXECUTABLE=/opt/gfx-deps/venv/bin/python',
                     '-DSKBUILD_SABI_COMPONENT=Development.SABIModule', '-DSKBUILD_SABI_VERSION=3.10',
                     f'-DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL={CUTLASS}']
        run(configure, directory / 'configure', environment)
        verify_commands(directory)
        run(['cmake', '--build', str(build), '--target', TARGET, '--parallel', '2', '--verbose'],
            directory / 'qk-target-build', environment)
        assert files(source) == inputs['build_snapshot_files']
        native = build / 'src' / LIBRARY
        run(['nm', '-D', '--defined-only', '-C', str(native)], directory / 'native-nm', environment)
        symbols = (directory / 'native-nm.log').read_text()
        assert re.search(r'\bT at::native::xpu::fused_qk_norm_rope\(', symbols)
        run(['readelf', '-Ws', str(native)], directory / 'native-readelf-symbols', environment)
        run(['readelf', '-d', str(native)], directory / 'native-readelf-needed', environment)
        dynamic = (directory / 'native-readelf-symbols.log').read_text().split('Symbol table \'.symtab\'')[0]
        assert any('GLOBAL' in line and 'DEFAULT' in line and ' UND ' not in line
                   and 'fused_qk_norm_rope' in line for line in dynamic.splitlines())
        site = directory / 'site'
        shutil.copytree(HISTORY / historical / 'site', site)
        shutil.copy2(native, site / 'sgl_kernel' / LIBRARY)
        original_site = files(HISTORY / historical / 'site')
        fresh_site = files(site)
        assert set(original_site) == set(fresh_site)
        assert {name for name in original_site if original_site[name] != fresh_site[name]} == {f'sgl_kernel/{LIBRARY}'}
        save(directory / 'site-manifest.json', fresh_site)
        import_environment = {**environment, 'PYTHONPATH': str(site),
                              'LD_LIBRARY_PATH': '/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:' + environment.get('LD_LIBRARY_PATH', '')}
        code = ('import pathlib,sgl_kernel,torch; '
                f'assert pathlib.Path(sgl_kernel.__file__).resolve().is_relative_to(pathlib.Path({str(site)!r})); '
                'assert callable(sgl_kernel.fused_qk_norm_rope); '
                'assert hasattr(torch.ops.sgl_kernel,"fused_qk_norm_rope"); '
                'print("NORMAL_SOURCE_PACKAGE_IMPORT_CPU_ONLY_OK",sgl_kernel.__file__)')
        run(['/opt/gfx-deps/venv/bin/python', '-c', code], directory / 'normal-package-import', import_environment)
        save(directory / 'native-complete.json', {'role': role, 'ref': reference,
             'fresh_native_sha256': sha(native), 'site': str(site),
             'site_manifest_sha256': sha(directory / 'site-manifest.json'),
             'original_deps_pin': PIN, 'QK_target_actual_exit': 0,
             'normal_package_CPU_import_actual_exit': 0, 'native_ABI_export_verified': True,
             'scope': 'Original normal complete package with ONLY genuine original-ABI QK native freshly rebuilt icpx; other natives retained legacy; NOT full icpx SOURCE package and NOT PORT acceptance',
             'GPU_accuracy_or_performance_executed': False})
        print(f'ROLE_COMPLETE={role} native_sha256={sha(native)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare', 'container'), required=True)
    arguments = parser.parse_args()
    try:
        prepare() if arguments.mode == 'prepare' else container_build()
    except BaseException:
        traceback.print_exc()
        if RUN.exists():
            (RUN / f'{arguments.mode}-supervisor.exit').write_text('1\n')
        raise SystemExit(1)
    (RUN / f'{arguments.mode}-supervisor.exit').write_text('0\n')
