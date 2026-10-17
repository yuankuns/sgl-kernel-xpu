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
import time
import uuid

from process_tree import ProcessTree, TOKEN_KEY, budget_reason

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = ('memory', 'position', 'logits-vocab')
CXX = '/opt/gfx-deps/dpcpp/bin/icpx'
CC = '/opt/gfx-deps/dpcpp/bin/icx'
GATE_OLD = 'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")'
GATE_NEW = 'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR\n   CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(value):
    before, after = ('/root/', '/home/syk/') if str(ROOT).startswith('/home/syk/') else ('/home/syk/', '/root/')
    return Path(value.replace(before, after, 1) if value.startswith(before) else value)


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2) + '\n')
    path.chmod(0o644)


def process_ticks(process_id):
    return Path(f'/proc/{process_id}/stat').read_text().rsplit(')', 1)[1].split()[19]


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def git(source, *arguments):
    return subprocess.check_output(['git', '-c', f'safe.directory={source}', '-C', str(source), *arguments])


def metadata(component):
    directory = ROOT / component / 'fresh-source-icpx-v1'
    return directory, json.loads((directory / 'inputs.json').read_text())


def verify_inputs(component):
    directory, config = metadata(component)
    source = local_path(config['source_worktree'])
    if git(source, 'rev-parse', 'HEAD').decode().strip() != config['source_revision']:
        raise RuntimeError('Authoritative source revision changed')
    for relative, expected in config['source_files_sha256'].items():
        original = git(source, 'show', config['source_revision'] + ':' + relative)
        if hashlib.sha256(original).hexdigest() != expected:
            raise RuntimeError('Pinned source/helper differs: ' + relative)
    for relative, expected in config['artifact_inputs_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError('Reviewed baseline input changed: ' + relative)
    for name, record in config['kernels'].items():
        original = git(source, 'show', config['source_revision'] + ':' + record['source'])
        if original != (ROOT / component / 'source-baseline/xe35' / (name + '.cpp')).read_bytes():
            raise RuntimeError('Baseline kernel differs from authoritative pin')
    dependency = local_path(config['cutlass_source'])
    if git(dependency, 'rev-parse', 'HEAD').decode().strip() != config['cutlass_revision'] or git(dependency, 'status', '--porcelain').strip():
        raise RuntimeError('Original clean source dependency required')
    return config


def verify_snapshot(config, snapshot):
    source = local_path(config['source_worktree'])
    entries = git(source, 'ls-tree', '-r', '-z', config['source_revision']).split(b'\0')
    expected_paths = set()
    for entry in entries:
        if not entry:
            continue
        metadata_text, name_bytes = entry.split(b'\t', 1)
        mode, kind, blob = metadata_text.decode().split()
        if kind != 'blob':
            raise RuntimeError('Unexpected gitlink in source snapshot')
        name = name_bytes.decode()
        expected_paths.add(name)
        path = snapshot / name
        content = os.readlink(path).encode() if mode == '120000' else path.read_bytes()
        if name == 'cmake/BuildFlags.cmake':
            original = git(source, 'show', config['source_revision'] + ':' + name).decode()
            if original.count(GATE_OLD) != 1 or content != original.replace(GATE_OLD, GATE_NEW).encode():
                raise RuntimeError('Only reviewed two-line IntelLLVM gate allowed')
        elif hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() != blob:
            raise RuntimeError('SOURCE body/helper changed: ' + name)
    actual_paths = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob('*') if path.is_file() or path.is_symlink()}
    if actual_paths != expected_paths:
        raise RuntimeError('Source snapshot inventory changed')
    return {'original_tree': git(source, 'rev-parse', config['source_revision'] + '^{tree}').decode().strip(), 'files_checked': len(expected_paths), 'only_source_delta': ['cmake/BuildFlags.cmake'], 'all_kernel_bodies_helpers_preserved': True}


def prepare(component, run):
    directory, config = metadata(component)
    verify_inputs(component)
    attempt = directory / 'runs' / run
    attempt.mkdir(parents=True, exist_ok=False)
    snapshot = attempt / 'source'
    snapshot.mkdir()
    source = local_path(config['source_worktree'])
    archive = attempt / 'original-source.tar'
    with archive.open('wb') as output:
        subprocess.run(['git', '-C', str(source), 'archive', config['source_revision']], stdout=output, check=True)
    with tarfile.open(archive) as contents:
        contents.extractall(snapshot, filter='data')
    gate = snapshot / 'cmake/BuildFlags.cmake'
    patch = '*** Begin Patch\n*** Update File: ' + str(gate) + '\n@@\n-' + GATE_OLD + '\n+' + GATE_NEW.replace('\n', '\n+') + '\n*** End Patch\n'
    subprocess.run(['apply_patch', patch], check=True)
    save(attempt / 'source-snapshot.json', {'component': component, 'source_revision': config['source_revision'], 'archive_sha256': digest(archive), 'compiler_gate_sha256': digest(gate), **verify_snapshot(config, snapshot)})
    return attempt


def stop_group(process, tracker, stem):
    result = tracker.drain()
    save(stem.with_suffix('.drain.json'), result)
    if process.poll() is None:
        process.wait(timeout=10)
    if not result['drained']:
        raise RuntimeError('Owned descendants remain after multi-PGID drain')


def run_command(command, stem, environment, timeout=3600, memory_guard=False):
    stem.with_suffix('.command').write_text(shlex.join(command) + '\n')
    start = time.time()
    token = uuid.uuid4().hex
    environment = dict(environment, **{TOKEN_KEY: token})
    reason = None
    with stem.with_suffix('.log').open('x') as output, stem.with_suffix('.resources.jsonl').open('x') as resources:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        tracker = ProcessTree(process.pid, token)
        save(stem.with_suffix('.started.json'), {'pid': process.pid, 'pgid': process.pid, 'proc_start_ticks': process_ticks(process.pid), 'epoch': start, 'command': command, 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'token': token})
        try:
            while True:
                metrics = tracker.sample()
                resources.write(json.dumps(metrics) + '\n')
                resources.flush()
                save(stem.with_suffix('.owned.json'), {'token': token, 'identities': metrics['identities']})
                reason = budget_reason(metrics) if memory_guard else None
                if time.time() - start > timeout:
                    reason = 'bounded command timeout'
                if reason:
                    stop_group(process, tracker, stem)
                    break
                if process.poll() is not None:
                    metrics = tracker.sample()
                    if metrics['live']:
                        reason = 'root exited while owned descendants remained'
                        stop_group(process, tracker, stem)
                    break
                time.sleep(0.25)
        except BaseException:
            previous_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
            previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                stop_group(process, tracker, stem)
            finally:
                stem.with_suffix('.exit').write_text(str(process.poll()) + '\n')
                signal.signal(signal.SIGTERM, previous_term)
                signal.signal(signal.SIGINT, previous_int)
            raise
    status = process.wait()
    final = tracker.sample()
    save(stem.with_suffix('.drain.json'), {'drained': not final['live'], 'live_after': final['live'], 'identities': final['identities'], 'groups_signalled': sorted(tracker.groups_signalled)})
    stem.with_suffix('.exit').write_text(str(status) + '\n')
    save(stem.with_suffix('.terminal.json'), {'actual_exit': status, 'elapsed_seconds': time.time() - start, 'stop_reason': reason, 'all_descendants_drained': not final['live'], 'compiler_rss_limit_gib': 8, 'tree_rss_limit_gib': 16, 'available_floor_gib': 96})
    if status != 0 or reason or final['live']:
        raise RuntimeError(f'Actual command failure {status}: {stem}; {reason}')


def setting(text, name):
    match = re.search(r'^set\(' + re.escape(name) + r' (.*)\)(?: #.*)?$', text, re.MULTILINE)
    if not match:
        raise RuntimeError('Missing actual generated compiler metadata: ' + name)
    return match.group(1)


def verify_compilers(component, attempt):
    directory, config = metadata(component)
    build = attempt / 'build'
    cache = dict(re.findall(r'^([^#/:\n][^:\n]*):[^=\n]+=(.*)$', (build / 'CMakeCache.txt').read_text(), re.MULTILINE))
    expected = {'CMAKE_CXX_COMPILER': CXX, 'CMAKE_C_COMPILER': CC, 'SYCL_COMPILER': CXX, 'SYCL_HOST_COMPILER': CXX, 'XPU_SYCL_COMPILER': 'icx'}
    if any(cache.get(name) != value for name, value in expected.items()):
        raise RuntimeError('Real CXX/SYCL/host/link icpx; C icx required')
    identities = list((build / 'CMakeFiles').glob('*/CMakeCXXCompiler.cmake'))
    if len(identities) != 1 or 'set(CMAKE_CXX_COMPILER_ID "IntelLLVM")' not in identities[0].read_text():
        raise RuntimeError('Real IntelLLVM compiler ID required')
    rules = (build / 'CMakeFiles/rules.ninja').read_text()
    ninja = (build / 'build.ninja').read_text()
    for forbidden in ['/usr/bin/c++', '/usr/bin/g++', '/usr/bin/gcc', 'fsycl-host-compiler=gcc']:
        if forbidden in rules + ninja:
            raise RuntimeError('Unexpected GNU compiler command')
    proof = {}
    for name, record in config['kernels'].items():
        target = 'sgl-ops-sycl-' + name
        generated = list((build / 'CMakeFiles' / (target + '.dir')).rglob('*.Release.cmake'))
        if len(generated) != 2:
            raise RuntimeError('Expected original SOURCE plus reviewed alias translation unit')
        for path in generated:
            text = path.read_text()
            for variable, value in record['original_compile_settings'].items():
                if setting(text, variable) != value:
                    raise RuntimeError('Original precision/host flags changed: ' + variable)
            if setting(text, 'SYCL_executable') != '"' + CXX + '"' or setting(text, 'SYCL_host_compiler') != '"' + CXX + '"':
                raise RuntimeError('Actual generated SYCL/host command is not icpx')
        commands = [line.strip() for line in ninja.splitlines() if line.strip().startswith('COMMAND =') and target + '.dir/' in line and '-fsycl-link ' in line]
        if len(commands) != 1 or CXX + ' ' not in commands[0]:
            raise RuntimeError('Actual icpx AOT command required')
        tokens = shlex.split(commands[0])
        if tokens[tokens.index('-Xs') + 1] != record['original_aot_backend_options']:
            raise RuntimeError('Original AOT backend options changed')
        for token in record['original_aot_flag_tokens']:
            if token not in tokens:
                raise RuntimeError('Original AOT flag lost: ' + token)
        rule = re.search(r'^rule CXX_SHARED_LIBRARY_LINKER__' + re.escape(target) + r'_Release\n.*?(?=\n\n)', rules, re.MULTILINE | re.DOTALL)
        if not rule or CXX + ' ' not in rule.group(0):
            raise RuntimeError('Actual final shared-link driver is not icpx')
        proof[target] = {'aot_command': commands[0], 'host_link_rule': rule.group(0), 'compile_metadata_sha256': {str(path.relative_to(build)): digest(path) for path in generated}}
    return {'compiler_id': 'IntelLLVM', **expected, 'source_flags_AOT_preserved': True, 'targets': proof}


def build_component(component, run):
    directory, config = metadata(component)
    attempt = directory / 'runs' / run
    if set(os.sched_getaffinity(0)) != set(range(112, 128)):
        raise RuntimeError('Sidecar must use CPU112–127')
    save(attempt / 'container-supervisor.json', {'pid': os.getpid(), 'proc_start_ticks': process_ticks(os.getpid()), 'component': component, 'run': run})
    environment = dict(os.environ, MAX_JOBS='1', CMAKE_BUILD_PARALLEL_LEVEL='2', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH='/opt/gfx-deps/profiling_tools/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
    run_command(['/opt/gfx-deps/venv/bin/python', '-c', 'import torch; assert not torch.xpu.is_initialized(); print("TORCH_CPU_IMPORT_PROBE=0 GPU_INITIALIZED=false", torch.__version__)'], attempt / 'torch-cpu-probe', environment, timeout=300, memory_guard=True)
    configure = ['cmake', '-S', str(directory), '-B', str(attempt / 'build'), '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_CXX_COMPILER={CXX}', f'-DCMAKE_C_COMPILER={CC}', f'-DSYCL_COMPILER={CXX}', f'-DSYCL_HOST_COMPILER={CXX}', '-DXPU_SYCL_COMPILER=icx', '-DDPCPP_SYCL_TARGET=cri', f'-DSGL_OPS_XPU_ROOT={attempt / "source"}', '-DPYTHON3_EXECUTABLE=/opt/gfx-deps/venv/bin/python']
    run_command(configure, attempt / 'configure', environment, memory_guard=True)
    save(attempt / 'compiler-policy-verified.json', verify_compilers(component, attempt))
    targets = ['sgl-ops-sycl-' + name for name in config['kernels']]
    run_command(['cmake', '--build', str(attempt / 'build'), '--parallel', '2', '--target', *targets], attempt / 'tested-native-build', environment, memory_guard=True)
    verify_inputs(component)
    snapshot = verify_snapshot(config, attempt / 'source')
    native = {}
    for name in config['kernels']:
        library = attempt / 'build' / ('libsgl-ops-sycl-' + name + '.so')
        old_library = ROOT / component / 'source-baseline-build' / library.name
        symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(library)], text=True)
        old_symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(old_library)], text=True)
        expected_symbols = {line.split()[-1] for line in old_symbols.splitlines() if len(line.split()) >= 3 and line.split()[-2] == 'T'}
        actual_symbols = {line.split()[-1] for line in symbols.splitlines() if len(line.split()) >= 3 and line.split()[-2] == 'T'}
        if not expected_symbols.issubset(actual_symbols):
            raise RuntimeError('Original ABI/reviewed aliases hidden or changed')
        dynamic = subprocess.check_output(['readelf', '-d', str(library)], text=True)
        if f'Library soname: [{library.name}]' not in dynamic:
            raise RuntimeError('Original native SONAME changed')
        native[library.name] = {'sha256': digest(library), 'defined_original_and_alias_symbols': sorted(expected_symbols), 'dt_needed': re.findall(r'\(NEEDED\).*\[([^\]]+)\]', dynamic)}
    save(attempt / 'tested-native-source-manifest.json', {'component': component, 'mode': 'SOURCE-only fresh IntelLLVM finite targets; NOT default-all/full/AFTER', 'source_revision': config['source_revision'], 'source_snapshot': snapshot, 'cutlass_revision': config['cutlass_revision'], 'source_input_manifest_sha256': digest(directory / 'inputs.json'), 'compiler_policy_sha256': digest(attempt / 'compiler-policy-verified.json'), 'tested_native_actual_exit': 0, 'native': native, 'canonical_common_ops_compatibility': 'original and already-reviewed alias mangled exports preserved; actual current canonical common_ops nm/import verification pending main staging', 'normal_package_import_verified': False, 'accuracy_verified': False, 'performance_verified': False, 'GPU_started': False, 'official_full_acceptance': False})


def verify_grant(args):
    if not args.grant or not re.fullmatch(r'[A-Za-z0-9_-]+', args.run or ''):
        raise RuntimeError('Fresh run and explicit main build grant required; prepare-only is not authorization')
    configs = {component: verify_inputs(component) for component in COMPONENTS}
    grant = json.loads(args.grant.read_text())
    expected = {'issuer': 'main', 'scope': 'fresh-source-icpx-v1-tested-native-only', 'run': args.run, 'components': list(COMPONENTS), 'cpu_affinity': list(range(112, 128)), 'outer_jobs': 2, 'inner_jobs': 1, 'gpu_work': False, 'source_revisions': {component: config['source_revision'] for component, config in configs.items()}, 'input_manifest_sha256': {component: digest(metadata(component)[0] / 'inputs.json') for component in COMPONENTS}}
    if any(grant.get(name) != value for name, value in expected.items()):
        raise RuntimeError('Main serial source-only grant mismatch')
    if not args.grant.resolve().is_relative_to(ROOT):
        raise RuntimeError('Main grant must reside under the mounted artifact root')
    return configs


def stop_owned(args):
    attempt = metadata(args.component)[0] / 'runs' / args.run
    results = {}
    for stem in ('torch-cpu-probe', 'configure', 'tested-native-build'):
        path = attempt / (stem + '.owned.json')
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        tracker = ProcessTree(0, record['token'], known=record['identities'])
        results[stem] = tracker.drain()
    path = attempt / 'container-supervisor.json'
    if path.is_file():
        record = json.loads(path.read_text())
        try:
            if process_ticks(record['pid']) == record['proc_start_ticks']:
                os.kill(record['pid'], signal.SIGTERM)
        except (ProcessLookupError, FileNotFoundError):
            pass
    save(attempt / 'stop-owned-drain.json', results)
    if any(not result['drained'] for result in results.values()):
        raise RuntimeError('Owned process-tree cleanup incomplete')


def host_serial(args):
    verify_grant(args)
    if not shutil.which('apply_patch'):
        raise RuntimeError('Host apply_patch required for the sole snapshot compiler gate edit')
    lock_path = Path(__file__).resolve().parent / 'serial-build.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for component in COMPONENTS:
            attempt = prepare(component, args.run)
            container_script = str(Path(__file__).resolve()).replace('/home/syk/', '/root/', 1)
            base = ['/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh', '--workdir', '/root', '--', 'bash', '/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh', '7', '/opt/gfx-deps/venv/bin/python', container_script]
            controls = ['--component', component, '--run', args.run, '--grant', str(args.grant.resolve()).replace('/home/syk/', '/root/', 1)]
            command = [*base, '--container-build', *controls]
            environment = dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015')
            try:
                run_command(command, attempt / 'supervisor', environment, timeout=7500)
            except BaseException:
                subprocess.run([*base, '--stop-owned', *controls], env=environment, timeout=60, check=False)
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--container-build', action='store_true')
    parser.add_argument('--stop-owned', action='store_true')
    parser.add_argument('--component', choices=COMPONENTS)
    parser.add_argument('--run')
    parser.add_argument('--grant', type=Path)
    args = parser.parse_args()
    if args.check_only:
        result = {component: {'source_revision': verify_inputs(component)['source_revision'], 'inputs_sha256': digest(metadata(component)[0] / 'inputs.json')} for component in COMPONENTS}
        print(json.dumps({'mode': 'CPU-only preparation/input check', 'components': result, 'configure_started': False, 'compiler_started': False, 'GPU_started': False}, indent=2))
    elif args.container_build or args.stop_owned:
        if str(ROOT).startswith('/home/syk/') or not args.component or not re.fullmatch(r'[A-Za-z0-9_-]+', args.run or ''):
            raise RuntimeError('Container-only internal build entry')
        verify_grant(args)
        if args.stop_owned:
            stop_owned(args)
        else:
            with (ROOT / 'memory/fresh-source-icpx-v1/container-serial.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                build_component(args.component, args.run)
    else:
        if not str(ROOT).startswith('/home/syk/'):
            raise RuntimeError('Host serial supervisor entry required')
        host_serial(args)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
