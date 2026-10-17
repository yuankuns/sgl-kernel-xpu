import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import traceback
from pathlib import Path


ICPX = '/opt/gfx-deps/dpcpp/bin/icpx'
ICX = '/opt/gfx-deps/dpcpp/bin/icx'


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def resources(group):
    processes = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
            if int(fields[2]) != group:
                continue
            rss = int((directory / 'statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
            processes.append({'pid': int(directory.name), 'rss_bytes': rss,
                              'command': (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return processes


def run_monitored(stage, name, command):
    (stage / (name + '.command')).write_text(shlex.join(command) + '\n')
    peak_group = 0
    peak_process = 0
    guard = None
    with (stage / (name + '.log')).open('w') as output, (stage / (name + '.resources.jsonl')).open('w') as samples:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(stage / (name + '.started.json'), {'pid': child.pid, 'epoch': time.time(), 'command': command})
        while child.poll() is None:
            processes = resources(child.pid)
            total = sum(process['rss_bytes'] for process in processes)
            maximum = max((process['rss_bytes'] for process in processes), default=0)
            peak_group = max(peak_group, total)
            peak_process = max(peak_process, maximum)
            samples.write(json.dumps({'epoch': time.time(), 'group_rss_bytes': total, 'processes': processes}) + '\n')
            samples.flush()
            if maximum > 8 * 1024**3 or total > 16 * 1024**3:
                guard = '8GiB process or 16GiB lane aggregate memory budget exceeded'
                os.killpg(child.pid, signal.SIGTERM)
                time.sleep(3)
                if resources(child.pid):
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(0.5)
        status = child.wait()
        residual = resources(child.pid)
        if residual:
            os.killpg(child.pid, signal.SIGTERM)
            time.sleep(3)
            if resources(child.pid):
                os.killpg(child.pid, signal.SIGKILL)
            guard = guard or 'Residual build process group required drain'
        (stage / (name + '.exit')).write_text(str(status) + '\n')
        write_json(stage / (name + '.resource-summary.json'), {'actual_process_exit': status,
                   'peak_group_rss_bytes': peak_group, 'peak_process_rss_bytes': peak_process,
                   'guard': guard, 'group_drained': not resources(child.pid)})
        if status != 0 or guard:
            raise RuntimeError(f'{name} failed: actual_exit={status}; guard={guard}')


def compiler_policy(root, stage, plan):
    build = stage / 'build'
    cache = (build / 'CMakeCache.txt').read_text()
    expected = {'CMAKE_CXX_COMPILER': ICPX, 'CMAKE_C_COMPILER': ICX,
                'SYCL_COMPILER': ICPX, 'SYCL_EXECUTABLE': ICPX, 'SYCL_HOST_COMPILER': ICPX}
    for key, value in expected.items():
        assert re.search(r'^' + key + r':[^=]+=' + re.escape(value) + r'$', cache, re.MULTILINE), key
    compiler = next(build.glob('CMakeFiles/*/CMakeCXXCompiler.cmake'))
    assert 'set(CMAKE_CXX_COMPILER_ID "IntelLLVM")' in compiler.read_text()
    stem = plan['compile_target'].removeprefix('sgl-ops-sycl-')
    script = next(build.rglob('*' + stem + '*.Release.cmake'))
    text = script.read_text()
    assert 'set(SYCL_executable "' + ICPX + '")' in text
    assert 'set(SYCL_host_compiler "' + ICPX + '")' in text
    flags = re.search(r'^set\(SYCL_compile_flags (.+)\) # list$', text, re.MULTILINE).group(1)
    historical = json.loads((root / 'official-v3-icpx-tools/source-compiler-provenance.json').read_text())
    original_flags = next(line for line in historical['generated_compile_sets'] if line.startswith('set(SYCL_compile_flags '))
    assert 'set(SYCL_compile_flags ' + flags + ') # list' == original_flags
    ninja = (build / 'build.ninja').read_text()
    lines = ninja.splitlines()
    target = plan['compile_target']
    device = next(lines[index + 1] for index, line in enumerate(lines)
                  if line.startswith('build ') and target in line and 'sycl_device_obj.o' in line and ': CUSTOM_COMMAND' in line)
    assert ICPX + ' -fPIC -fsycl ' in device and '-fsycl-max-parallel-link-jobs=1' in device
    if root.name == 'bf16-gemm':
        assert '-cl-intel-256-GRF-per-thread' in device
    else:
        assert '-cl-intel-256-GRF-per-thread' not in device and '-cl-intel-512-GRF-per-thread' not in device
    rules = (build / 'CMakeFiles/rules.ninja').read_text()
    link_rule = rules.split('rule CXX_SHARED_LIBRARY_LINKER__' + target + '_Release\n', 1)[1].split('\n\n', 1)[0]
    assert ICPX in link_rule and '/usr/bin/c++' not in link_rule
    commands = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands', target], text=True)
    (stage / 'target-expanded.commands').write_text(commands)
    assert '/usr/bin/c++' not in commands and not re.search(r'\b(g\+\+|gcc)\b', commands)
    write_json(stage / 'compiler-policy-verified.json', {'compiler_id': 'IntelLLVM', 'CXX': ICPX,
               'SYCL': ICPX, 'SYCL_host': ICPX, 'device_link': ICPX, 'native_link': ICPX, 'C': ICX,
               'kernel_compile_flags_byte_equal_historical': True, 'compile_flags': flags,
               'actual_device_link_command': device, 'actual_native_link_rule': link_rule,
               'CMakeCache_sha256': digest(build / 'CMakeCache.txt'), 'build_ninja_sha256': digest(build / 'build.ninja'),
               'generated_SYCL_recipe_sha256': digest(script), 'original_precision_AOT_GRF_preserved': True})


def stage_source_package(root, stage, plan):
    target = plan['compile_target']
    native = stage / 'build/src' / ('lib' + target + '.so')
    original = root / 'source/site/sgl_kernel'
    baseline = json.loads((root / 'source-provenance.json').read_text())
    old = original / native.name
    assert digest(old) == baseline['binary_sha256'][native.name]
    symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(native)], text=True)
    old_symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(old)], text=True)
    suffix = 'bf16_gemm_nt_xpu' if root.name == 'bf16-gemm' else 'mxfp4_small_permutation'
    required = [line.split()[-1] for line in old_symbols.splitlines() if suffix in line]
    assert required and all(symbol in symbols.split() for symbol in required), 'Actual native ABI export mismatch; do not add glue'
    dynamic = subprocess.check_output(['readelf', '-d', str(native)], text=True)
    assert 'Library soname: [' + native.name + ']' in dynamic, 'Actual native SONAME mismatch; do not add glue'
    (stage / 'native-export.nm').write_text(symbols)
    (stage / 'native-dynamic.readelf').write_text(dynamic)
    destination = stage / 'site/sgl_kernel'
    destination.mkdir(parents=True)
    before_files = {}
    for path in sorted(original.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        relative = path.relative_to(original)
        checksum = digest(path)
        if path.name in baseline['binary_sha256']:
            assert checksum == baseline['binary_sha256'][path.name], path
        before_files[str(relative)] = checksum
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        if path.name == native.name:
            shutil.copy2(native, output)
        else:
            subprocess.run(['cp', '--reflink=auto', '--preserve=mode,timestamps', str(path), str(output)], check=True)
    current = {str(path.relative_to(destination)): digest(path) for path in sorted(destination.rglob('*')) if path.is_file()}
    assert current.keys() == before_files.keys()
    assert [name for name in current if current[name] != before_files[name]] == [native.name]
    common = [name for name in current if name.startswith('common_ops.')]
    assert len(common) == 1 and current[common[0]] == before_files[common[0]]
    write_json(stage / 'source-native-binary-manifest.json', {'source_HEAD': plan['source_HEAD'],
               'scope': 'genuine original normal SOURCE package with fresh icpx tested-native candidate ONLY; not all-icpx package; never AFTER',
               'native_candidate_built': True, 'normal_import_and_GPU_accuracy_not_run': True,
               'native_SHA256': digest(native), 'native_ABI_export_checked': required, 'native_SONAME': native.name,
               'source_pin': plan['dependency']['pin'], 'SOURCE_original_files': before_files, 'site_files': current,
               'changed_package_files': [native.name], 'source_preverify_sha256': digest(stage / 'source-preverify.json'),
               'compiler_policy_sha256': digest(stage / 'compiler-policy-verified.json'),
               'actual_configure_exit': int((stage / 'configure.exit').read_text()),
               'actual_native_build_exit': int((stage / 'native-build.exit').read_text()), 'GPU_started': False})


def main():
    os.umask(0o022)
    root = Path(__file__).resolve().parent
    plan = json.loads((root / 'fresh-source-icpx-plan.json').read_text())
    stage = root / 'fresh-source-icpx-v1'
    status = 1
    try:
        assert os.environ['MAX_JOBS'] == '1' and os.environ['CMAKE_BUILD_PARALLEL_LEVEL'] == '2'
        assert os.sched_getaffinity(0) == set(range(plan['CPU_lane'] * 16, plan['CPU_lane'] * 16 + 16))
        assert os.environ['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:cpu'
        assert not (stage / 'build').exists()
        command = shlex.split(plan['configure_proposed_command'].replace('\\\n', ' '))
        run_monitored(stage, 'configure', command)
        compiler_policy(root, stage, plan)
        run_monitored(stage, 'native-build', ['cmake', '--build', str(stage / 'build'),
                      '--target', plan['compile_target'], '--parallel', '2'])
        stage_source_package(root, stage, plan)
        status = 0
    except Exception:
        traceback.print_exc()
    (stage / 'native-container-supervisor.exit').write_text(str(status) + '\n')
    print(json.dumps({'actual_exit': status, 'GPU_started': False, 'normal_import_tested': False}), flush=True)
    raise SystemExit(status)


if __name__ == '__main__':
    main()
