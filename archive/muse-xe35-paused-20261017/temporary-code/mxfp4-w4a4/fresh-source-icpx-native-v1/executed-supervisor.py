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
import sys
import time
import traceback
import zipfile


ROOT = Path(__file__).resolve().parent.parent
ICPX = '/opt/gfx-deps/dpcpp/bin/icpx'
ICX = '/opt/gfx-deps/dpcpp/bin/icx'
PIN = 'e87870bb7aacf83bfb3be9da685f148316b8df23'
FAMILIES = {
    'mxfp4-w4a4': ('muse-spark-mxfp4-w4a4-main-20261014', '54216962c83b2d489702fd38c58d71116f3a6969', ['moe_grouped_mm_mxfp4_w4a4', 'mxfp4_moe_scale_layout']),
    'routing-quant': ('muse-spark-mxfp4-routing-quant-main-20261014', '4cb5e929f84ebcebf98ec6702a26f1ee93190828', ['TopKSigMoid', 'per_token_group_quant_fp4', 'mxfp4_moe_scale_layout']),
}


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def record(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def inventory(root, names=None):
    if names is None:
        names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')[:-1]
    return {name: digest(root / name) for name in sorted(names)}


def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments], text=True).strip()


def resources(root_pid, tracked):
    table = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
            table[int(directory.name)] = {'pid': int(directory.name), 'ppid': int(fields[1]),
                'pgrp': int(fields[2]), 'state': fields[0], 'start_ticks': int(fields[19]),
                'rss_bytes': int((directory / 'statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE'),
                'argv': (directory / 'cmdline').read_bytes().split(b'\0')[:-1]}
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    selected = {pid for pid, item in table.items() if pid == root_pid or tracked.get(pid) == item['start_ticks']}
    while True:
        expanded = selected | {pid for pid, item in table.items() if item['ppid'] in selected}
        if expanded == selected:
            break
        selected = expanded
    output = []
    for pid in sorted(selected):
        item = table[pid]
        tracked[pid] = item['start_ticks']
        if item['state'] != 'Z':
            item['argv'] = [argument.decode(errors='replace') for argument in item['argv']]
            output.append(item)
    return output


def terminate(processes, kind):
    for item in reversed(processes):
        try:
            fields = Path('/proc', str(item['pid']), 'stat').read_text().rsplit(') ', 1)[1].split()
            if int(fields[19]) == item['start_ticks']:
                os.kill(item['pid'], kind)
        except (FileNotFoundError, ProcessLookupError):
            pass


def monitored(stage, name, command, timeout=1800):
    (stage / (name + '.command')).write_text(shlex.join(command) + '\n')
    tracked = {}
    guard = None
    peak_tree = 0
    peak_process = 0
    started = time.monotonic()
    with (stage / (name + '.log')).open('w') as output, (stage / (name + '.resources.jsonl')).open('w') as samples:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        record(stage / (name + '.started.json'), {'pid': child.pid, 'epoch': time.time(), 'command': command})
        while child.poll() is None:
            processes = resources(child.pid, tracked)
            total = sum(item['rss_bytes'] for item in processes)
            maximum = max((item['rss_bytes'] for item in processes), default=0)
            peak_tree = max(peak_tree, total)
            peak_process = max(peak_process, maximum)
            samples.write(json.dumps({'epoch': time.time(), 'tree_rss_bytes': total, 'processes': processes}) + '\n')
            samples.flush()
            if maximum > 8 * 1024**3 or total > 16 * 1024**3 or time.monotonic() - started > timeout:
                guard = '8GiB process / 16GiB tree / stage timeout exceeded'
                terminate(processes, signal.SIGTERM)
                time.sleep(3)
                terminate(resources(child.pid, tracked), signal.SIGKILL)
                break
            time.sleep(0.25)
        status = child.wait()
        if resources(child.pid, tracked):
            guard = guard or 'Residual descendants required drain'
            terminate(resources(child.pid, tracked), signal.SIGTERM)
            time.sleep(3)
            terminate(resources(child.pid, tracked), signal.SIGKILL)
        (stage / (name + '.exit')).write_text(str(status) + '\n')
        record(stage / (name + '.resource-summary.json'), {'actual_exit': status, 'guard': guard,
               'peak_tree_rss_bytes': peak_tree, 'peak_process_rss_bytes': peak_process,
               'drained': not resources(child.pid, tracked), 'coverage': 'recursive PPID descendants and start-time-bound retained PIDs; not PGID-only'})
        if status or guard:
            raise RuntimeError(f'{name}: actual_exit={status}, guard={guard}')


def policy(artifact, stage, stems):
    build = stage / 'build'
    cache = (build / 'CMakeCache.txt').read_text()
    for name, value in {'CMAKE_CXX_COMPILER': ICPX, 'CMAKE_C_COMPILER': ICX, 'SYCL_COMPILER': ICPX,
                        'SYCL_EXECUTABLE': ICPX, 'SYCL_HOST_COMPILER': ICPX, 'XPU_SYCL_COMPILER': 'icx'}.items():
        assert re.search(r'^' + name + r':[^=]+=' + re.escape(value) + r'$', cache, re.MULTILINE), name
    identity = next(build.glob('CMakeFiles/*/CMakeCXXCompiler.cmake')).read_text()
    assert 'set(CMAKE_CXX_COMPILER_ID "IntelLLVM")' in identity
    original = json.loads((artifact / 'source-before-provenance-v3-icpx.json').read_text())['native_builds']
    fallback = json.loads((ROOT / 'routing-quant/source-before-provenance-v3-icpx.json').read_text())['native_builds']
    lines = (build / 'build.ninja').read_text().splitlines()
    results = []
    for stem in stems:
        target = 'sgl-ops-sycl-' + stem
        reference = next(item for item in original + fallback if Path(item['native']).name == 'lib' + target + '.so')
        previous = reference['generated_compile_settings'][0]['fields']
        script = next((build / 'src/CMakeFiles' / (target + '.dir')).rglob('*.Release.cmake'))
        text = script.read_text()
        fields = {name: re.search(r'^set\(' + name + r' (.*)\)(?: #.*)?$', text, re.MULTILINE).group(1)
                  for name in ('SYCL_executable', 'SYCL_host_compiler', 'SYCL_compile_flags', 'CMAKE_HOST_FLAGS', 'SYCL_compile_definitions')}
        assert fields['SYCL_executable'] == '"' + ICPX + '"'
        assert fields['SYCL_host_compiler'] == '"' + ICPX + '"'
        for name in ('SYCL_compile_flags', 'CMAKE_HOST_FLAGS', 'SYCL_compile_definitions'):
            assert fields[name] == previous[name], (stem, name, fields[name], previous[name])
        device = next(lines[index + 1] for index, line in enumerate(lines) if line.startswith('build ') and
                      target in line and 'sycl_device_obj.o' in line and ': CUSTOM_COMMAND' in line)
        assert ICPX + ' -fPIC -fsycl ' in device and '-fsycl-max-parallel-link-jobs=1' in device
        tokens = shlex.split(device)
        old_tokens = shlex.split(reference['actual_device_link_command'])
        def device_options(arguments):
            first = arguments.index('-fPIC')
            last = arguments.index('-o')
            return [argument for argument in arguments[first:last] if not argument.endswith('.o')]
        assert device_options(tokens) == device_options(old_tokens), (stem, device)
        commands = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands', target], text=True)
        assert 'common_ops' not in commands and 'torch_extension' not in commands
        assert '/usr/bin/c++' not in commands and not re.search(r'\b(g\+\+|gcc)\b', commands)
        (stage / (target + '.expanded.commands')).write_text(commands)
        results.append({'target': target, 'generated_settings_sha256': digest(script), 'fields': fields,
                        'device_link': device, 'original_flags_and_definitions_byte_equal': True,
                        'device_options_equal_original': True, 'reference_native': reference['native']})
    record(stage / 'compiler-policy-verified.json', {'compiler_id': 'IntelLLVM', 'CXX': ICPX, 'SYCL': ICPX,
           'SYCL_host': ICPX, 'C': ICX, 'Torch_family_token': 'icx', 'compiler_binary_sha256': digest(ICPX),
           'compiler_version': subprocess.check_output([ICPX, '--version'], text=True), 'targets': results,
           'CMakeCache_sha256': digest(build / 'CMakeCache.txt'), 'ninja_sha256': digest(build / 'build.ninja')})


def package(artifact, stage, stems):
    from wheel.wheelfile import WheelFile
    original_wheel = next((artifact / 'source-wheels').glob('*.whl'))
    expected = json.loads((artifact / 'source-package-provenance.json').read_text())['wheel_sha256']
    assert digest(original_wheel) == expected
    original_site = next((artifact / 'source-venv/lib').glob('python*/site-packages/sgl_kernel'))
    common = next(original_site.glob('common_ops*.so'))
    undefined = {line.split()[-1] for line in subprocess.check_output(['nm', '-D', '--undefined-only', str(common)], text=True).splitlines()}
    replacements = {}
    abi = {}
    for stem in stems:
        name = 'libsgl-ops-sycl-' + stem + '.so'
        native = stage / 'build/src' / name
        old = original_site / name
        old_exports = {line.split()[-1] for line in subprocess.check_output(['nm', '-D', '--defined-only', str(old)], text=True).splitlines()}
        exports = subprocess.check_output(['nm', '-D', '--defined-only', str(native)], text=True)
        required = sorted(old_exports & undefined)
        assert required and set(required) <= {line.split()[-1] for line in exports.splitlines()}, (name, required)
        dynamic = subprocess.check_output(['readelf', '-d', str(native)], text=True)
        assert 'Library soname: [' + name + ']' in dynamic
        (stage / (name + '.nm')).write_text(exports)
        (stage / (name + '.readelf')).write_text(dynamic)
        replacements['sgl_kernel/' + name] = native
        abi[name] = {'required_original_common_ops_symbols': required, 'old_sha256': digest(old), 'fresh_sha256': digest(native)}
    wheels = stage / 'wheels'
    wheels.mkdir()
    fresh_wheel = wheels / original_wheel.name
    before = {}
    after = {}
    with zipfile.ZipFile(original_wheel) as source, WheelFile(fresh_wheel, 'w') as destination:
        for name in source.namelist():
            if name.endswith('/') or name.endswith('.dist-info/RECORD'):
                continue
            content = source.read(name)
            before[name] = hashlib.sha256(content).hexdigest()
            if name in replacements:
                assert before[name] == abi[Path(name).name]['old_sha256']
                content = replacements[name].read_bytes()
            after[name] = hashlib.sha256(content).hexdigest()
            destination.writestr(name, content)
    changed = [name for name in before if before[name] != after[name]]
    assert set(changed) == set(replacements), changed
    monitored(stage, 'create-private-venv', [sys.executable, '-m', 'venv', '--system-site-packages', str(stage / 'venv')], 120)
    python = str(stage / 'venv/bin/python')
    monitored(stage, 'normal-package-install', [python, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '--force-reinstall', str(fresh_wheel)], 120)
    site = next((stage / 'venv/lib').glob('python*/site-packages'))
    for name, expected_hash in after.items():
        assert digest(site / name) == expected_hash, name
    probe = "import sgl_kernel, pathlib, json; root=pathlib.Path(sgl_kernel.__file__).resolve().parent; print(json.dumps({'package':str(root),'mapped':sorted({line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if '/sgl_kernel/' in line and '.so' in line})})); assert str(root)==" + repr(str(site / 'sgl_kernel'))
    monitored(stage, 'normal-package-import', [python, '-c', probe], 120)
    mapped = json.loads((stage / 'normal-package-import.log').read_text().splitlines()[-1])['mapped']
    assert str(site / 'sgl_kernel' / common.name) in mapped
    assert all(str(site / name) in mapped for name in replacements)
    record(stage / 'source-native-package-manifest.json', {'scope': 'original complete normal SOURCE registration/Python/package; only listed native replaced by fresh original-source icpx builds; NOT whole-source fresh/all-icpx; NOT official AFTER',
           'original_wheel_sha256': digest(original_wheel), 'fresh_wheel_sha256': digest(fresh_wheel),
           'original_package_files': before, 'current_install_files': after, 'changed_package_files': changed,
           'ABI': abi, 'mapped_native_sha256': {name: digest(name) for name in mapped},
           'normal_import_exit': 0, 'GPU_started': False, 'reference_precision_not_run': True})


def container(module):
    artifact = ROOT / module
    stage = artifact / 'fresh-source-icpx-native-v1'
    status = 1
    record(stage / 'container.started.json', {'pid': os.getpid(), 'epoch': time.time(), 'CPU_affinity': sorted(os.sched_getaffinity(0))})
    try:
        assert os.sched_getaffinity(0) == set(range(64, 80))
        assert os.environ['MAX_JOBS'] == '1' and os.environ['CMAKE_BUILD_PARALLEL_LEVEL'] == '2'
        assert os.environ['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:cpu'
        command = ['cmake', '-S', str(stage / 'source'), '-B', str(stage / 'build'), '-G', 'Ninja',
                   '-DCMAKE_BUILD_TYPE=Release', '-DDPCPP_SYCL_TARGET=cri', '-DCMAKE_CXX_COMPILER=' + ICPX,
                   '-DCMAKE_C_COMPILER=' + ICX, '-DSYCL_COMPILER=' + ICPX, '-DSYCL_HOST_COMPILER=' + ICPX,
                   '-DXPU_SYCL_COMPILER=icx', '-DPYTHON3_EXECUTABLE=/opt/gfx-deps/venv/bin/python',
                   '-DPython3_EXECUTABLE=/opt/gfx-deps/venv/bin/python', '-DSKBUILD_SABI_COMPONENT=Development.SABIModule',
                   '-DSKBUILD_SABI_VERSION=3.9', '-DSKBUILD_SOABI=abi3',
                   '-DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL=/root/worktrees/cutlass_internal/muse-w4a4-e87870b']
        monitored(stage, 'configure', command, 300)
        stems = FAMILIES[module][2]
        policy(artifact, stage, stems)
        monitored(stage, 'native-build', ['cmake', '--build', str(stage / 'build'), '--parallel', '2', '--target', *['sgl-ops-sycl-' + stem for stem in stems]])
        package(artifact, stage, stems)
        status = 0
    except Exception:
        traceback.print_exc()
    (stage / 'container.exit').write_text(str(status) + '\n')
    return status


def host(module):
    folder, head, stems = FAMILIES[module]
    artifact = ROOT / module
    stage = artifact / 'fresh-source-icpx-native-v1'
    stage.mkdir()
    record(stage / 'supervisor.started.json', {'pid': os.getpid(), 'epoch': time.time(), 'module': module})
    source = Path('/home/syk/worktrees/sgl-kernel-xpu') / folder
    vendor = Path('/home/syk/worktrees/cutlass_internal/muse-w4a4-e87870b')
    status = 1
    before = None
    dependency = None
    snapshot = None
    try:
        assert git(source, 'rev-parse', 'HEAD') == head and not git(source, 'status', '--porcelain')
        assert git(vendor, 'rev-parse', 'HEAD') == PIN and not git(vendor, 'status', '--porcelain')
        before = inventory(source)
        dependency = inventory(vendor)
        (stage / 'source').mkdir()
        archive = subprocess.Popen(['git', '-C', str(source), 'archive', head], stdout=subprocess.PIPE)
        subprocess.run(['tar', '-x', '-C', str(stage / 'source')], stdin=archive.stdout, check=True)
        archive.stdout.close()
        assert archive.wait() == 0 and inventory(stage / 'source', before) == before
        flags = stage / 'source/cmake/BuildFlags.cmake'
        original = flags.read_text()
        gate = 'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")'
        message = next(line for line in original.splitlines() if 'Currently only support GCC compiler' in line)
        new_message = message.replace('only support GCC compiler', 'support GNU and IntelLLVM compilers')
        patch = '*** Begin Patch\n*** Update File: ' + str(flags) + '\n@@\n-' + gate + '\n+' + gate[:-1] + ' OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")\n@@\n-' + message + '\n+' + new_message + '\n*** End Patch\n'
        (stage / 'compiler-gate.patch').write_text(patch)
        subprocess.run(['apply_patch', patch], check=True)
        snapshot = inventory(stage / 'source', before)
        assert [name for name in before if snapshot[name] != before[name]] == ['cmake/BuildFlags.cmake']
        assert flags.read_text() == original.replace(gate, gate[:-1] + ' OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")').replace(message, new_message)
        record(stage / 'source-preverify.json', {'source_HEAD': head, 'original_tracked': before, 'snapshot_tracked': snapshot,
               'dependency_HEAD': PIN, 'dependency_tracked': dependency, 'only_delta': 'cmake/BuildFlags.cmake two-line compiler gate/message',
               'targets': stems, 'GPU_started': False})
        wrapper = '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh'
        container_root = str(ROOT).replace('/home/syk/', '/root/')
        command = [wrapper, '--workdir', str(stage).replace('/home/syk/', '/root/'), '--', 'bash',
                   container_root + '/build-lane.sh', '4', 'env', 'MAX_JOBS=1', 'CMAKE_BUILD_PARALLEL_LEVEL=2',
                   'OMP_NUM_THREADS=1', 'ONEAPI_DEVICE_SELECTOR=level_zero:cpu',
                   '/opt/gfx-deps/venv/bin/python', container_root + '/mxfp4-w4a4/fresh_source_native_v1.py', '--container', '--module', module]
        (stage / 'wrapper.command').write_text('SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 ' + shlex.join(command) + '\n')
        environment = dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015')
        with (stage / 'wrapper.log').open('w') as output:
            child = subprocess.Popen(command, env=environment, stdout=output, stderr=subprocess.STDOUT)
            record(stage / 'wrapper.started.json', {'pid': child.pid, 'epoch': time.time(), 'command': command})
            actual = child.wait()
        (stage / 'wrapper.exit').write_text(str(actual) + '\n')
        assert actual == 0, f'Container wrapper actual_exit={actual}'
        status = 0
    except Exception:
        traceback.print_exc()
    finally:
        try:
            checks = {'source_HEAD_unchanged': git(source, 'rev-parse', 'HEAD') == head,
                      'source_tracked_unchanged': before is not None and inventory(source) == before,
                      'dependency_HEAD_unchanged': git(vendor, 'rev-parse', 'HEAD') == PIN,
                      'dependency_tracked_unchanged': dependency is not None and inventory(vendor) == dependency,
                      'snapshot_unchanged': snapshot is not None and inventory(stage / 'source', before) == snapshot}
            record(stage / 'source-postverify.json', checks)
            assert all(checks.values()), checks
        except Exception:
            traceback.print_exc()
            status = 1
        (stage / 'supervisor.exit').write_text(str(status) + '\n')
    return status


if __name__ == '__main__':
    os.umask(0o022)
    parser = argparse.ArgumentParser()
    parser.add_argument('--container', action='store_true')
    parser.add_argument('--module', choices=list(FAMILIES))
    options = parser.parse_args()
    if options.container:
        raise SystemExit(container(options.module))
    modules = [options.module] if options.module else list(FAMILIES)
    for module in modules:
        result = host(module)
        print(json.dumps({'module': module, 'actual_exit': result, 'GPU_started': False}), flush=True)
        if result:
            raise SystemExit(result)
