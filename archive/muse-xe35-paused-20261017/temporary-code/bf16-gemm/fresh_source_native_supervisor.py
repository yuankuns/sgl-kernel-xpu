import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def tracked_inventory(source):
    names = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z']).decode().split('\0')
    return {name: digest(source / name) for name in sorted(names) if name}


def main():
    os.umask(0o022)
    root = Path(__file__).resolve().parent
    plan = json.loads((root / 'fresh-source-icpx-plan.json').read_text())
    stage = root / 'fresh-source-icpx-v1'
    stage.mkdir()
    write_json(stage / 'supervisor.started.json', {'pid': os.getpid(), 'epoch': time.time(), 'module': root.name,
                                                 'CPU_bounded_grant': True, 'GPU_grant': False})
    status = 1
    try:
        original = Path(plan['source_tree'])
        head = subprocess.check_output(['git', '-C', str(original), 'rev-parse', 'HEAD'], text=True).strip()
        assert head == plan['source_HEAD']
        archive = subprocess.check_output(['git', '-C', str(original), 'archive', head])
        snapshot = stage / 'source'
        snapshot.mkdir()
        committed = {}
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            for member in stream.getmembers():
                if member.isdir():
                    continue
                assert member.isfile() and '..' not in Path(member.name).parts and not Path(member.name).is_absolute()
                data = stream.extractfile(member).read()
                assert (original / member.name).read_bytes() == data, member.name
                destination = snapshot / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                destination.chmod(member.mode)
                committed[member.name] = hashlib.sha256(data).hexdigest()
        assert tracked_inventory(original) == committed
        vendor = Path(plan['dependency']['path_host'])
        assert subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip() == plan['dependency']['pin']
        assert not subprocess.check_output(['git', '-C', str(vendor), 'status', '--porcelain'])
        vendor_files = tracked_inventory(vendor)
        flags = snapshot / 'cmake/BuildFlags.cmake'
        before = flags.read_text()
        expected = before.replace('if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")',
                                  'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")', 1)
        expected = expected.replace('Currently only support GCC compiler on Linux as CXX compiler.',
                                    'Currently only support GNU or IntelLLVM compiler on Linux as CXX compiler.', 1)
        assert sum(first != second for first, second in zip(before.splitlines(), expected.splitlines())) == 2
        patch = '\n'.join(['*** Begin Patch', '*** Update File: ' + str(flags), '@@',
                           '-if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")',
                           '+if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")', '@@',
                           '-  message("Not compiling with XPU. Currently only support GCC compiler on Linux as CXX compiler.")',
                           '+  message("Not compiling with XPU. Currently only support GNU or IntelLLVM compiler on Linux as CXX compiler.")',
                           '*** End Patch'])
        (stage / 'compiler-support.patch').write_text(patch + '\n')
        subprocess.run(['apply_patch', patch], check=True)
        assert flags.read_text() == expected
        frozen = {name: digest(snapshot / name) for name in committed}
        assert [name for name in committed if frozen[name] != committed[name]] == ['cmake/BuildFlags.cmake']
        write_json(stage / 'source-preverify.json', {'source_HEAD': head, 'original_files': committed,
                    'snapshot_files': frozen, 'vendor_pin': plan['dependency']['pin'], 'vendor_files': vendor_files,
                    'verified': True, 'epoch': time.time(), 'compiler_support_changed_lines': 2})
        container_stage = str(stage).replace('/home/syk/', '/root/', 1)
        command = ['env', 'SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015',
                   '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh',
                   '--workdir', container_stage, '--', 'bash',
                   '/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh',
                   str(plan['CPU_lane']), 'bash', '-c',
                   'export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=2 ONEAPI_DEVICE_SELECTOR=level_zero:cpu; '
                   'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; '
                   'exec python -B "$1"', '_',
                   str(root / 'fresh_source_native_container.py').replace('/home/syk/', '/root/', 1)]
        (stage / 'container.command').write_text(shlex.join(command) + '\n')
        with (stage / 'container.log').open('w') as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            write_json(stage / 'container.started.json', {'host_wrapper_pid': child.pid, 'epoch': time.time()})
            status = child.wait()
        (stage / 'container.exit').write_text(str(status) + '\n')
        assert tracked_inventory(original) == committed
        assert subprocess.check_output(['git', '-C', str(original), 'rev-parse', 'HEAD'], text=True).strip() == head
        assert {name: digest(snapshot / name) for name in committed} == frozen
        assert tracked_inventory(vendor) == vendor_files
        assert subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip() == plan['dependency']['pin']
        assert not subprocess.check_output(['git', '-C', str(vendor), 'status', '--porcelain'])
        write_json(stage / 'source-postverify.json', {'verified': True, 'epoch': time.time(),
                   'source_HEAD': head, 'original_tracked_all_byte_unchanged': True,
                   'snapshot_all_byte_unchanged': True, 'vendor_all_tracked_byte_unchanged': True,
                   'source_preverify_sha256': digest(stage / 'source-preverify.json'), 'actual_container_exit': status})
    except Exception:
        status = status or 1
        traceback.print_exc()
    (stage / 'supervisor.exit').write_text(str(status) + '\n')
    print(json.dumps({'supervisor_actual_exit': status, 'artifact': str(stage), 'GPU_started': False}), flush=True)
    raise SystemExit(status)


if __name__ == '__main__':
    main()
