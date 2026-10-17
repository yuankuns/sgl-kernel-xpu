import argparse
import json
import os
import subprocess
import tarfile
import traceback

import native_icpx_cpu_v2 as previous


engine = previous.engine
engine.RUN = engine.ARTIFACT / 'native-icpx-cpu-v3'


def exact_git(arguments):
    assert arguments[0] == 'git' and '-C' in arguments
    repository = arguments[arguments.index('-C') + 1]
    assert os.path.isabs(repository)
    return ['git', '-c', f'safe.directory={repository}', *arguments[1:]]


def checked_git(arguments):
    return subprocess.check_output(exact_git(arguments), text=True).strip()


engine.checked = checked_git


def prepare_exact_git():
    assert not engine.RUN.exists(), 'Fresh versioned run only; no automatic retry'
    assert checked_git(['git', '-C', str(engine.CUTLASS), 'rev-parse', 'HEAD']) == engine.PIN
    assert not checked_git(['git', '-C', str(engine.CUTLASS), 'status', '--porcelain'])
    engine.RUN.mkdir()
    tracked = subprocess.check_output(exact_git(['git', '-C', str(engine.CUTLASS), 'ls-files', '-z']))
    dependencies = {name.decode(): engine.sha(engine.CUTLASS / name.decode())
                    for name in tracked.split(b'\0') if name}
    engine.save(engine.RUN / 'original-deps.json', {'path': str(engine.CUTLASS),
                'head': engine.PIN, 'clean': True, 'tracked_files': dependencies,
                'historical_preopt_cutlass_mount_HEAD_unverified': True})
    for role, checkout, reference, historical in engine.ROLES:
        directory = engine.RUN / role
        directory.mkdir()
        archive = directory / 'tracked-source.tar'
        arguments = exact_git(['git', '-C', str(checkout), 'archive', '--format=tar', reference])
        (directory / 'archive.command.json').write_text(json.dumps(arguments) + '\n')
        with archive.open('wb') as output:
            subprocess.run(arguments, stdout=output, check=True)
        source = directory / 'source'
        source.mkdir()
        with tarfile.open(archive) as stream:
            stream.extractall(source, filter='data')
        original = engine.files(source)
        flags = source / 'cmake/BuildFlags.cmake'
        contents = flags.read_bytes()
        gate = b'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")'
        message = b'Currently only support GCC compiler on Linux as CXX compiler.'
        assert contents.count(gate) == contents.count(message) == 1
        flags.write_bytes(contents.replace(gate, b'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")').replace(message, b'Currently only support GNU or IntelLLVM compiler on Linux as CXX compiler.'))
        configured = engine.files(source)
        assert {name for name in original if original[name] != configured[name]} == {'cmake/BuildFlags.cmake'}
        engine.save(directory / 'inputs.json', {'role': role, 'ref': reference,
                    'checkout': str(checkout), 'archive_sha256': engine.sha(archive),
                    'all_tracked_original_files': original, 'build_snapshot_files': configured,
                    'only_changed_path': 'cmake/BuildFlags.cmake',
                    'historical_site': str(engine.HISTORY / historical / 'site'),
                    'original_dependency_pin': engine.PIN, 'QK_target_only': engine.TARGET,
                    'default_full_or_common_ops_or_PORT_built': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare', 'container'), required=True)
    arguments = parser.parse_args()
    try:
        if arguments.mode == 'prepare':
            prepare_exact_git()
            previous.dependency_comparison()
        else:
            assert set(os.sched_getaffinity(0)) == set(range(32, 48))
            assert os.environ['LD_LIBRARY_PATH'].startswith('/opt/gfx-deps/profiling_tools/lib:')
            os.environ.update({'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'safe.directory',
                               'GIT_CONFIG_VALUE_0': str(engine.CUTLASS)})
            engine.save(engine.RUN / 'initial-container-supervisor.json', {
                'pid': os.getpid(), 'ppid': os.getppid(),
                'affinity': sorted(os.sched_getaffinity(0)),
                'scoped_Git_safe_directory': str(engine.CUTLASS),
            })
            previous.dependency_comparison()
            probe = (
                'import os,torch; assert not torch.xpu.is_initialized(); '
                'assert torch.zeros(1,device="cpu").device.type=="cpu"; '
                'assert not torch.xpu.is_initialized(); '
                'print("TORCH_CPU_IMPORT_PROBE_OK",torch.__version__,"XPU_INITIALIZED",torch.xpu.is_initialized()); '
                'print("LD_LIBRARY_PATH",os.environ["LD_LIBRARY_PATH"])'
            )
            engine.run(['/opt/gfx-deps/venv/bin/python', '-c', probe],
                       engine.RUN / 'torch-cpu-import-probe', dict(os.environ))
            engine.container_build()
    except BaseException:
        traceback.print_exc()
        if engine.RUN.exists():
            (engine.RUN / f'{arguments.mode}-supervisor.exit').write_text('1\n')
        raise SystemExit(1)
    (engine.RUN / f'{arguments.mode}-supervisor.exit').write_text('0\n')
