import argparse
import json
import os
import traceback

import native_icpx_cpu_v1 as engine


engine.RUN = engine.ARTIFACT / 'native-icpx-cpu-v2'


def dependency_comparison():
    original = json.loads((engine.ARTIFACT / 'native-icpx-cpu-v1/original-deps.json').read_text())
    current = json.loads((engine.RUN / 'original-deps.json').read_text())
    assert original['head'] == current['head'] == engine.PIN
    assert original['tracked_files'] == current['tracked_files']
    assert engine.checked(['git', '-C', str(engine.CUTLASS), 'rev-parse', 'HEAD']) == engine.PIN
    assert not engine.checked(['git', '-C', str(engine.CUTLASS), 'status', '--porcelain'])
    for relative, expected in current['tracked_files'].items():
        assert engine.sha(engine.CUTLASS / relative) == expected, relative
    for role, checkout, reference, historical in engine.ROLES:
        old = json.loads((engine.ARTIFACT / 'native-icpx-cpu-v1' / role / 'inputs.json').read_text())
        fresh = json.loads((engine.RUN / role / 'inputs.json').read_text())
        assert old['all_tracked_original_files'] == fresh['all_tracked_original_files']
        assert old['build_snapshot_files'] == fresh['build_snapshot_files']
        assert old['archive_sha256'] == fresh['archive_sha256']
        source = engine.RUN / role / 'source'
        declared = (source / 'CMakeLists.txt').read_text()
        assert f'set(CUTLASS_SYCL_TAG  "{engine.PIN}")' in declared
    engine.save(engine.RUN / 'e878-filehash-equivalent-comparison.json', {
        'actual_exit': 0, 'clean_actual_HEAD': engine.PIN,
        'original_declared_SOURCE_and_preopt_CRI_pin': engine.PIN,
        'tracked_file_count': len(current['tracked_files']),
        'current_clean_e878_filehashes_equal_v1_clean_e878_filehashes': True,
        'all_original_ref_source_math_helpers_archive_hashes_equal_v1': True,
        'historical_preopt_cutlass_mount_HEAD_or_byte_equivalence_proven': False,
        'scope': 'Fresh preopt uses original-declared clean e878 with identical documented per-file hashes as SOURCE/v1; NOT proof of old vanished /cutlass mounted checkout',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare', 'container'), required=True)
    arguments = parser.parse_args()
    try:
        if arguments.mode == 'prepare':
            engine.prepare()
            dependency_comparison()
        else:
            assert set(os.sched_getaffinity(0)) == set(range(32, 48))
            assert os.environ['LD_LIBRARY_PATH'].startswith('/opt/gfx-deps/profiling_tools/lib:')
            dependency_comparison()
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
