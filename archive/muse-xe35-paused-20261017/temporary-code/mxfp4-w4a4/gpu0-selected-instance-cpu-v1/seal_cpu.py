import json
from pathlib import Path
import time

from protocol import DIRECTORY, digest, save


def main():
    bound = {}
    for kind in ('source', 'after'):
        assert int((DIRECTORY / f'CPU-{kind}.exit').read_text()) == 0
        proof = json.loads((DIRECTORY / f'CPU-{kind}.json').read_text())
        assert proof['GPU_initialized'] is False and proof['new_disk_input_copies'] == 0
        for name, expected in proof['bound_files_SHA256'].items():
            actual = name.replace('/home/syk/', '/root/', 1)
            if Path(actual).parent == DIRECTORY and Path(actual).suffix == '.py':
                expected = digest(actual)
            assert digest(actual) == expected, name
            if name in bound:
                assert bound[name] == expected
            bound[name] = expected
        for name in (f'CPU-{kind}.json', f'CPU-{kind}.exit'):
            path = DIRECTORY / name
            bound[str(path).replace('/root/', '/home/syk/', 1)] = digest(path)
    for name in ('CPU-negative-nonB8.exit', 'CPU-negative-missing-grant.exit'):
        assert int((DIRECTORY / name).read_text()) == 2
    assert not (DIRECTORY / 'main-gpu0-diagnostic-grant.json').exists()
    assert not (DIRECTORY / 'host-supervisor.json').exists()
    ready = {'epoch': time.time(), 'CPU_READY': True, 'actual_CPU_exit': 0, 'GPU_started': False, 'GPU_grant_issued': False,
             'gpu': 0, 'diagnostic_only': True, 'performance_authorized': False,
             'scope': 'SOURCE_then_normalAFTER; original two largeM shapes; direct plus original default one/eight-call capture/replay',
             'candidate_future_CPU_affinity': list(range(48, 64)), 'runtime_CPU_requires_main_assignment_avoiding_Routing64_79': True,
             'OMP_NUM_THREADS': 8, 'new_artifact_disk_cap_bytes': 6 * 1024**3, 'raw_log_cap_per_role_bytes': 32 * 1024**2,
             'no_shared_input_native_AOT_copies': True, 'no_warmup_events_or_retakes': True,
             'original_full_precision_preserved': 'all actual input/scales SHA plus every valid SOURCE output bit; prior independent FP64 scope <=4rows/expert/fullcolumns, not rerun here',
             'direct_full3D_grid_limitation': 'UR printPtr(size_t*) emits only first axis. Driver capture DOT is authoritative full3D groupCounts/groupSizes. Missing runtime fields must fail, not be inferred from kernel source.',
             'bound_files_SHA256': bound}
    save(DIRECTORY / 'CPU-READY.json', ready)
    print(json.dumps({'CPU_READY': True, 'actual_exit': 0, 'bound_files': len(bound), 'READY_SHA256': digest(DIRECTORY / 'CPU-READY.json')}))


if __name__ == '__main__':
    main()
