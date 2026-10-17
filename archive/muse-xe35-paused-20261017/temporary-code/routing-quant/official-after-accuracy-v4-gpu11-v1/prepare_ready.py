import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    artifact = directory.parent
    assert sorted(os.sched_getaffinity(0)) == list(range(72, 80))
    for name in ['normal-after-precheck', 'CPU-mechanical-preparation', 'CPU-provider', 'CPU-collect-adapted']:
        assert int((directory / f'{name}.exit').read_text()) == 0
    before = (artifact / 'fresh-source-full-accuracy-v1/CPU-collect-original.log').read_text()
    after = (directory / 'CPU-collect-adapted.log').read_text()
    original_nodes = [line.split('::', 1)[1] for line in before.splitlines() if '.py::' in line]
    adapted_nodes = [line.split('::', 1)[1] for line in after.splitlines() if '.py::' in line]
    assert len(original_nodes) == len(adapted_nodes) == 1300
    assert original_nodes == adapted_nodes
    adaptation = json.loads((directory / 'mechanical-adaptation-audit.json').read_text())
    for name, record in adaptation['records'].items():
        assert digest(directory / name) == record['adapted_SHA256']
    for path in directory.rglob('*.py'):
        ast.parse(path.read_text())
    for path in directory.glob('*.sh'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    prior = artifact / 'historical-fresh-source-precision-v1'
    assert int((prior / 'source-paired-precision.exit').read_text()) == 0
    source_records = json.loads((prior / 'round-1-fresh.json').read_text())
    assert len(source_records) == 6
    required = [path for path in directory.rglob('*') if path.is_file() and path.suffix in ['.py', '.sh', '.pt']]
    required += [directory / name for name in ['normal-after-precheck.json', 'fixture-provider-precheck.json', 'mechanical-adaptation-audit.json', 'CPU-original-fixtures.json', 'CPU-collect-adapted.log']]
    required += [artifact / name for name in ['canonical-source-freeze-v4-icpx.json', 'COORDINATOR_FULL_ICPX_AUDIT_v4.json',
                 'canonical-v4-icpx/source-binary-manifest.json', 'canonical-v4-icpx/compiler-policy-verified.json',
                 'fresh-source-icpx-native-v2/source-native-package-manifest.json',
                 'fresh-source-icpx-native-v2/cpu-native-acceptance.json',
                 'fresh-three-source-post-v1/GPU8-actual-drain-and-classification.json',
                 'fresh-three-source-post-v1/activation-boundary-classification.json', 'cooldown.py']]
    required += [prior / name for name in ['round-1-fresh.json', 'source-paired-precision.json']]
    required += [Path(record['accuracy']['output'].replace('/root/', '/home/syk/', 1)) for record in source_records]
    hashes = {str(path): digest(path) for path in required}
    package = json.loads((directory / 'normal-after-precheck.json').read_text())
    ready = {'module': 'routing-quant', 'version': 'canonical-v4-icpx', 'CPU_all_gates_actual_zero': True,
             'CPU_affinity': list(range(72, 80)), 'OMP_NUM_THREADS': 8, 'gpu': 11,
             'created_epoch': time.time(), 'bound_files_sha256': hashes,
             'normal_current_manifest_sha256': package['manifest_sha256'],
             'current_native_count': package['current_native_count'], 'current_mapped_native_sha256': package['mapped_native_hashes'],
             'source_digest': package['source_digest'], 'original_and_adapted_nodeids_count': 1300,
             'nodeids_order_and_cases_equal': True, 'independent_complete_four_component_fixtures': 3,
             'production_shapes': 6, 'SOURCE_full_outputs_round': 1,
             'production_reference_scope': 'original six production refs: IDs exact/weights atol2.5e-4 rtol0; all active FP4 bytes and all used group32 scales/padding/tails zero; no undefined padding-q/slack claim',
             'full_SOURCE_bitwise_comparison_scope': 'every saved valid tensor element; report bitwise separately and preserve any failure',
             'original_1300_gate': 'run all even if independent baseline fixture gate fails; actual exit retained',
             'previous_gpu11_owner_handoff_authority': 'latest explicit user: main16533 terminal and51179 CPU audit0/drain/lock11 released; preparation only',
             'GPU_grant_created_or_acquired': False, 'GPU_launched': False,
             'requires_new_main_grant': True, 'performance_or_compile_authorized': False,
             'no_diag_bridge_or_native_replacement': True}
    (directory / 'AFTER_CPU_READY_MANIFEST.json').write_text(json.dumps(ready, indent=2) + '\n')
    print(json.dumps({name: value for name, value in ready.items() if name not in ['bound_files_sha256', 'current_mapped_native_sha256']}, indent=2))


if __name__ == '__main__':
    main()
