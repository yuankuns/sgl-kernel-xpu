import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    root = Path(__file__).resolve().parent.parent
    directory = root / 'routing-quant/fresh-current-source-native-v3-preparation'
    grant_path = directory / 'main-current-source-four-native-compile-grant-v1.json'
    assert not grant_path.exists()
    assert not (root / 'routing-quant/fresh-current-source-icpx-native-v3').exists()
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 80))
    assert int((directory / 'CPU-preparation-v2.exit').read_text()) == int((directory / 'root-readonly-protection.exit').read_text()) == 0
    preparation = json.loads((directory / 'CPU-preparation.json').read_text())
    assert preparation['coverage_four_native'] and not preparation['fifth_tested_helper_found']
    spec = importlib.util.spec_from_file_location('maintenance', root / 'mxfp4-w4a4/official-performance-v4-gpu6-v2/maintenance_snapshot.py')
    maintenance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maintenance)
    snapshot = maintenance.snapshot()
    assert snapshot['external_count'] == 0, snapshot
    bind = [directory / name for name in ['CPU-preparation.json', 'four-native-coverage.json',
            'TripleOps-original-generated-flags.json', 'old-stage-protected-SHA256.json',
            'root-readonly-stage-protection-and-container-drain.json', 'maintenance-and-stable-owner-drain.json']]
    bind += [root / 'mxfp4-w4a4' / name for name in ['fresh_current_source_four_native_v1.py',
             'fresh_current_source_resource_monitor_v1.py', 'stage_fresh_source_site.py', 'fresh_source_native_v1.py']]
    grant = {'issuer': 'main', 'authorization': 'latest explicit user grants artifact-only current4cb5/e878 fresh four tested native including TripleOps after BFstable4 true0/W4stable6 true5/GPU7v2 true1 all physically drained and external maintenance empty',
        'epoch': time.time(), 'compile_authorized': True, 'GPU_authorized': False,
        'precision_14_and_1300_require_separate_new_main_grant': True,
        'stage': 'fresh-current-source-icpx-native-v3', 'module': 'routing-quant',
        'CPU_affinity': list(range(64, 80)), 'outer_jobs': 2, 'MAX_JOBS': 1,
        'file_compiler_subtree_budget_GiB': 8, 'tree_budget_GiB': 16,
        'source_HEAD': preparation['source_HEAD'], 'dependency_HEAD': preparation['dependency_HEAD'],
        'targets': ['sgl-ops-sycl-' + name for name in ['TopKSigMoid', 'per_token_group_quant_fp4', 'mxfp4_moe_scale_layout', 'TripleOps']],
        'actual_CXX_SYCL_host_compiler': '/opt/gfx-deps/dpcpp/bin/icpx', 'actual_C_compiler': '/opt/gfx-deps/dpcpp/bin/icx',
        'fresh_no_copied_old_objects': True, 'registration_original_preserved': True,
        'approved_artifact_source_delta_only': 'BuildFlags GNU OR IntelLLVM gate/message two lines',
        'last_actual_external_and_compiler_scan': snapshot, 'bound_files_SHA256': {str(path): digest(path) for path in bind}}
    grant_path.write_text(json.dumps(grant, indent=2) + '\n')
    print(json.dumps({'grant_epoch': grant['epoch'], 'grant_SHA256': digest(grant_path), 'stage': grant['stage']}))


if __name__ == '__main__':
    main()
