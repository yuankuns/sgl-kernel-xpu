import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
QK = ROOT.parent
sys.path.insert(0, str(QK / 'controlled-method-runner-cpu-v2'))
from protocol import load_plan, read, require, save

require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'Bounded metadata CPU0–15')
preopt = QK / 'native-icpx-cpu-v3/preopt'
inputs = read(preopt / 'inputs.json')
compiler = read(preopt / 'compiler-policy-verified.json')
native = read(preopt / 'native-complete.json')
deps = read(QK / 'native-icpx-cpu-v3/original-deps.json')
plan, fields, cases = load_plan()
semantic = [case for case in cases if case['group'] == 'actual-config-semantic-synthetic']
require(len(semantic) == 17, 'Full17 semantic fixtures, no selection')
library = preopt / 'site/sgl_kernel/libsgl-ops-sycl-FusedQKNormRope.so'
require(library.is_file() and (preopt / 'site/sgl_kernel/__init__.py').is_file(), 'Existing normal truepreopt package path')
known = read(QK / 'source-accuracy-v1/fresh-preopt/accuracy-t32768.json')
require(known['violations'] == 3508 and (QK / 'source-accuracy-v1/fresh-preopt/accuracy-t32768.exit').read_text().strip() == '1', 'Known strict3508/exit1 preserved')
save(ROOT / 'CPU_PLAN.json', {'actual_CPU_plan_exit': 0, 'CPU_only_metadata_no_package_reaudit': True,
    'preopt': {'ref': inputs['ref'], 'recorded_native_SHA': native['fresh_native_sha256'],
        'recorded_full_site_manifest_SHA': native['site_manifest_sha256'], 'normal_site': str(preopt / 'site'),
        'native_file_exists': True, 'native_file_bytes_stat_only': library.stat().st_size,
        'original_tracked_archive_SHA': inputs['archive_sha256'], 'only_source_snapshot_changed_path': inputs['only_changed_path'],
        'tracked_math_source_SHA': {name: checksum for name, checksum in inputs['all_tracked_original_files'].items()
            if 'FusedQKNormRope' in name or name.endswith(('SYCLHelpers.h', 'MemoryAccess.h', 'Utils.h'))},
        'dependency_pin': native['original_deps_pin'], 'historical_actual_cutlass_HEAD_unverified': deps['historical_preopt_cutlass_mount_HEAD_unverified'],
        'compiler': compiler, 'native_build_original_command': (preopt / 'qk-target-build.command').read_text(),
        'native_build_recorded_exit': int((preopt / 'qk-target-build.exit').read_text()),
        'normal_import_recorded_exit': int((preopt / 'normal-package-import.exit').read_text()),
        'scope': native['scope']},
    'all17_cases': semantic, 'fixture_plan_SHA': fields['fixture_plan_SHA'],
    'known_old_aggregate_failure': {'tokens': 32768, 'violations': 3508, 'phase_exit': 1,
        'path': str(QK / 'source-accuracy-v1/fresh-preopt/accuracy-t32768.json'), 'NOT_actual17_validation': True},
    'minimum_future_write_scope': ['new baseline artifact only/contract.py', 'run_accuracy.py', 'audit_cpu.py', 'launch.sh',
        'new external grant/READY/actual17 outputs/logs/exits'],
    'normal_actual_framework_adapter_generic_original_ABI_required': True,
    'atol': 0.005, 'rtol': 0.005, 'full_original_CPU_reference_and_QKVO_tail_bits_required': True,
    'preopt_golden_exactbits_difference_is_diagnostic_NOT_required_optimized_equivalence': True,
    'all17_precision_STILL_pending': True, 'no_GPU_or_compile_authorization': True,
    'future_GPU10_accuracy_requires_separate_main_grant': True, 'GPU10_performance_NOT_proposed': True,
    'optimized_SOURCE_to_PORT_NOT_true_preopt_to_optimized': True})
print('Metadata-only truepreopt existing provider/all17 CPU plan actual0; original3508 unchanged; no GPU/build or package reaudit')
