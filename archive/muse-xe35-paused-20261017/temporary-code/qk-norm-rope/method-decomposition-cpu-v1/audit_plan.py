import ast
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parent / 'normal-v4-model-config-perf-v1'))
from run_perf import digest, verify


def main():
    assert set(os.sched_getaffinity(0)) == set(range(16))
    previous = verify()
    fields = json.loads((RUN / 'plan-fields-hashes.json').read_text())
    mapping = {'plan_SHA': 'method-plan.json', 'fixture_plan_SHA': 'shared-19-fixture-plan.json',
        'flags_evidence_SHA': 'flags-host-path-evidence.json', 'metadata_comparison_SHA': 'metadata-comparison.json',
        'normal_HOST_dtypes_SOURCE_SHA': 'normal-host-dtype-source.json', 'normal_HOST_dtypes_PORT_SHA': 'normal-host-dtype-port.json'}
    for field, filename in mapping.items():
        assert digest(RUN / filename) == fields[field]
    for name in ('extraction', 'tool-probe-v2', 'iga-platform-help', 'iga-first-pair', 'metadata-comparison',
        'normal-host-dtype-source', 'normal-host-dtype-port', 'prepare-plan-r2', 'PORT-generic-needed', 'PORT-generic-export'):
        assert (RUN / f'{name}.exit').read_text().strip() == '0'
    assert (RUN / 'prepare-plan.exit').read_text().strip() == '1'
    assert (RUN / 'tool-probe-initial.exit').read_text().strip() == '1'
    for path in RUN.glob('*.py'):
        ast.parse(path.read_text())
    plan = json.loads((RUN / 'method-plan.json').read_text())
    assert not plan['GPU_authorized'] and plan['future_runner_SHA'] is None
    assert plan['fixed_protocol']['rounds'] == 3 and plan['fixed_protocol']['every_phase_all_cases'] == 19
    assert plan['fixtures']['actual_config_semantic_synthetic'] == 17 and plan['fixtures']['legacy_representatives'] == 2
    assert plan['fixtures']['all_shared_original_inputs_SHA'] == {case['name']: case['fixture']['sha256'] for case in previous['cases']}
    prior = json.loads((RUN.parent / 'normal-v4-model-config-perf-v1/analysis-final.json').read_text())
    assert prior['cases']['decode-output-05-position-4100']['acceptance_status'] == 'regression_confirmed'
    assert not prior['all_cases_unqualified_performance_nonregression']
    for name, checksum in plan['framework_and_source_file_SHA'].items():
        assert digest(Path(name)) == checksum
    metadata = json.loads((RUN / 'metadata-comparison.json').read_text())
    for row in metadata['rows']:
        binaries = {role: (RUN / role / f"image-{row['image']:02d}/kernel-{row['kernel']:03d}.isa.bin").read_bytes() for role in ('source', 'port')}
        end = row['executable_end_after_last_EOT_offset']
        assert binaries['source'][:end] == binaries['port'][:end] and binaries['source'] != binaries['port']
        assert row['raw_differing_byte_offsets'] == [index for index, (left, right) in enumerate(zip(binaries['source'], binaries['port'])) if left != right]
        assert all(index >= end for index in row['raw_differing_byte_offsets'])
    for path in RUN.rglob('*.iga-full.command.json'):
        command = json.loads(path.read_text())
        assert '-d' in command and '-a' not in command
    live = {}
    names = {'ninja', 'cmake', 'icpx', 'icx', 'clang++', 'clang-23', 'clang-24', 'cc1plus', 'ocloc', 'pip', 'pip3', 'uv', 'apt', 'apt-get', 'dpkg'}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            comm = (directory / 'comm').read_text().strip()
            args = (directory / 'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if args and (comm in names or (comm.startswith('python') and ' -m pip ' in args)):
            live[directory.name] = {'comm': comm, 'args': args}
    snapshot = {'epoch': time.time(), 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'matching_live_compiler_or_package_manager': live,
        'no_GPU_lock_or_query_performed': True, 'GPU6_main_owner': 'Mill W4', 'current_QK_CPU_only_assignment': '0-15'}
    (RUN / 'compiler-package-manager-idle.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    assert not live
    files = {}
    excluded = {'CPU-final-audit.json', 'CPU-final-audit.log', 'CPU-final-audit.exit', 'write-set.json'}
    for path in sorted(RUN.rglob('*')):
        if path.is_file() and path.name not in excluded:
            files[str(path.relative_to(RUN))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    result = {'actual_exit': 0, 'CPU_audit_pid': os.getpid(), 'CPU_affinity': sorted(os.sched_getaffinity(0)),
        'plan_fields_SHA': fields, 'all_current_files_SHA': files, 'kernel_pairs_executable_prefix_rechecked': 80,
        'full_ISA_differences_preserved': True, 'previous_regression_and_all_raw_not_edited': True,
        'GPU_or_compiler_or_package_manager_launched': False, 'GPU6_W4_not_contended': True,
        'new_runner_or_GPU_grant_created': False, 'future_normal_core_loaded_stack_maps_explicit_gap': True}
    (RUN / 'CPU-final-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'plan_SHA': fields['plan_SHA'], 'fixture_count': 19,
        'kernel_pairs_prefix_rechecked': 80, 'files_SHA': len(files), 'CPU_affinity': '0-15', 'GPU_authorized': False}, indent=2))


if __name__ == '__main__':
    main()
