import hashlib
import json
import re
import statistics
from pathlib import Path

from device_equivalence import elf_sections


FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parent
OLD = ROOT / 'formal-v4-perf-gpu4-v3'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    source_sections = elf_sections((FOLDER / 'SOURCE-device.elf').read_bytes())
    after_sections = elf_sections((FOLDER / 'AFTER-device.elf').read_bytes())
    original = json.loads((FOLDER / 'device-equivalence.json').read_text())
    source_metadata = source_sections['.ze_info'].decode()
    after_metadata = after_sections['.ze_info'].decode()
    body_checks = []
    for dtype in ['int32', 'int64']:
        source_body = (FOLDER / ('SOURCE-' + dtype + '.bin')).read_bytes()
        after_body = (FOLDER / ('AFTER-' + dtype + '.bin')).read_bytes()
        source_asm = (FOLDER / ('SOURCE-' + dtype + '-3pxpc.asm')).read_text()
        after_asm = (FOLDER / ('AFTER-' + dtype + '-3pxpc.asm')).read_text()
        assert all((FOLDER / (label + '-' + dtype + '-3pxpc.exit')).read_text().strip() == '0' for label in ['SOURCE', 'AFTER'])
        instructions = [(int(match.group(1), 16), line) for line in source_asm.splitlines()
                        if (match := re.search(r'\[([0-9A-F]+)\]', line))]
        final_eot_index = max(index for index, (_, line) in enumerate(instructions) if '{EOT,' in line or ',EOT}' in line)
        last_eot_pc = instructions[final_eot_index][0]
        executed_end = instructions[final_eot_index + 1][0]
        branch_targets = [int(target) for line in source_asm.splitlines() if not line.endswith(':')
                          for target in re.findall(r'\bL(\d+)\b', line)]
        assert all(target < executed_end for target in branch_targets)
        dynamic_control = [line for line in source_asm.splitlines()
                           if re.search(r'\b(jmpi|calla|call|ret|brd|brc)\b.*\br\d', line)]
        assert not dynamic_control
        assert source_body[:executed_end] == after_body[:executed_end]
        differing = [index for index, (before, after) in enumerate(zip(source_body, after_body)) if before != after]
        assert len(differing) == 8 and all(index >= executed_end for index in differing)
        suffix = '<int>' if dtype == 'int32' else '<long>'
        source_info = next(value for name, value in original['SOURCE']['bodies'].items() if name.endswith(suffix))
        after_info = next(value for name, value in original['AFTER']['bodies'].items() if name.endswith(suffix))
        source_name = source_info['section'].removeprefix('.text.')
        after_name = after_info['section'].removeprefix('.text.')
        source_metadata = source_metadata.replace(source_name, 'SEMANTIC_' + dtype)
        after_metadata = after_metadata.replace(after_name, 'SEMANTIC_' + dtype)
        source_rel = source_sections['.rel.text.' + source_name]
        after_rel = after_sections['.rel.text.' + after_name]
        assert source_rel == after_rel
        body_checks.append({'dtype': dtype, 'last_EOT_pc': last_eot_pc, 'executed_prefix_bytes': executed_end,
                            'executed_prefix_sha256': digest(source_body[:executed_end]), 'executed_prefix_byte_equal': True,
                            'raw_complete_text_byte_equal': source_body == after_body, 'raw_differing_byte_offsets': differing,
                            'all_direct_branch_targets_within_prefix': True, 'dynamic_control_flow_found': False,
                            'relocations_byte_identical': True,
                            'trailer_differences': 'Two mov null immediate values after final EOT; full raw images retained. No executable body bytes were masked.'})
    assert source_metadata == after_metadata
    assert source_sections['.misc.buildOptions'] == after_sections['.misc.buildOptions']
    rounds = [json.loads((OLD / ('canonical-v4-icpx-formal-gpu4-v3-round' + str(cycle) + '-BAAB-summary.json')).read_text()) for cycle in [1, 2, 3]]
    aggregate = json.loads((OLD / 'aggregate-summary.json').read_text())
    rows = []
    for case_index, pooled in enumerate(aggregate['rows']):
        per_round = []
        for cycle_index, cycle in enumerate(rounds):
            row = cycle['rows'][case_index]
            source_values = row['all_raw_samples_ms'][0] + row['all_raw_samples_ms'][3]
            after_values = row['all_raw_samples_ms'][1] + row['all_raw_samples_ms'][2]
            source_iqr = statistics.quantiles(source_values, n=4, method='inclusive')
            after_iqr = statistics.quantiles(after_values, n=4, method='inclusive')
            medians = row['run_medians_ms']
            per_round.append({'round': cycle_index + 1, 'delta_ns': (row['port_ms'] - row['source_ms']) * 1e6,
                              'delta_pct': row['change_pct'], 'SOURCE_MAD_ns': row['source_MAD_ms'] * 1e6,
                              'AFTER_MAD_ns': row['port_MAD_ms'] * 1e6,
                              'SOURCE_drift_ns': (medians[3] - medians[0]) * 1e6,
                              'AFTER_drift_ns': (medians[2] - medians[1]) * 1e6,
                              'SOURCE_drift_pct': row['source_drift_pct'], 'AFTER_drift_pct': row['port_drift_pct'],
                              'run_medians_ns': [value * 1e6 for value in medians],
                              'run_MAD_ns': [value * 1e6 for value in row['run_MAD_ms']],
                              'IQR_overlap': max(source_iqr[0], after_iqr[0]) <= min(source_iqr[2], after_iqr[2]),
                              'single_round_all_AFTER_above_SOURCE': row['small_stable_positive_regression']})
        rows.append({'case': pooled['case'], 'pooled_delta_ns': (pooled['port_ms'] - pooled['source_ms']) * 1e6,
                     'pooled_delta_pct': pooled['change_pct'], 'SOURCE_MAD_ns': pooled['source_MAD_ms'] * 1e6,
                     'AFTER_MAD_ns': pooled['port_MAD_ms'] * 1e6, 'rounds': per_round})
    record = {'rows': rows, 'native_device_body_checks': body_checks, 'normalized_entire_ze_info_equal': True,
              'native_backend_buildOptions_byte_equal': True,
              'native_backend_buildOptions': source_sections['.misc.buildOptions'].decode(),
              'metadata': {'GRF_count': 96, 'SIMD': 32, 'SLM_bytes': 4096, 'EU_threads': 8},
              'interpretation': 'Round2 +0.205056% is a 6.51ns pooled gap with overlapping IQR and MAD of similar scale. Single-round median ordering is not proof of distinguishable systematic slowdown. Executed ISA/resource/backend flags equivalence excludes a detected math/launch-body change, not host/driver/graph timing variation. All six pooled deltas remain positive; statistical equivalence or zero regression is not established.',
              'historical_perf_hold_exit3_preserved': True, 'GPU_started': False, 'build_started': False,
              'executed_prefix_limit': 'Equality established through final EOT and all direct targets, no dynamic branch found. Complete .text differs in unreachable trailer, never called fully byte-identical.'}
    (FOLDER / 'CPU_NS_AND_NATIVE_BODY_DIAGNOSIS.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'case_count': len(rows), 'executed_bodies_equal': True, 'backend_options_equal': True,
                      'metadata_equal': True, 'GPU_started': False}), flush=True)


if __name__ == '__main__':
    main()
