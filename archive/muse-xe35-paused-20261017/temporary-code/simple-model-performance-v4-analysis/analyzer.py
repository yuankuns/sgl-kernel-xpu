import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN_NAME = 'simple-model-verified-performance-v4-gpu7-run1'
ORDER = ('before', 'after', 'after', 'before')
COMPONENTS = ('memory', 'position', 'logits-vocab')
FAMILIES = ('semantic', 'representative')
HASHES = {
    'benchmark_memory_position_model_performance_v3.py': '2f44178e14c1f6530f154b813a1b6eb2c7c98718151f47ab40baf2cc620d54fa',
    'benchmark_logits_model_performance_v4.py': 'a1499ee298ec4759c65e9f99e7c0b65e02528452b995c49f38444ce348fffd1e',
    'benchmark_simple_canonical_verified_perf_v3.py': '6fab26379c3c3fc689a54bf5af3c25163ceee4d293e8663e75e3fa69d56bde8e',
    'simple_verified_graph_v2.py': 'cbb357ca8d8db8f1ae606c1ae0d1770c3b5f86b3262ba67ffdc89ae22488e48c',
    'simple_representative_performance_cases_v2.py': '7548d8563411ff5ad0a22ed1ca59a247fa4c200c3e6a261e11e5fbfa5353cfca',
    'benchmark_simple.py': 'ee8b80880b07fb679db74a1d18c250be412fc9c5faed0a5ca3e6956752befff3',
    'canonical_tools_loader_v1.py': '75afc9d8193a32a03a77f8b06ddf7db125058ff3c6a359e3d19ea9cf57a00ac0',
    'canonical-tools/run_simple_verified_performance_v4.py': 'e995941cbdfa24abbc0024e0c2f4e63be3561755402b74b9f5398725c1dd604c',
}
SCOPE = {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1}
SCOPES = {
    ('memory', 'semantic'): 'memory-position-model-performance-v3',
    ('position', 'semantic'): 'memory-position-model-performance-v3',
    ('logits-vocab', 'semantic'): 'logits-model-performance-v4',
}


class Rejected(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Rejected(message)


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def bad_constant(value):
    raise Rejected('nonfinite JSON constant: ' + value)


def decode(text):
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=bad_constant)


def exact(record, expected, context):
    for key, value in expected.items():
        require(type(record.get(key)) is type(value) and record[key] == value, context + ': ' + key)


def positive(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, name + ': finite positive number required')
    return value


def close(actual, expected, name):
    require(type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), name + ': inconsistent statistic or units')


def names(component, family):
    if family == 'representative':
        return {
            'memory': ['alloc-extend-prefill-b8', 'write-prefill-b8', 'alloc-write-prefill-b8', 'alloc-decode-osl1-b8', 'alloc-decode-osl16-b8'],
            'position': ['position-t8-b8', 'position-hint-t8-b8', 'position-t32768-b8', 'position-hint-t32768-b8'],
            'logits-vocab': ['softcap-fp32-b8-v202048', 'embedding-bf16-t8-b8', 'vocab-mask-t8-b8', 'embedding-bf16-t32768-b8', 'vocab-mask-t32768-b8'],
        }[component]
    if component == 'memory':
        result = [f'{operation}-wave{active}-tokens{active * 4096}' for active in (3, 5) for operation in ('extend', 'write', 'fused')]
        return result + [f'decode-output{step}-cache{4096 + step}' for step in range(1, 16)]
    if component == 'position':
        bases = ['prefill-wave3', 'prefill-wave5'] + [f'decode-output{step}' for step in range(1, 16)]
        return [name + suffix for name in bases for suffix in ('', '-hint')]
    return [kind + '-' + label for label in ('decode8', 'prefill-wave12288', 'prefill-wave20480') for kind in ('embedding', 'mask')]


def driver(component, family):
    if family == 'representative':
        return 'benchmark_simple_canonical_verified_perf_v3.py'
    return 'benchmark_logits_model_performance_v4.py' if component == 'logits-vocab' else 'benchmark_memory_position_model_performance_v3.py'


def local_path(value):
    text = str(value)
    if str(ROOT).startswith('/root/') and text.startswith('/home/syk/'):
        text = text.replace('/home/syk/', '/root/', 1)
    if str(ROOT).startswith('/home/syk/') and text.startswith('/root/'):
        text = text.replace('/root/', '/home/syk/', 1)
    return Path(text).resolve()


class Evidence:
    def __init__(self):
        self.files = {}

    def file(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_file() and not path.is_symlink(), 'missing evidence: ' + str(path))
        before = path.stat()
        sha = digest(path)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'live changing evidence: ' + str(path))
        require(expected is None or sha == expected, 'SHA changed: ' + str(path))
        require(str(path) not in self.files or self.files[str(path)]['sha256'] == sha, 'evidence changed between reads')
        self.files[str(path)] = {'sha256': sha, 'bytes': after.st_size}
        return path

    def json(self, path, expected=None):
        return decode(self.file(path, expected).read_text())

    def exit0(self, path):
        require(self.file(path).read_text().strip() == '0', 'missing or failed exit: ' + str(path))

    def stable(self):
        for path, record in self.files.items():
            require(digest(path) == record['sha256'], 'evidence changed at final seal: ' + path)


def parse_native_dot(text):
    require(re.match(r'\s*digraph\s+"graph"\s*\{', text) is not None and text.rstrip().endswith('}'), 'native directed DOT envelope missing')
    declarations = list(re.finditer(r'^\s*(L[0-9]+_S[0-9]+_C[0-9]+)\s*\[label\s*=', text, re.MULTILINE))
    identities = [match.group(1) for match in declarations]
    require(identities and len(set(identities)) == len(identities), 'zero or duplicate native DOT nodes')
    for index, match in enumerate(declarations):
        stop = declarations[index + 1].start() if index + 1 < len(declarations) else len(text)
        body = text[match.end():stop]
        require('zeCommandListAppendLaunchKernel' in body and 'kernelName' in body, 'non-kernel or fabricated native node declaration')
        require('</TABLE>>' in body or '</TABLE>\n  >' in body or re.search(r'</TABLE>\s*>\s*\];', body), 'incomplete native node label')
    declared = set(identities)
    for source, target in re.findall(r'(L[0-9]+_S[0-9]+_C[0-9]+)\s*->\s*(L[0-9]+_S[0-9]+_C[0-9]+)', text):
        require(source in declared and target in declared, 'undeclared native DOT edge endpoint')
    return sorted(identities)


def validate_sample_arrays(result, rows):
    exact(result, {'calls': 128, 'samples': 101, 'warmup': 5, 'accuracy': 'PASS'}, 'case protocol')
    require(len(rows) == 101 and len(result['raw_us']) == 101 and len(result['raw_graph_ms']) == 101, 'sample count must be101; no drops')
    for index, row in enumerate(rows):
        require(set(row) == {'sample_index', 'graph_ms', 'per_logical_call_us', 'calls'}, 'unknown sample schema')
        exact(row, {'sample_index': index, 'calls': 128}, 'sample index/calls')
        elapsed = positive(row['graph_ms'], 'graph_ms')
        per_call = positive(row['per_logical_call_us'], 'per_logical_call_us')
        positive(result['raw_graph_ms'][index], 'raw_graph_ms')
        positive(result['raw_us'][index], 'raw_us')
        require(per_call == elapsed * 1000 / 128, 'graph_ms128->us conversion mismatch')
        require(result['raw_graph_ms'][index] == elapsed and result['raw_us'][index] == per_call, 'raw array/JSONL mismatch')
    median = statistics.median(result['raw_us'])
    close(result['p50_us'], median, 'p50_us')
    close(result['mad_us'], statistics.median(abs(value - median) for value in result['raw_us']), 'mad_us')


def audit_case(evidence, directory, index, name):
    result = evidence.json(directory / 'result.json')
    exact(result, {'case_index': index, 'case': name}, 'case identity')
    proof_path = directory / 'graph-proof.json'
    require(local_path(result['graph_proof_path']) == proof_path.resolve(), 'graph proof path escaped case')
    proof = evidence.json(proof_path, result['graph_proof_sha256'])
    require(proof == result['graph_proof'], 'embedded proof differs from saved proof')
    exact(proof, {'case_index': index, 'case': name, 'calls': 128, 'samples': 101, 'warmup': 5, 'timing_events_created': False, 'reset_and_sync_outside_events': True}, 'graph protocol')
    counts = []
    for role, count, stem in (('single', 1, 'single-call'), ('repeated', 128, 'calls-128')):
        record = proof[role]
        exact(record, {'captured_calls': count, 'logical_calls': count, 'keep_graph': True, 'timing_events_created': False}, role + ' capture')
        require(evidence.json(directory / (stem + '-proof.json')) == record, 'separate capture proof mismatch')
        dot_path = directory / (stem + '.dot')
        require(local_path(record['DOT_path']) == dot_path.resolve(), 'DOT path escaped or wrong suffix')
        require(record['DOT_files'] == {dot_path.name: record['DOT_sha256']}, 'exact native DOT filename/hash required')
        nodes = parse_native_dot(evidence.file(dot_path, record['DOT_sha256']).read_text())
        require(record['node_ids'] == nodes and type(record['DOT_nodes']) is int and record['DOT_nodes'] == len(nodes), 'native node list/count mismatch')
        counts.append(len(nodes))
    require(counts[0] > 0 and counts[1] == 128 * counts[0], 'actual native nodes must be128*single>0')
    samples_path = directory / 'samples.jsonl'
    require(local_path(result['samples_path']) == samples_path.resolve(), 'samples path escaped case')
    rows = [decode(line) for line in evidence.file(samples_path, result['samples_sha256']).read_text().splitlines()]
    validate_sample_arrays(result, rows)
    return result


def validate_phase_watch(record):
    exact(record, {'external_processes_modified': False}, 'maintenance monitor')
    require(type(record.get('contamination')) is list and not record['contamination'], 'maintenance/loaded-stack contamination: hold')


def validate_manifest_case_set(results, expected):
    require(type(results) is list and len(results) == len(expected), 'incomplete/extra case set')
    require([record['case'] for record in results] == expected, 'missing/duplicate/out-of-order case')
    require(len({record['case'] for record in results}) == len(expected), 'duplicate cases')


def context(evidence, run):
    require(run.resolve() == (ROOT / RUN_NAME).resolve(), 'only reviewed v4 run is in scope')
    plan = evidence.json(run / 'launch-plan.json')
    exact(plan, {'rounds': 3, 'order': list(ORDER), 'semantic_cases': 61, 'representative_cases': 14, 'samples': 101, 'calls': 128}, 'launch plan')
    expected_drivers = {key: value for key, value in HASHES.items() if not key.startswith('canonical-tools/')}
    require(plan['drivers_sha256'] == expected_drivers, 'unreviewed driver/helper SHA in plan')
    for name, sha in HASHES.items():
        evidence.file(ROOT / name, sha)
    for key, filename in (
        ('quiet_evidence_sha256', 'MAINTENANCE_QUIET_AFTER_ROUTING_v4.json'),
        ('routing_source_audit_sha256', 'ROUTING_CURRENT_SOURCE_MAIN_PACKAGE_AUDIT_v1.json'),
        ('integrated_CPU_preflight_sha256', 'SIMPLE_INTEGRATED_PERFORMANCE_CPU_PREFLIGHT_v4.json'),
    ):
        record = evidence.json(ROOT / filename, plan[key])
        require(record.get('actual_exit') == 0, 'failed prerequisite: ' + filename)
        if key == 'quiet_evidence_sha256':
            require(record['continuous_quiet_seconds'] >= 60 and record['loaded_stack'] == plan['loaded_stack'], 'initial quiet/stack mismatch')
    evidence.exit0(ROOT / 'simple-model-performance-v4-cpu-preparation/full-normal-CPU-gates.exit')
    grants = {}
    for component in COMPONENTS:
        official = ROOT / component / 'canonical-v4-icpx'
        comparator_path = ROOT / component / 'canonical-validation-v4-icpx/source-comparator-manifest.json'
        comparator = evidence.json(comparator_path)
        manifest_sha = digest(evidence.file(official / 'source-binary-manifest.json'))
        evidence.file(local_path(comparator['source_native_manifest']), comparator['source_native_manifest_sha256'])
        for family in FAMILIES:
            grant = evidence.json(run / f'{component}-{family}-grant.json')
            exact(grant, dict(SCOPE, issuer='main-coordinator', scope=SCOPES.get((component, family), 'simple-representative-verified-performance-v3'), module=component, phase='performance', gpu=7, coordinator_release=True, accuracy_authorized=False, performance_authorized=True, start_epoch=plan['start_epoch'], CPU_affinity=list(range(112, 128))), 'grant')
            exact(grant, {'driver_sha256': HASHES[driver(component, family)], 'graph_helper_sha256': HASHES['simple_verified_graph_v2.py'], 'case_helper_sha256': HASHES['simple_representative_performance_cases_v2.py'], 'package_manifest_sha256': manifest_sha, 'source_comparator_manifest_sha256': digest(comparator_path), 'source_native_manifest_sha256': comparator['source_native_manifest_sha256']}, 'grant source/native/SHA')
            grants[component, family] = grant
        for label in ('before', 'after'):
            gate = evidence.json(ROOT / 'simple-model-performance-v4-cpu-preparation' / f'{component}-{label}.json')
            exact(gate, {'actual_exit': 0, 'GPU_initialized': False, 'component': component, 'label': label, 'package_manifest_sha256': manifest_sha, 'source_comparator_manifest_sha256': digest(comparator_path), 'source_native_manifest_sha256': comparator['source_native_manifest_sha256']}, 'normal CPU provider gate')
    return plan, grants


def stage_files(evidence, component, label):
    if label == 'before':
        comparator = evidence.json(ROOT / component / 'canonical-validation-v4-icpx/source-comparator-manifest.json')
        return local_path(comparator['stage']), comparator['files']
    manifest = evidence.json(ROOT / component / 'canonical-v4-icpx/source-binary-manifest.json')
    return local_path(manifest['site']), manifest['installed_files'] | manifest['python_snapshot_files']


def provider(evidence, report, component, family, label):
    stage, files = stage_files(evidence, component, label)
    tested = {'memory': ('PagedAlloc', 'WriteReqToTokenPool'), 'position': ('ComputePosition',), 'logits-vocab': ('SoftcapLogits', 'VocabMask')}[component]
    if family == 'semantic':
        mapped = report['mapped_native_sha256']
        require(bool(mapped) and any('common_ops' in name for name in mapped), 'normal registration mapping missing')
        for relative, sha in mapped.items():
            require(relative in files and files[relative]['sha256'] == sha, 'mapped native/provider manifest mismatch')
            require(local_path(stage / relative).is_relative_to(stage), 'mapped path escaped stage')
            evidence.file(stage / relative, sha)
        required = ('VocabMask',) if component == 'logits-vocab' else tested
        for name in required:
            require('sgl_kernel/libsgl-ops-sycl-' + name + '.so' in mapped, 'tested native not mapped')
    else:
        require(local_path(report['package']) == stage / 'sgl_kernel/__init__.py', 'wrong normal package provider')
        mapped = report['libraries']
        require(len(mapped) == len(tested), 'representative tested native mappings missing')
        for filename, sha in mapped.items():
            path = local_path(filename)
            require(path.is_relative_to(stage), 'representative mixed-stage provider')
            relative = path.relative_to(stage).as_posix()
            require(relative in files and files[relative]['sha256'] == sha, 'representative native SHA mismatch')
            evidence.file(path, sha)
        require({local_path(filename).name for filename in mapped} == {'libsgl-ops-sycl-' + name + '.so' for name in tested}, 'wrong tested natives')


def tensor_identity(torch, first, second, name, counters):
    if isinstance(first, torch.Tensor):
        require(isinstance(second, torch.Tensor) and first.device.type == second.device.type == 'cpu', 'CPU tensor required: ' + name)
        require(first.dtype == second.dtype and first.shape == second.shape and first.layout == second.layout == torch.strided, 'tensor dtype/shape/layout differs: ' + name)
        first_bytes = first.contiguous().reshape(-1).view(torch.uint8)
        second_bytes = second.contiguous().reshape(-1).view(torch.uint8)
        require(torch.equal(first_bytes, second_bytes), 'whole tensor bits differ: ' + name)
        counters['tensors'] += 1
        counters['bytes'] += first_bytes.numel()
    elif isinstance(first, dict):
        require(type(second) is dict and first.keys() == second.keys(), 'tensor tree keys differ: ' + name)
        for key in first:
            tensor_identity(torch, first[key], second[key], name + '/' + str(key), counters)
    else:
        require(type(first) in (int, str, bool, float, type(None)) and type(second) is type(first) and first == second, 'unsupported or different tensor tree leaf: ' + name)


def statistics_case(phases):
    require(len(phases) == 12, 'three complete BAAB rounds required')
    for result in phases:
        positive(statistics.median(result['raw_us']), 'phase median')
    rounds = []
    role_values = {'before': [], 'after': []}
    for round_index in range(3):
        group = phases[round_index * 4:round_index * 4 + 4]
        row = {'round': round_index, 'phase_medians_us': [statistics.median(record['raw_us']) for record in group]}
        for role, slots in (('before', (0, 3)), ('after', (1, 2))):
            values = [sample for slot in slots for sample in group[slot]['raw_us']]
            role_values[role].extend(values)
            row[role + '_median_us'] = statistics.median(values)
            pair = [row['phase_medians_us'][slot] for slot in slots]
            row[role + '_drift_pct'] = abs(pair[1] - pair[0]) / min(pair) * 100
        row['after_vs_before_pct'] = (row['after_median_us'] / row['before_median_us'] - 1) * 100
        rounds.append(row)
    before = statistics.median(role_values['before'])
    after = statistics.median(role_values['after'])
    delta = (after / before - 1) * 100
    drift = max(row[role + '_drift_pct'] for row in rounds for role in ('before', 'after'))
    role_round_drifts = {}
    for role in ('before', 'after'):
        medians = [row[role + '_median_us'] for row in rounds]
        role_round_drifts[role] = (max(medians) / min(medians) - 1) * 100
    mad_before = statistics.median(abs(value - before) for value in role_values['before'])
    mad_after = statistics.median(abs(value - after) for value in role_values['after'])
    larger_than_noise = after - before > max(mad_before, mad_after)
    stable_regression = all(row['after_vs_before_pct'] > 0 for row in rounds) and delta > 0 and larger_than_noise
    over5 = delta > 5 or any(row['after_vs_before_pct'] > 5 for row in rounds)
    drift_hold = drift > 5 or max(role_round_drifts.values()) > 5
    return {'rounds': rounds, 'role_pooled_samples': {role: len(values) for role, values in role_values.items()}, 'before_pooled_median_us': before, 'after_pooled_median_us': after, 'before_pooled_MAD_us': mad_before, 'after_pooled_MAD_us': mad_after, 'after_vs_before_pct': delta, 'within_round_max_drift_pct': drift, 'between_round_role_drift_pct': role_round_drifts, 'stable_regression': stable_regression and not drift_hold, 'regression_over5pct': over5, 'drift_hold': drift_hold, 'hold': stable_regression or over5 or drift_hold, 'outliers_dropped': 0, 'automatic_acceptance': False, 'interpretation': 'original SOURCE versus canonical PORT; not preoptimization/optimization benefit'}


def terminal_requirements(evidence, run, controller_exit):
    complete = evidence.json(run / 'complete.json')
    exact(complete, {'actual_exit': 0, 'phases': 12, 'precision_performance_raw_completed': True, 'independent_comparison_analysis_pending': True, 'acceptance_complete': False}, 'controller complete')
    require(controller_exit is not None, 'explicit real controller exit file required; no inferred success')
    evidence.exit0(controller_exit)
    expected = {f'round-{round_index}-slot-{slot}-{label}' for round_index in range(3) for slot, label in enumerate(ORDER)}
    require({path.name for path in run.glob('round-*') if path.is_dir()} == expected, 'missing/extra phase directories')
    for name in sorted(expected):
        phase = run / name
        evidence.exit0(phase / 'wrapper.exit')
        validate_phase_watch(evidence.json(phase / 'maintenance-watch.json'))
        for component in COMPONENTS:
            for family in FAMILIES:
                evidence.exit0(phase / f'{component}-{family}.exit')


def analyze(evidence, run, controller_exit, torch):
    terminal_requirements(evidence, run, controller_exit)
    plan, grants = context(evidence, run)
    records = {}
    tensor_baselines = {}
    tensor_checks = []
    metadata_baselines = {}
    for round_index in range(3):
        for slot, label in enumerate(ORDER):
            phase = run / f'round-{round_index}-slot-{slot}-{label}'
            for component in COMPONENTS:
                for family in FAMILIES:
                    prefix = phase / f'{component}-{family}'
                    report = evidence.json(prefix.with_suffix('.json'))
                    exact(report, dict(SCOPE, label=label, phase='performance', samples=101, calls=128, driver_sha256=HASHES[driver(component, family)]), 'phase report')
                    gpu_key = 'gpu_affinity' if family == 'representative' else 'gpu'
                    exact(report, {gpu_key: '7'}, 'phase GPU')
                    if component != 'logits-vocab' or family == 'representative':
                        exact(report, {'component': component}, 'component')
                    if family == 'semantic':
                        exact(report, {'model_config_sha256': grants[component, family]['model_config_sha256']}, 'model config identity')
                    else:
                        exact(report, {'reference_driver_sha256': HASHES['benchmark_simple.py']}, 'representative factory')
                    provider(evidence, report, component, family, label)
                    expected = names(component, family)
                    validate_manifest_case_set(report['results'], expected)
                    for index, name in enumerate(expected):
                        result = audit_case(evidence, prefix.with_suffix('.graphs') / f'case-{index:04d}-{name}', index, name)
                        require(all(report['results'][index].get(key) == value for key, value in result.items()), 'case result differs from final phase report')
                        records.setdefault((component, family, name), []).append(result)
                    if family != 'semantic':
                        continue
                    metadata = {key: report[key] for key in ('weight_sha256',) if key in report}
                    baseline_metadata = metadata_baselines.setdefault(component, metadata)
                    require(metadata == baseline_metadata, 'semantic shared input-weight SHA changed')
                    tensor_path = prefix.with_suffix('.pt')
                    evidence.file(tensor_path)
                    if 'raw_output_file' in report:
                        require(local_path(report['raw_output_file']) == tensor_path.resolve(), 'semantic tensor output path mismatch')
                    data = torch.load(tensor_path, map_location='cpu', weights_only=True)
                    require(type(data) is dict and set(data) == set(expected), 'semantic PT case coverage mismatch')
                    if component in ('memory', 'position'):
                        require(all(set(tree) == {'inputs', 'outputs'} for tree in data.values()), 'semantic saved input/output scope missing')
                    else:
                        for name, tree in data.items():
                            require(set(tree) == ({'ids', 'output'} if name.startswith('embedding') else {'ids', 'mapped', 'invalid'}), 'logits semantic PT scope mismatch')
                    baseline = tensor_baselines.setdefault(component, data)
                    counters = {'tensors': 0, 'bytes': 0}
                    tensor_identity(torch, baseline, data, f'{phase.name}/{component}', counters)
                    tensor_checks.append({'phase': phase.name, 'component': component, 'cases': len(expected), 'whole_input_output_bits': True, **counters, 'PT_sha256': digest(tensor_path)})
                    if component == 'logits-vocab':
                        for result in report['results']:
                            if result['case'].startswith('embedding'):
                                ids = data[result['case']]['ids']
                                actual_ids_sha = hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest()
                                require(result['input_ids_sha256'] == actual_ids_sha, 'embedding actual input identity mismatch')
    require(len(records) == 75 and all(len(group) == 12 for group in records.values()), 'whole75case/12phase coverage required')
    stats = [{'component': component, 'family': family, 'case': name, **statistics_case(group)} for (component, family, name), group in records.items()]
    hold = any(row['hold'] for row in stats)
    return {'mode': 'full_CPU_analysis', 'actual_exit': 3 if hold else 0, 'cases': 75, 'phases': 12, 'raw_samples': 75 * 12 * 101, 'cases_statistics': stats, 'semantic_whole_input_output_bits': tensor_checks, 'representative_PT_saved': False, 'representative_wholebits_claimed': False, 'representative_accuracy_scope': 'pinned runtime full-output checks only; softcap128independentbuffers reset outside events, cap20/3e-5; no saved representative output PT', 'embedding_weight_identity_scope': 'driver full input-weight SHA only; full weights not saved in semantic PT', 'maintenance_scope': 'initial60second quiet observation plus perphase2second monitor; not continuous nanosecond proof', 'performance_accepted': False, 'full_model_or_allocator_replay_claimed': False, 'hold': hold, 'plan_start_epoch': plan['start_epoch']}


def first_source(evidence, run):
    plan, _ = context(evidence, run)
    phase = run / 'round-0-slot-0-before'
    audited = []
    for index, name in enumerate(names('memory', 'semantic')[:6]):
        directory = phase / 'memory-semantic.graphs' / f'case-{index:04d}-{name}'
        result = audit_case(evidence, directory, index, name)
        audited.append({'case_index': index, 'case': name, 'samples': 101, 'single_nodes': result['graph_proof']['single']['DOT_nodes'], 'repeated_nodes': result['graph_proof']['repeated']['DOT_nodes'], 'samples_sha256': result['samples_sha256'], 'graph_proof_sha256': result['graph_proof_sha256'], 'p50_us': result['p50_us']})
    return {'mode': 'first_SOURCE_readonly_partial_audit', 'actual_exit': 0, 'SOURCE_phase': phase.name, 'completed_cases': audited, 'full75case_analysis': False, 'phase_terminal_asserted': False, 'maintenance_terminal_asserted': False, 'semantic_PT_wholebits_compared': False, 'performance_accepted': False, 'runtime_checks_scope': 'recorded accuracy PASS plus pinned graph/check protocol; no new GPU or reference execution', 'plan_start_epoch': plan['start_epoch']}


def cpu_environment():
    require(set(os.sched_getaffinity(0)) == set(range(80, 96)), 'CPU80-95 required')
    require(os.environ.get('ONEAPI_DEVICE_SELECTOR') == 'level_zero:cpu', 'explicit CPU-only selector required')
    import torch
    require(not torch.xpu.is_initialized(), 'Torch XPU already initialized')
    torch.set_num_threads(8)
    return torch


def save_new(path, record):
    path = Path(path)
    require(path.resolve().parent == Path(__file__).resolve().parent, 'output only in new analysis directory')
    with path.open('x') as output:
        output.write(json.dumps(record, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('ready', 'audit-first-source', 'analyze'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controller-exit', type=Path)
    args = parser.parse_args()
    evidence = Evidence()
    status = 0
    try:
        torch = cpu_environment()
        run = ROOT / RUN_NAME
        if args.mode == 'ready':
            context(evidence, run)
            report = {'mode': 'CPU_READY', 'actual_exit': 0, 'expected_cases': {component + '/' + family: names(component, family) for component in COMPONENTS for family in FAMILIES}, 'semantic_cases': 61, 'representative_cases': 14, 'terminal_full_analysis_not_started': True, 'performance_accepted': False}
        elif args.mode == 'audit-first-source':
            report = first_source(evidence, run)
        else:
            report = analyze(evidence, run, args.controller_exit, torch)
        require(not torch.xpu.is_initialized(), 'CPU analysis unexpectedly initialized XPU')
        evidence.stable()
        status = report['actual_exit']
        report.update(Torch_provider=torch.__file__, Torch_version=torch.__version__, XPU_initialized=False, CPU_affinity=sorted(os.sched_getaffinity(0)), GPU_started=False, compiler_started=False, evidence_files=evidence.files, analyzer_SHA256=digest(__file__))
    except (Rejected, OSError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as error:
        status = 2
        report = {'mode': args.mode, 'actual_exit': status, 'rejected': str(error), 'performance_accepted': False, 'raw_modified': False, 'evidence_files': evidence.files, 'analyzer_SHA256': digest(__file__)}
    save_new(args.output, report)
    print(json.dumps({key: report.get(key) for key in ('mode', 'actual_exit', 'rejected', 'performance_accepted', 'XPU_initialized')}), flush=True)
    raise SystemExit(status)


if __name__ == '__main__':
    main()
