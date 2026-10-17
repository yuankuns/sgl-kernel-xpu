import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from prepare_cpu import DIRECTORY, FROZEN, policy


def cache_digest(directory, name, dtype, shape):
    import torch
    paths = sorted(directory.glob(name + '-*.pt'), key=lambda path: int(path.stem.rsplit('-', 1)[1]))
    block = 1024 if name == 'output' else 16
    assert len(paths) == math.ceil(shape[0] / block)
    checksum = hashlib.sha256()
    file_seals = {}
    rows = 0
    for index, path in enumerate(paths):
        assert path.name == f'{name}-{index}.pt'
        tensor = torch.load(path, weights_only=True, mmap=True)
        assert tensor.dtype == dtype and tuple(tensor.shape[1:]) == tuple(shape[1:])
        assert tensor.shape[0] == min(block, shape[0] - rows)
        assert tensor.is_contiguous()
        checksum.update(tensor.view(torch.uint8).numpy().tobytes())
        rows += tensor.shape[0]
        file_seals[str(path.relative_to(FROZEN))] = policy.digest(path)
    assert rows == shape[0]
    return checksum.hexdigest(), file_seals


def main():
    policy.preflight()
    assert os.environ['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:cpu'
    assert set(os.sched_getaffinity(0)) == set(range(80, 96))
    import torch
    assert not torch.xpu.is_initialized()
    grant = DIRECTORY / 'main-grant.json'
    grant_sha = policy.digest(grant)
    policy.verify_grant(grant)
    policy.verify_ready()
    for path, sha in json.loads((DIRECTORY / 'FINAL_CPU_READY.json').read_text())['external_supervisor_SHA256'].items():
        assert policy.digest(path) == sha
    observed = int((DIRECTORY / 'observed-suite.exit').read_text())
    assert observed in (0, 5), 'Incomplete or failed native suite; reject partial'
    assert int((FROZEN / 'suite.exit').read_text()) == observed
    drain = json.loads((DIRECTORY / 'container-drain.json').read_text())
    assert drain['all_owned_drained'] and drain['stop_reason'] is None
    assert json.loads((DIRECTORY / 'HOST_RELEASE.json').read_text())['GPU1_lock_probe_exit'] == 0
    groups = {}
    seals = {}
    total_records = total_samples = 0
    run = 'nv-normal-v5-performance-gpu1-v1'
    for round_index in range(3):
        for phase, label in enumerate(policy.ORDER):
            for family, shapes in (('representative', policy.REPRESENTATIVE), ('production', policy.PRODUCTION)):
                stem = FROZEN / 'phases' / f'r{round_index}-p{phase}-{label}-{family}'
                assert stem.with_suffix('.exit').read_text().strip() == '0'
                status = json.loads(stem.with_suffix('.drain.json').read_text())
                assert status['all_owned_drained'] and status['failure'] is None
                for line in stem.with_suffix('.resources.jsonl').read_text().splitlines():
                    assert json.loads(line)['maintenance'] == []
                path = FROZEN / 'results' / run / family / f'round{round_index}-phase{phase}-{label}.json'
                record = json.loads(path.read_text())
                assert record['grant_sha256'] == grant_sha and record['GPU'] == 1
                assert record['canonical_version'] == 'canonical-v5-icpx' and record['family'] == family
                assert record['provider']['label'] == label
                assert record['provider']['after_manifest_sha256'] == policy.EXPECTED_MANIFEST
                assert len(record['results']) == 4
                for row, shape in zip(record['results'], shapes, strict=True):
                    assert row['label'] == label and row['shape'] == list(shape)
                    assert row['round'] == round_index and row['phase'] == phase
                    assert row['warmup'] == 10 and len(row['samples_us']) == 20
                    assert all(type(sample) in (int, float) and math.isfinite(sample) and sample > 0 for sample in row['samples_us'])
                    assert row['output']['all_elements_checked'] and row['post_timing_output']['all_elements_checked']
                    assert row['output']['exact'] and row['post_timing_output']['exact']
                    assert row['post_timing_output']['sha256'] == row['output']['sha256']
                    assert all(row['scales'][name]['exact'] and row['scales'][name]['all_elements_checked'] for name in ('c', 'sf'))
                    groups.setdefault((family, tuple(shape), row['mode']), []).append(row)
                    total_records += 1
                    total_samples += len(row['samples_us'])
                seals[str(path.relative_to(FROZEN))] = policy.digest(path)
    assert total_records == 96 and total_samples == 1920 and len(groups) == 8
    results = []
    for (family, shape, mode), rows in groups.items():
        assert len(rows) == 12
        baseline = rows[0]
        for row in rows:
            assert row['inputs'] == baseline['inputs'] and row['counts'] == baseline['counts']
            assert row['output']['sha256'] == baseline['output']['sha256']
            for name in ('c', 'sf'):
                assert row['scales'][name]['sha256'] == baseline['scales'][name]['sha256']
        experts, count, channels, inner = shape
        cache = FROZEN / 'results' / run / family / f'E{experts}-M{count}-N{channels}-K{inner}'
        for name, tensor_dtype, tensor_shape in (('c', torch.bfloat16, (experts, channels, inner // 16)),
                                               ('sf', torch.uint8, (experts, inner // 32, channels)),
                                               ('output', torch.float32, (count, channels))):
            directory = cache / mode if name == 'output' else cache
            actual_digest, chunk_seals = cache_digest(directory, name, tensor_dtype, tensor_shape)
            assert actual_digest == (baseline['output']['sha256'] if name == 'output' else baseline['scales'][name]['sha256'])
            seals.update(chunk_seals)
        pooled = {'source': [], 'after': []}
        rounds = []
        for round_index in range(3):
            pair = rows[round_index * 4:round_index * 4 + 4]
            medians = {}
            drifts = {}
            for label, slots in (('source', (0, 3)), ('after', (1, 2))):
                samples = [sample for slot in slots for sample in pair[slot]['samples_us']]
                pooled[label] += samples
                medians[label] = statistics.median(samples)
                slot_medians = [statistics.median(pair[slot]['samples_us']) for slot in slots]
                drifts[label] = (max(slot_medians) / min(slot_medians) - 1) * 100
            rounds.append({'round': round_index, 'role_median_us': medians, 'within_round_drift_pct': drifts,
                           'after_vs_source_pct': (medians['after'] / medians['source'] - 1) * 100})
        medians = {label: statistics.median(samples) for label, samples in pooled.items()}
        delta = (medians['after'] / medians['source'] - 1) * 100
        between = {label: (max(row['role_median_us'][label] for row in rounds) / min(row['role_median_us'][label] for row in rounds) - 1) * 100 for label in pooled}
        within = max(value for row in rounds for value in row['within_round_drift_pct'].values())
        hold = delta > 5 or any(row['after_vs_source_pct'] > 5 for row in rounds) or within > 5 or max(between.values()) > 5
        results.append({'family': family, 'shape': shape, 'mode': mode, 'rounds': rounds,
                        'pooled_median_us': medians, 'pooled_MAD_us': {label: statistics.median(abs(value - medians[label]) for value in samples) for label, samples in pooled.items()},
                        'after_vs_source_pct': delta, 'max_within_round_drift_pct': within,
                        'between_round_drift_pct': between, 'hold': hold, 'outliers_dropped': 0})
    expected_exit = 5 if any(row['hold'] for row in results) else 0
    assert expected_exit == observed
    original = json.loads((FROZEN / 'PERFORMANCE_ANALYSIS.json').read_text())
    assert original['actual_exit'] == expected_exit and original['samples'] == 1920
    for independent, frozen in zip(results, original['cases'], strict=True):
        for key in ('family', 'mode', 'pooled_median_us', 'pooled_MAD_us', 'after_vs_source_pct', 'max_within_round_drift_pct', 'between_round_drift_pct', 'hold'):
            assert independent[key] == frozen[key], key
    assert not torch.xpu.is_initialized()
    policy.save(DIRECTORY / 'INDEPENDENT_FULL_RAW_ANALYSIS.json', {'actual_CPU_audit_exit': 0,
        'actual_performance_exit': expected_exit, 'records': total_records, 'samples': total_samples,
        'cases': results, 'raw_sha256': seals, 'GPU_initialized': False,
        'performance_accepted': False, 'automatic_retry': False,
        'reference_scope': 'representatives full rows; production CPU FP32 max3rows/active expert only round0 phases0/1; all valid cached outputs/scales bitwise all96 records',
        'interpretation': 'SOURCE versus normalPORT integration fidelity; not preopt optimization benefit'})
    print(json.dumps({'actual_CPU_audit_exit': 0, 'actual_performance_exit': expected_exit, 'records': total_records, 'samples': total_samples, 'cases': results}))


if __name__ == '__main__':
    main()
