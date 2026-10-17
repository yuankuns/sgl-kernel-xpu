import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

import policy


STOP = False
KNOWN = {}
TOKEN = 'nv-v5-perf1-' + uuid.uuid4().hex


def interrupt(signum, frame):
    global STOP
    STOP = True


def processes():
    records = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            records[int(path.name)] = {'state': fields[0], 'ppid': int(fields[1]), 'pgid': int(fields[2]), 'start_ticks': int(fields[19]), 'rss': int(fields[21]) * os.sysconf('SC_PAGE_SIZE')}
        except (OSError, ValueError, IndexError):
            continue
    return records


def owned(records):
    selected = {os.getpid()}
    changed = True
    while changed:
        additions = {pid for pid, record in records.items() if record['ppid'] in selected}
        changed = bool(additions - selected)
        selected |= additions
    for pid, record in records.items():
        try:
            if ('NV_PERFORMANCE_TOKEN=' + TOKEN).encode() in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0'):
                selected.add(pid)
        except OSError:
            pass
    for pid in selected - {os.getpid()}:
        if pid in records:
            KNOWN[pid] = records[pid]['start_ticks']
    return {pid: record for pid, record in records.items() if pid in KNOWN and record['start_ticks'] == KNOWN[pid] and record['state'] != 'Z'}


def drain():
    evidence = []
    for sig, seconds in ((signal.SIGTERM, 10), (signal.SIGKILL, 10)):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            active = owned(processes())
            if not active:
                return evidence
            evidence.append({'signal': int(sig), 'PID_startticks': {str(pid): record['start_ticks'] for pid, record in active.items()}, 'PGIDs': sorted({record['pgid'] for record in active.values()})})
            for pid in active:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            time.sleep(0.2)
    raise RuntimeError('Owned descendant drainage failed')


def run(command, prefix, environment, expected_stack):
    policy.save(prefix.with_suffix('.command.json'), {'argv': command, 'PYTHONPATH': environment['PYTHONPATH'], 'CPU_affinity': list(range(80, 96))})
    failure = None
    with prefix.with_suffix('.log').open('x') as output, prefix.with_suffix('.resources.jsonl').open('x') as resources:
        child = subprocess.Popen(command, env=environment, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        KNOWN[child.pid] = processes()[child.pid]['start_ticks']
        policy.save(prefix.with_suffix('.started.json'), {'PID': child.pid, 'start_ticks': KNOWN[child.pid], 'epoch': time.time()})
        while child.poll() is None:
            active = owned(processes())
            available = int(next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')).split()[1]) * 1024
            tree = sum(record['rss'] for record in active.values())
            peak_process = max((record['rss'] for record in active.values()), default=0)
            contamination = policy.maintenance()
            resources.write(json.dumps({'epoch': time.time(), 'tree_RSS_gib': tree / 2**30, 'largest_process_RSS_gib': peak_process / 2**30, 'MemAvailable_gib': available / 2**30, 'maintenance': contamination, 'owned': active}) + '\n')
            resources.flush()
            if STOP or contamination or policy.stack() != expected_stack or tree > 24 * 2**30 or peak_process > 16 * 2**30 or available < 96 * 2**30:
                failure = 'signal/maintenance/stack/resource guard; no automatic retry or cap increase'
                drain()
                break
            time.sleep(1)
        status = child.wait()
    with prefix.with_suffix('.exit').open('x') as output:
        output.write(str(status) + '\n')
    lingering = owned(processes())
    cleanup = drain() if lingering else []
    policy.save(prefix.with_suffix('.drain.json'), {'actual_exit': status, 'failure': failure, 'cleanup': cleanup, 'known_PID_startticks': KNOWN, 'all_owned_drained': not owned(processes())})
    if status or failure:
        raise RuntimeError(f'Actual phase exit {status}: {failure}')


def analysis(run_name, grant_path):
    groups = {}
    for round_index in range(3):
        for phase, label in enumerate(policy.ORDER):
            for family in ('representative', 'production'):
                path = policy.DIRECTORY / 'results' / run_name / family / f'round{round_index}-phase{phase}-{label}.json'
                record = json.loads(path.read_text())
                assert record['grant_sha256'] == policy.digest(grant_path) and record['GPU'] == 1 and not record['accuracy_only']
                expected = policy.REPRESENTATIVE if family == 'representative' else policy.PRODUCTION
                assert [tuple(result['shape']) for result in record['results']] == expected
                for result in record['results']:
                    assert result['warmup'] == 10 and len(result['samples_us']) == 20
                    assert result['output']['exact'] and result['post_timing_output']['exact']
                    assert all(result['scales'][name]['exact'] for name in ('c', 'sf'))
                    groups.setdefault((family, tuple(result['shape']), result['mode']), []).append(result)
    stats = []
    for key, phases in groups.items():
        assert len(phases) == 12
        roles = {'source': [], 'after': []}
        rounds = []
        for round_index in range(3):
            group = phases[4 * round_index:4 * round_index + 4]
            row = {'round': round_index}
            for label, slots in (('source', (0, 3)), ('after', (1, 2))):
                values = [value for slot in slots for value in group[slot]['samples_us']]
                roles[label].extend(values)
                row[label + '_median_us'] = statistics.median(values)
                medians = [statistics.median(group[slot]['samples_us']) for slot in slots]
                row[label + '_drift_pct'] = (max(medians) / min(medians) - 1) * 100
            row['after_vs_source_pct'] = (row['after_median_us'] / row['source_median_us'] - 1) * 100
            rounds.append(row)
        medians = {label: statistics.median(values) for label, values in roles.items()}
        delta = (medians['after'] / medians['source'] - 1) * 100
        drift = max(row[label + '_drift_pct'] for row in rounds for label in roles)
        between = {label: (max(row[label + '_median_us'] for row in rounds) / min(row[label + '_median_us'] for row in rounds) - 1) * 100 for label in roles}
        hold = delta > 5 or any(row['after_vs_source_pct'] > 5 for row in rounds) or drift > 5 or max(between.values()) > 5
        stats.append({'family': key[0], 'shape': key[1], 'mode': key[2], 'rounds': rounds, 'pooled_median_us': medians, 'pooled_MAD_us': {label: statistics.median(abs(value - medians[label]) for value in values) for label, values in roles.items()}, 'after_vs_source_pct': delta, 'max_within_round_drift_pct': drift, 'between_round_drift_pct': between, 'hold': hold, 'outliers_dropped': 0, 'performance_accepted': False})
    assert len(stats) == 8
    status = 5 if any(row['hold'] for row in stats) else 0
    policy.save(policy.DIRECTORY / 'PERFORMANCE_ANALYSIS.json', {'actual_exit': status, 'cases': stats, 'samples': 1920, 'performance_accepted': False, 'interpretation': 'SOURCE-to-PORT integration fidelity; not preopt optimization benefit', 'automatic_retry': False})
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-authorized', action='store_true')
    parser.add_argument('--grant', type=Path)
    args = parser.parse_args()
    policy.preflight()
    if not args.main_authorized or args.grant is None:
        parser.error('No default launch; fresh main GPU1 grant required')
    grant = policy.verify_grant(args.grant)
    while time.time() < grant['start_epoch']:
        if STOP:
            raise SystemExit(143)
        time.sleep(min(1, grant['start_epoch'] - time.time()))
    policy.runtime_grant()
    run_name = 'nv-normal-v5-performance-gpu1-v1'
    phase_directory = policy.DIRECTORY / 'phases'
    phase_directory.mkdir(exist_ok=False)
    status = 1
    try:
        for round_index in range(3):
            for phase, label in enumerate(policy.ORDER):
                for family in ('representative', 'production'):
                    policy.runtime_grant()
                    prefix = phase_directory / f'r{round_index}-p{phase}-{label}-{family}'
                    environment = dict(os.environ, PYTHONPATH=str(policy.SITES[label]), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', NV_PERFORMANCE_TOKEN=TOKEN)
                    environment['LD_LIBRARY_PATH'] = '/opt/gfx-deps/profiling_tools/lib:' + environment.get('LD_LIBRARY_PATH', '')
                    command = [sys.executable, '-B', str(policy.DIRECTORY / 'run_case.py'), label, '--round', str(round_index), '--phase', str(phase), '--run', run_name, '--family', family]
                    run(command, prefix, environment, grant['running_stack'])
        status = analysis(run_name, args.grant)
    finally:
        cleanup = drain()
        policy.save(policy.DIRECTORY / 'SUITE_DRAIN.json', {'known_PID_startticks': KNOWN, 'cleanup': cleanup, 'all_owned_drained': not owned(processes()), 'exit_before_finally': status})
        with (policy.DIRECTORY / 'suite.exit').open('x') as output:
            output.write(str(status) + '\n')
    raise SystemExit(status)


if __name__ == '__main__':
    for signal_number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signal_number, interrupt)
    main()
