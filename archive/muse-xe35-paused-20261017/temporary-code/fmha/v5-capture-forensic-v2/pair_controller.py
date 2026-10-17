import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import forensic_runner as forensic


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grant', type=Path, required=True)
    parser.add_argument('--gpu-id', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    grant = forensic.check_grant(SimpleNamespace(grant=args.grant, gpu_id=args.gpu_id, role='source'))
    assert grant['controller_sha256'] == forensic.digest(Path(__file__))
    assert grant['forensic_retains_all8_outputs'] and grant['full_reference_each_eager_and_all8_replay_outputs']
    assert os.environ['ZE_AFFINITY_MASK'] == str(args.gpu_id) and os.sched_getaffinity(0) == set(range(16, 32))
    args.output_dir.mkdir(exist_ok=False)
    journal = forensic.Journal(args.output_dir / 'pair-protocol-markers.jsonl')
    reports = {}
    for role in ('source', 'port'):
        forensic.check_grant(SimpleNamespace(grant=args.grant, gpu_id=args.gpu_id, role=role))
        site = ART / ('matched-native-icpx-v1/source/site' if role == 'source' else 'canonical-v5-icpx/site')
        environment = dict(os.environ, FMHA_CANONICAL_VERSION='canonical-v5-icpx', PYTHONDONTWRITEBYTECODE='1')
        environment['PYTHONPATH'] = f'{site}:/root/sglang/python:/root/sglang/sglang_995c774/sglang_meta_common'
        environment['LD_LIBRARY_PATH'] = f'/opt/gfx-deps/profiling_tools/lib:{site}/sgl_kernel:/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
        output = args.output_dir / f'{role}.json'
        command = [sys.executable, str(ROOT / 'forensic_runner.py'), '--role', role,
            '--gpu-id', str(args.gpu_id), '--grant', str(args.grant), '--output', str(output)]
        with (args.output_dir / f'{role}.log').open('x') as log:
            child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            journal.emit('role_child_started', role=role, actual_container_PID=child.pid, command=command)
            status = child.wait()
        with (args.output_dir / f'{role}.exit').open('x') as stream:
            stream.write(f'{status}\n')
        journal.emit('role_child_terminal', role=role, actual_returncode=status, automatic_retry=False)
        if status != 0:
            raise SystemExit(status if status > 0 else 128 - status)
        report = json.loads(output.read_text())
        assert report['forensic_generations_PASS'] and len(report['cases']) == forensic.GENERATIONS
        for case in report['cases']:
            assert case['eager_and_all8_replay_SOURCE_wholebits']
            assert len(case['all_eight_replay_outputs']) == len(case['actual_fwd_calls']) == forensic.CAPTURE_COUNT
        reports[role] = report
    source = reports['source']['cases']
    port = reports['port']['cases']
    for before, after in zip(source, port):
        assert before['generation'] == after['generation']
        assert [row['output_sha256'] for row in before['all_eight_replay_outputs']] == [row['output_sha256'] for row in after['all_eight_replay_outputs']]
    summary = {'actual_exit': 0, 'paired_roles': ['source', 'port'], 'case_name': forensic.CASE_NAME,
        'generations_each': forensic.GENERATIONS, 'capture_each': forensic.CAPTURE_COUNT,
        'all8_replay_full_reference_and_paired_bits_each_generation': True,
        'no_timing_warmup_events': True, 'full89_acceptance': False,
        'lifetime_boundary': 'forensic explicitly holds all eight outputs; not unchanged last-only production protocol',
        'grant_sha256': forensic.digest(args.grant), 'controller_sha256': forensic.digest(Path(__file__)),
        'runner_sha256': forensic.digest(ROOT / 'forensic_runner.py'),
        'role_report_SHA': {role: forensic.digest(args.output_dir / f'{role}.json') for role in reports}}
    with (args.output_dir / 'paired-summary.json').open('x') as stream:
        json.dump(summary, stream, indent=2)
    journal.emit('paired_forensic_terminal_PASS', full89_acceptance=False)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'paired forensic firstfail: {error!r}', file=sys.stderr, flush=True)
        raise SystemExit(2)
