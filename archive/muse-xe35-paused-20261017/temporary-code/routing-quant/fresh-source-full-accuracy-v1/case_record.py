import json
from pathlib import Path
import os


def pytest_sessionstart(session):
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 72))
    assert os.environ['OMP_NUM_THREADS'] == '8'


def pytest_collection_finish(session):
    directory = Path(__file__).resolve().parent
    nodes = [item.nodeid for item in session.items]
    assert len(nodes) == 1300
    (directory / 'actual-collected-nodeids.json').write_text(json.dumps(nodes, indent=2) + '\n')


def pytest_runtest_logreport(report):
    directory = Path(__file__).resolve().parent
    record = {'nodeid': report.nodeid, 'phase': report.when, 'outcome': report.outcome,
              'longrepr': str(report.longrepr) if report.longrepr else None}
    with (directory / 'per-case-actual-reports.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')


def pytest_sessionfinish(session, exitstatus):
    directory = Path(__file__).resolve().parent
    (directory / 'pytest-session-actual-exit.json').write_text(json.dumps({'actual_exit': int(exitstatus),
        'testscollected': session.testscollected, 'testsfailed': session.testsfailed,
        'expected_baseline_failure_not_mapped_to_success': True}) + '\n')
