import json
import os
from pathlib import Path


def pytest_sessionstart(session):
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 80))
    assert os.environ['OMP_NUM_THREADS'] == '8'


def pytest_collection_finish(session):
    directory = Path(__file__).resolve().parent
    nodes = [item.nodeid for item in session.items]
    expected = 14 if 'silu-and-mul-opmath' in os.environ['CURRENT_SOURCE_TEST_PHASE'] else 1300
    assert len(nodes) == expected
    (directory / (os.environ['CURRENT_SOURCE_TEST_PHASE'] + '-actual-collected-nodeids.json')).write_text(json.dumps(nodes, indent=2) + '\n')


def pytest_runtest_logreport(report):
    directory = Path(__file__).resolve().parent
    record = {'nodeid': report.nodeid, 'phase': report.when, 'outcome': report.outcome,
              'longrepr': str(report.longrepr) if report.longrepr else None}
    with (directory / (os.environ['CURRENT_SOURCE_TEST_PHASE'] + '-per-case-actual-reports.jsonl')).open('a') as stream:
        stream.write(json.dumps(record) + '\n')


def pytest_sessionfinish(session, exitstatus):
    directory = Path(__file__).resolve().parent
    (directory / (os.environ['CURRENT_SOURCE_TEST_PHASE'] + '-pytest-session-actual-exit.json')).write_text(json.dumps({
        'actual_exit': int(exitstatus), 'testscollected': session.testscollected, 'testsfailed': session.testsfailed,
        'failures_not_mapped_to_success': True}) + '\n')
