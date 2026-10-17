import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from unittest.mock import patch

from process_tree import ProcessTree, TOKEN_KEY, budget_reason

ROOT = Path(__file__).resolve().parent
for name in ('process_tree.py', 'sidecar.py', 'test_process_tree_cpu.py'):
    ast.parse((ROOT / name).read_text())
child = 'import time; time.sleep(60)'
ninja = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",' + repr(child) + ',"synthetic-child"],start_new_session=True); time.sleep(60)'
root_code = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",' + repr(ninja) + ',"synthetic-Ninja"],start_new_session=True); time.sleep(60)'
token = uuid.uuid4().hex
process = subprocess.Popen([sys.executable, '-c', root_code, 'synthetic-root'], env=dict(os.environ, **{TOKEN_KEY: token}), start_new_session=True)
tracker = ProcessTree(process.pid, token)
try:
    deadline = time.monotonic() + 10
    while True:
        measured = tracker.sample()
        if len(measured['pgids']) >= 3:
            break
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert measured['tree_rss_gib'] > 0
    os.kill(process.pid, signal.SIGKILL)
    process.wait()
    restored = ProcessTree(0, token, known=measured['identities'])
    orphaned = restored.sample()
    assert len(orphaned['live']) >= 2
    result = restored.drain()
    assert result['drained'] and not result['live_after']
    assert len(result['groups_signalled']) >= 2
finally:
    tracker.drain()
    process.wait()

token = uuid.uuid4().hex
process = subprocess.Popen([sys.executable, '-c', child, 'synthetic-signal-exception'], env=dict(os.environ, **{TOKEN_KEY: token}), start_new_session=True)
tracker = ProcessTree(process.pid, token)
try:
    try:
        raise KeyboardInterrupt('synthetic signal path')
    except KeyboardInterrupt:
        assert tracker.drain()['drained']
finally:
    tracker.drain()
    process.wait()

base = {'largest_compiler_rss_gib': 0.1, 'tree_rss_gib': 0.3, 'available_gib': 128}
assert budget_reason(base) is None
assert budget_reason({**base, 'largest_compiler_rss_gib': 8.01}) == 'compiler RSS exceeded 8GiB'
assert budget_reason({**base, 'tree_rss_gib': 16.01}) == 'owned tree RSS exceeded 16GiB'
assert budget_reason({**base, 'available_gib': 95.99}) == 'available memory below 96GiB'
assert budget_reason({**base, 'largest_owned_process_rss_gib': 8.01}) == 'unrecognized owned tool RSS exceeded conservative 8GiB cap'
reused = {'pid': 999999, 'name': 'synthetic', 'state': 'S', 'ppid': 1, 'pgid': 999999, 'session': 999999, 'start_ticks': 200, 'rss_bytes': 1}
tracker = ProcessTree(0, 'not-a-live-token', known=[{**reused, 'start_ticks': 100}])
with patch('process_tree.snapshot', return_value={999999: reused}), patch('process_tree.token_matches', return_value=False):
    assert not tracker.discover()[1]
versioned = {**reused, 'name': 'clang-23', 'rss_bytes': 9 * 1024 ** 3}
tracker = ProcessTree(0, 'synthetic-compiler-only', known=[versioned])
with patch('process_tree.snapshot', return_value={999999: versioned}), patch('process_tree.token_matches', return_value=False):
    assert tracker.sample()['largest_compiler_rss_gib'] == 9
print(json.dumps({'actual_exit': 0, 'CPU_only': True, 'actual_compiler_executed': False, 'GPU_imported': False, 'cross_PGID_count_observed': len(measured['pgids']), 'real_python_tree_rss_gib': measured['tree_rss_gib'], 'reparented_restore_drain': result, 'signal_exception_drain_passed': True, 'PID_starttick_reuse_rejected': True, 'synthetic_8GiB_16GiB_96GiB_boundary_checks_passed': True}))
