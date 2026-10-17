import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    helper = root / 'fresh_source_native_container_v2.py'
    specification = importlib.util.spec_from_file_location('future_monitor', helper)
    monitor = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(monitor)
    child_code = 'import time; payload=bytearray(64*1024*1024); time.sleep(30)'
    parent_code = ('import subprocess,sys,time; '
                   'child=subprocess.Popen([sys.executable,"-c",sys.argv[1]],start_new_session=True); '
                   'print(child.pid,flush=True); time.sleep(30)')
    parent = subprocess.Popen([sys.executable, '-c', parent_code, child_code],
                              start_new_session=True, stdout=subprocess.PIPE, text=True)
    child_pid = int(parent.stdout.readline())
    tracked = {}
    processes = []
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            processes = monitor.resources(parent.pid, tracked)
            child = next((process for process in processes if process['pid'] == child_pid), None)
            if child and child['rss_bytes'] >= 64 * 1024**2:
                break
            time.sleep(0.1)
        assert child and child['rss_bytes'] >= 64 * 1024**2
        assert os.getpgid(parent.pid) != os.getpgid(child_pid)
        assert child['pgrp'] != parent.pid
        parent.terminate()
        parent.wait(timeout=5)
        retained = monitor.resources(parent.pid, tracked)
        assert any(process['pid'] == child_pid for process in retained)
        monitor.terminate_subtree(retained, signal.SIGTERM)
        for _ in range(50):
            if not monitor.resources(parent.pid, tracked):
                break
            time.sleep(0.1)
        assert not monitor.resources(parent.pid, tracked)
        print(json.dumps({'actual_exit': 0, 'cross_Ninja_group_descendant_detected': True,
                          '64MiB_child_RSS_detected': True, 'orphan_retained_PID_starttime_bound': True,
                          'subtree_drained': True, 'compile_started': False, 'GPU_started': False}), flush=True)
    finally:
        for group in [parent.pid, child_pid]:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        parent.wait()


if __name__ == '__main__':
    main()
