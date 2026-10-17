import os
import subprocess
import sys
import time
from contracts import ROOT, write_json
from resource_runner import snapshot


def main():
    assert os.environ.get('ONEAPI_DEVICE_SELECTOR') == 'level_zero:cpu'
    assert os.sched_getaffinity(0) == set(range(96, 112))
    child = subprocess.Popen([sys.executable, '-c', 'import subprocess,sys,time; child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(2)"]);time.sleep(1);child.wait()'])
    try:
        time.sleep(0.5)
        tree = snapshot(child.pid)
        assert len(tree) == 2 and all(record['rss_bytes'] > 0 for record in tree.values())
        assert all(record['starttick'] > 0 for record in tree.values())
    finally:
        child.wait()
    write_json(ROOT / 'RESOURCE_MONITOR_CPU_REGRESSION.json', {'actual_exit': 0, 'two_level_descendants': tree, 'GPU_initialized': False})
    print('resource descendant/stdin-independent CPU regression0')


if __name__ == '__main__':
    main()
