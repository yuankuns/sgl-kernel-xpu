import json
import os
from pathlib import Path
import signal
import sys
import time

sys.dont_write_bytecode = True
from run_perf import RUN, digest, local_path, verify


def main():
    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    verify()
    release = json.loads((RUN / 'main-continue-release.json').read_text())
    assert digest(local_path(release['drain_evidence_path'])) == release['drain_evidence_sha256']
    assert digest(RUN / 'performance-grant.json') == release['old_immutable_grant_sha256']
    pid = release['exact_container_PID']
    process = Path('/proc') / str(pid)
    fields = (process / 'stat').read_text().rsplit(') ', 1)[1].split()
    assert pid == 234687 and fields[0] == 'T' and int(fields[19]) == release['process_starttick'] == 8042282
    arguments = (process / 'cmdline').read_bytes().replace(b'\x00', b' ').decode()
    assert arguments.rstrip().endswith('run_perf.py --mode suite')
    assert not (RUN / 'phase-plan.json').exists() and not list(RUN.glob('phases/*/*.json'))
    idle_start = time.time()
    time.sleep(5)
    fields = (process / 'stat').read_text().rsplit(') ', 1)[1].split()
    assert fields[0] == 'T' and int(fields[19]) == 8042282
    continued_at = time.time()
    os.kill(pid, signal.SIGCONT)
    time.sleep(0.25)
    fields = (process / 'stat').read_text().rsplit(') ', 1)[1].split()
    assert fields[0] != 'T' and int(fields[19]) == 8042282
    proof = {'actual_exit': 0, 'container_PID': pid, 'host_PID': release['exact_host_PID'],
        'same_process_starttick': int(fields[19]), 'actual_signal': 'SIGCONT', 'epoch': continued_at,
        'state_after_CONT': fields[0], 'main_release_sha256': digest(RUN / 'main-continue-release.json'),
        'additional_idle_before_CONT_seconds': continued_at - idle_start,
        'pause_before_CONT_seconds': continued_at - release['pause_start_epoch'],
        'wrapper_initial_cooling_seconds_already_completed': 5,
        'runner_shape_cooling_seconds': 2, 'no_live_script_or_old_grant_edit': True,
        'no_restart_no_new_compile': True}
    (RUN / 'actual-CONT-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2), flush=True)


if __name__ == '__main__':
    main()
