import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
FORENSIC = DIRECTORY.parent


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save(path, value):
    with path.open('x') as output:
        output.write(json.dumps(value, indent=2) + '\n')


def snapshot():
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            result[int(path.name)] = {'PPID': int(fields[1]), 'PGID': int(fields[2]), 'start_ticks': int(fields[19]), 'state': fields[0]}
        except (OSError, ValueError, IndexError):
            pass
    return result


def main():
    grant_path = DIRECTORY / 'main-grant.json'
    grant = json.loads(grant_path.read_text())
    assert set(os.sched_getaffinity(0)) == set(range(80, 96))
    assert os.environ['ZE_AFFINITY_MASK'] == '7'
    assert grant['runner_sha256'] == digest(FORENSIC / 'forensic_runner.py')
    assert grant['CPU_seal_sha256'] == digest(FORENSIC / 'CPU_PROPOSAL_READY_SEAL.json')
    assert grant['CPU_provider_sha256'] == digest(DIRECTORY / 'SOURCE_CPU_PROVIDER.json')
    for filename, sha in grant['launch_helpers_sha256'].items():
        assert digest(DIRECTORY / filename) == sha
    assert grant['input_recipe'] == {'ids': [-1, 0, 1, 202047, 202048, 100, 70000, 150000], 'dtype': 'int64', 'org_start': 0, 'org_end': 202048, 'num_org_padding': 0, 'added_start': 202048, 'added_end': 202048, 'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1}
    save(DIRECTORY / 'container-owned.json', {'PID': os.getpid(), 'identity': snapshot()[os.getpid()], 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'epoch': time.time(), 'grant_sha256': digest(grant_path)})
    while time.time() < grant['start_epoch']:
        time.sleep(min(1, grant['start_epoch'] - time.time()))
    environment = dict(os.environ, ONEAPI_DEVICE_SELECTOR='level_zero:cpu')
    with (DIRECTORY / 'RUNTIME_SOURCE_CPU_PROVIDER.log').open('x') as output:
        probe = subprocess.run([sys.executable, '-B', str(DIRECTORY / 'check_source_runtime_cpu.py')], env=environment, stdout=output, stderr=subprocess.STDOUT)
    with (DIRECTORY / 'RUNTIME_SOURCE_CPU_PROVIDER.exit').open('x') as output:
        output.write(str(probe.returncode) + '\n')
    if probe.returncode:
        raise SystemExit(probe.returncode)
    provider = json.loads((DIRECTORY / 'RUNTIME_SOURCE_CPU_PROVIDER.json').read_text())
    assert provider['actual_exit'] == 0 and provider['Torch_XPU_initialized'] is False
    assert provider['normal_stage_files'] == grant['normal_stage_files']
    assert provider['normal_stage_tree_SHA256'] == grant['normal_stage_tree_sha256']
    assert provider['actual_native_mappings'] == grant['actual_SOURCE_native_mappings']
    assert provider['Torch_native_mappings_SHA256'] == grant['Torch_native_mappings_SHA256']
    assert provider['SOURCE_native_sha256'] == grant['SOURCE_native_sha256']
    command = [sys.executable, '-B', str(FORENSIC / 'forensic_runner.py'), '--main-authorized', '--grant', str(grant_path)]
    save(DIRECTORY / 'runner.command.json', {'argv': command, 'SOURCE_normal_stage': os.environ['PYTHONPATH'], 'runtime_preinit_provider_sha256': digest(DIRECTORY / 'RUNTIME_SOURCE_CPU_PROVIDER.json'), 'epoch': time.time(), 'events_authorized': False, 'timing_samples': 0})
    known = {}
    with (DIRECTORY / 'runner.log').open('x') as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        while child.poll() is None:
            current = snapshot()
            selected = {child.pid}
            changed = True
            while changed:
                additions = {pid for pid, record in current.items() if record['PPID'] in selected}
                changed = bool(additions - selected)
                selected |= additions
            for pid in selected:
                if pid in current:
                    known[pid] = current[pid]
            if not (DIRECTORY / 'runner-owned.json').exists() and child.pid in current:
                save(DIRECTORY / 'runner-owned.json', {'PID': child.pid, 'identity': current[child.pid], 'epoch': time.time(), 'grant_sha256': digest(grant_path)})
            time.sleep(0.1)
        status = child.wait()
    current = snapshot()
    live = {pid: record for pid, record in current.items() if pid in known and record['start_ticks'] == known[pid]['start_ticks'] and record['state'] not in ('Z', 'X')}
    save(DIRECTORY / 'runner-drain.json', {'actual_runner_exit': status, 'known_PID_startticks': known, 'live_owned_descendants': live, 'all_owned_drained': not live, 'epoch': time.time()})
    with (DIRECTORY / 'runner.exit').open('x') as output:
        output.write(str(status) + '\n')
    print(json.dumps({'actual_runner_exit': status, 'all_owned_drained': not live, 'GPU': 7, 'timing_samples': 0}), flush=True)
    raise SystemExit(status if not live else 2)


if __name__ == '__main__':
    main()
