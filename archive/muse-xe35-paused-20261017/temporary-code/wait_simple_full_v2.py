import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ORIGINAL_PID = 1963040
ORIGINAL_STARTTIME = '6517955'


def original_live():
    status = Path(f'/proc/{ORIGINAL_PID}/stat')
    try:
        fields = status.read_text().split()
    except FileNotFoundError:
        return False
    return fields[21] == ORIGINAL_STARTTIME and fields[2] != 'Z'


def main():
    print(f'CPU-only official staging recovery waiter: live original build {ORIGINAL_PID}; no GPU work', flush=True)
    while original_live():
        time.sleep(15)
    exit_file = ROOT / 'simple-build-supervisor.exit'
    while not exit_file.is_file():
        time.sleep(2)
    original_status = exit_file.read_text().strip()
    if original_status == '0':
        print('Original full supervisor passed; scoped staging recovery not needed', flush=True)
        return
    log = (ROOT / 'simple-build-supervisor.log').read_text()
    if 'detected dubious ownership' not in log:
        raise SystemExit('Original full build ended for a different reason; no automatic restart')
    full_exit = ROOT / 'memory/full-default-build.exit'
    if not full_exit.is_file() or full_exit.read_text().strip() != '0':
        raise SystemExit('Memory full compilation did not pass; no automatic staging recovery')
    environment = dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015')
    command = [
        '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh',
        '--workdir', '/root/sgl-kernel-xpu-up', '--',
        'bash', '/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh',
        '7', 'bash', '/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-simple-container-v2.sh',
    ]
    with (ROOT / 'simple-build-supervisor-v2.log').open('w') as output:
        result = subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT)
    (ROOT / 'simple-build-supervisor-v2.exit').write_text(str(result.returncode) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
