import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BARRIER = ROOT / 'memory/gpu11-barrier.json'


def main():
    for name in ('BATCHES', 'BATCH_SIZE', 'SERVING_BATCH'):
        if os.environ.get(name, '8') != '8':
            raise SystemExit('Muse-Spark serving scope is B8-only')
    print('GPU11 CPU-only waiter ready; no GPU lock is held', flush=True)
    while True:
        if BARRIER.is_file():
            record = json.loads(BARRIER.read_text())
            if record.get('gpu') != 11:
                raise SystemExit('barrier GPU mismatch; refusing to launch')
            if record.get('coordinator_release') is True:
                delay = float(record['start_epoch']) - time.time()
                if delay > 0:
                    time.sleep(delay)
                break
        time.sleep(2)
    environment = dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015')
    command = [
        '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh',
        '--gpu-lock', '--gpu', '11', '--workdir', '/root/sgl-kernel-xpu-up', '--',
        'bash', '/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh',
        '7', 'bash', '/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/validate-simple-diagnostic-container.sh',
        'memory', 'position', 'logits-vocab',
    ]
    with (ROOT / 'simple-gpu11-diagnostic-round.log').open('w') as output:
        result = subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT)
    (ROOT / 'simple-gpu11-diagnostic-round.exit').write_text(str(result.returncode) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
