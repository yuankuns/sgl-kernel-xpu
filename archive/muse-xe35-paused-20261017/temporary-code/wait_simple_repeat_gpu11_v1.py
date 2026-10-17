import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    for name in ('BATCHES', 'BATCH_SIZE', 'SERVING_BATCH'):
        if os.environ.get(name, '8') != '8':
            raise SystemExit('Muse-Spark serving scope is B8-only')
    print('GPU11 repeat CPU-only waiter; original round must end and memory/position must pass', flush=True)
    original_exit = ROOT / 'simple-gpu11-diagnostic-round.exit'
    while not original_exit.is_file():
        time.sleep(2)
    for component in ('memory', 'position'):
        for suffix in ('accuracy-before', 'accuracy-after', 'perf-B0', 'perf-A0', 'perf-A1', 'perf-B1'):
            status_file = ROOT / component / f'diagnostic-{suffix}.exit'
            if status_file.read_text().strip() != '0':
                raise SystemExit(f'Required component did not pass: {status_file}')
    environment = dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015')
    command = [
        '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh',
        '--gpu-lock', '--gpu', '11', '--workdir', '/root/sgl-kernel-xpu-up', '--',
        'bash', '/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh',
        '7', 'bash', '/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/repeat_simple_diagnostic_container_v1.sh',
    ]
    with (ROOT / 'simple-gpu11-diagnostic-repeat1.log').open('w') as output:
        result = subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT)
    (ROOT / 'simple-gpu11-diagnostic-repeat1.exit').write_text(str(result.returncode) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
