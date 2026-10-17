#!/usr/bin/env bash
set -euo pipefail
art=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha
grant=$art/canonical-v5-icpx/gpu10-precision-only-barrier-v1.json
export PYTHONDONTWRITEBYTECODE=1
python3 -S - "$grant" <<'PY'
import json
import sys
import time
from pathlib import Path
grant = json.loads(Path(sys.argv[1]).read_text())
assert grant['coordinator_release'] and grant['gpu'] == 10 and not grant['closed']
assert grant['epoch'] >= grant['issued_epoch'] + 120
print(json.dumps({'barrier_wait': True, 'epoch': grant['epoch'], 'GPU_process_started': False}), flush=True)
time.sleep(max(0, grant['epoch'] - time.time()))
PY
exec bash "$art/v5-model51-gpu10-preparation/run_precision_only.sh" "$grant"
