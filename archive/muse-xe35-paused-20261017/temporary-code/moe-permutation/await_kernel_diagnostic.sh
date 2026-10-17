#!/usr/bin/env bash
set -euo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/moe-permutation
[[ "${BATCHES:-8}" == 8 ]] || { echo 'BATCHES must be 8 before GPU work' >&2; exit 2; }
exec 9>"$artifact/.diagnostic-round.lock"
flock -n 9 || { echo 'diagnostic supervisor already active' >&2; exit 2; }
python3 "$artifact/wait_gpu_barrier.py"
set +e
bash "$artifact/kernel-diagnostic-BAAB.command" > "$artifact/kernel-diagnostic-BAAB.log" 2>&1
status=$?
set -e
echo "$status" > "$artifact/kernel-diagnostic-BAAB.exit"
tail -n 18 "$artifact/kernel-diagnostic-BAAB.log"
echo 'GPU10 diagnostic ended; wrapper lock released; official integration remains pending'
exit "$status"
