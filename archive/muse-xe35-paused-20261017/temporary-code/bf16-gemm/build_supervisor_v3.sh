#!/usr/bin/env bash
set -euo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/bf16-gemm
exec 9>"$artifact/.full-supervisor.lock"
flock -n 9 || { echo 'full supervisor already active' >&2; exit 2; }
set +e
bash "$artifact/full-build-attempt3-j8.command" > "$artifact/full-build.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$artifact/full-build.exit"
printf '%s\n' "$status" > "$artifact/full-build-attempt3-j8.exit"
tail -n 18 "$artifact/full-build.log"
exit "$status"
