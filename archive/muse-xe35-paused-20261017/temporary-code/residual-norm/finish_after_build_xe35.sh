#!/usr/bin/env bash
set -euo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
while [[ ! -f "$artifact/port_full_build.exit" ]]; do
  sleep 5
done
status="$(cat "$artifact/port_full_build.exit")"
if [[ "$status" != 0 ]]; then
  printf 'Full build failed with exit %s; GPU validation is not launched.\n' "$status"
  exit "$status"
fi
bash "$artifact/run_validation_xe35.sh"
