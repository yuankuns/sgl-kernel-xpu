#!/usr/bin/env bash
set -euo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
while [[ ! -f "$artifact/diagnostic-ready" ]]; do
  if [[ -f "$artifact/diagnostic_gpu3.exit" ]]; then
    printf 'GPU3 launcher already ended: %s\n' "$(cat "$artifact/diagnostic_gpu3.exit")"
    exit 0
  fi
  sleep 2
done
test ! -f "$artifact/diagnostic-launch"
test "$(cat "$artifact/diagnostic_port_accuracy.exit")" = 0
test "$(cat "$artifact/diagnostic_compare_accuracy.exit")" = 0
expected='bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm/validate_diagnostic_container_xe35.sh'
actual="$(ps -p 1796643 -o args=)"
test "$actual" = "$expected"
children="$(ps --ppid 1796643 -o comm=)"
test -z "$children" || test "$children" = sleep
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 \
  /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --workdir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm \
  -- bash -c 'test "$(ps -p 60516 -o args=)" = "$1"; kill -TERM 60516' bash "$expected"
while [[ ! -f "$artifact/diagnostic_gpu3.exit" ]]; do
  sleep 1
done
flock -n /tmp/intel-gpu-3.lock true
mv "$artifact/diagnostic-ready" "$artifact/diagnostic-gpu3-completed-ready"
printf '%s\n' 'GPU3 strict accuracy/comparison completed; waiting barrier shell terminated before timing; GPU3 lock released; no new GPU0-3 launch permitted.'
