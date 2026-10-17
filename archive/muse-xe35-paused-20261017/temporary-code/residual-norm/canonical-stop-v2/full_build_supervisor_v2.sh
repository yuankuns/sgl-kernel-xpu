#!/usr/bin/env bash
set -euo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
exec 8>"$artifact/.full-supervisor-v2.lock"
flock -n 8
test "$(cat "$artifact/full_build_stop_v1.exit")" = 0
test -f "$artifact/full_build_stop_v1.json"
test -f "$artifact/port_full_build.exit"
test ! -f "$artifact/port_full_build_v2.exit"
printf '%s\n' "version=v2 host_pid=$$ lane=3 cpus=48-63 jobs=8 inner_link_jobs=1 default_features=unchanged"
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
set +e
/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --workdir /root/worktrees/sgl-kernel-xpu-up/pr/muse-residual-norm-xe35 \
  -- bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 3 \
  bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm/build_port_xe35.sh \
  >"$artifact/port_full_build_v2.log" 2>&1
status=$?
echo "$status" > "$artifact/port_full_build_v2.exit"
tail -n 20 "$artifact/port_full_build_v2.log"
if [[ "$status" != 0 ]]; then
  exit "$status"
fi
bash "$artifact/run_validation_xe35.sh"
status=$?
echo "$status" > "$artifact/full_supervisor_v2_validation.exit"
exit "$status"
