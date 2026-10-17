#!/usr/bin/env bash
set -euo pipefail
printf 'Cancelled by canonical full-build scheduling; main must arrange the official full build.\n' >&2
exit 125
artifact_root=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
w4="$artifact_root/mxfp4-w4a4"
[[ "$(cat "$w4/full-build-v3-stop-tree.exit")" == 0 ]] || exit 125
[[ "$(cat "$w4/compat-native-v2.exit")" == 0 ]] || exit 125
[[ "$(cat "$w4/compat-package-v1.exit")" == 0 ]] || exit 125
python3 - <<'PY'
import json
import pathlib
import time
artifact = pathlib.Path('/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4')
memory = {line.split(':')[0]: int(line.split()[1]) for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if ':' in line and line.split()[1].isdigit()}
record = {'time_epoch': time.time(), 'cpu_lane': 4, 'cpus': '64-79', 'requested_jobs': 8, 'max_jobs': 8, 'inner_device_link_jobs': 1, 'per_file_limit_gib': 8, 'feature_disable': False, 'host_mem_available_gib': memory['MemAvailable'] / 1048576, 'compat_native_and_package_complete': True, 'previous_full_subtree_drained': True}
(artifact / 'full-build-v4-j8-resource-snapshot.json').write_text(json.dumps(record, indent=2) + '\n')
PY
for module in mxfp4-w4a4 routing-quant; do
  artifact="$artifact_root/$module"
  target="/root/worktrees/sgl-kernel-xpu-up/pr/muse-$module-xe35"
  printf '%s\n' "SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --workdir $target -- bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 4 bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4/build_port_full_v4_j8.sh $module" > "$artifact/full-build-v4-j8.command"
  set +e
  SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --workdir "$target" -- bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 4 bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4/build_port_full_v4_j8.sh "$module" > "$artifact/full-build-v4-j8.log" 2>&1
  status=$?
  echo "$status" > "$artifact/full-build-v4-j8.exit"
  echo "$status" > "$artifact/full-build.exit"
  tail -c 2200 "$artifact/full-build-v4-j8.log"
  if [[ "$status" != 0 ]]; then exit "$status"; fi
  set -e
  echo "$module default full v4 normal package completed exit0"
done
