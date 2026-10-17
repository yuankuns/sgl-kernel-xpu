#!/usr/bin/env bash
set -euo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/bf16-gemm
container_artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/bf16-gemm
wrapper=/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
[[ "${BATCHES:-8}" == 8 ]] || exit 2
exec 9>"$artifact/.canonical-official-supervisor.lock"
flock -n 9 || { echo 'Another official BF supervisor is active' >&2; exit 2; }
[[ "$(cat "$artifact/canonical-v1/default-all-build.exit")" == 0 ]] || exit 2
[[ -f "$artifact/canonical-v1/source-binary-manifest.json" ]] || exit 2
run_round() {
  local round="$1"
  printf '%q ' env SGLANG_MUSE_CONTAINER="$SGLANG_MUSE_CONTAINER" "$wrapper" --gpu-lock --gpu 4 --workdir "$container_artifact" -- env CANONICAL_VERSION=canonical-v1 bash "$container_artifact/run_baab_container_xe35.sh" "$round" > "$artifact/$round.command"
  printf '\n' >> "$artifact/$round.command"
  set +e
  bash "$artifact/$round.command" > "$artifact/$round.log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" > "$artifact/$round.exit"
  tail -n 8 "$artifact/$round.log"
  [[ "$status" == 0 ]] || return "$status"
  printf '%q ' env SGLANG_MUSE_CONTAINER="$SGLANG_MUSE_CONTAINER" "$wrapper" --workdir "$container_artifact" -- env LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib:/opt/gfx-deps/dpcpp/lib taskset -c 0-15 python -B "$container_artifact/analyze.py" "$round" > "$artifact/$round-analysis.command"
  printf '\n' >> "$artifact/$round-analysis.command"
  set +e
  bash "$artifact/$round-analysis.command" > "$artifact/$round-analysis.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$artifact/$round-analysis.exit"
  tail -n 14 "$artifact/$round-analysis.log"
  return "$status"
}
set +e
(set -e; run_round canonical-v1-BAAB)
status=$?
set -e
if [[ "$status" == 3 ]]; then
  echo '>5% difference/drift: fresh BAAB-retest, no automatic performance acceptance'
  set +e
  (set -e; run_round canonical-v1-BAAB-retest)
  status=$?
  set -e
  if [[ "$status" == 0 ]]; then status=3; fi
fi
printf '%s\n' "$status" > "$artifact/canonical-official-supervisor-v1.exit"
python3 -B "$artifact/finalize_manifest.py"
echo 'BF GPU4 runtime wrappers ended; GPU lock released; review actual gates'
exit "$status"
