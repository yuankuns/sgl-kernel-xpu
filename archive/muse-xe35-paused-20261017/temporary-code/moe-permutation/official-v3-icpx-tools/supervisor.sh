#!/usr/bin/env bash
set -euo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/moe-permutation
container_artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/moe-permutation
tools="$artifact/official-v3-icpx-tools"
container_tools="$container_artifact/official-v3-icpx-tools"
wrapper=/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
export CANONICAL_VERSION=canonical-v3-icpx
[[ "${BATCHES:-8}" == 8 ]] || exit 2
[[ -n "${CANONICAL_FRESH_GRANT:-}" ]] || { echo 'Missing fresh v3 grant; no GPU lock requested' >&2; exit 2; }
phase=performance
gpu=4
accuracy_args=()
if [[ "${ACCURACY_ONLY:-0}" == 1 ]]; then phase=accuracy; gpu=10; accuracy_args=(--accuracy-only); fi
prefix="$CANONICAL_VERSION-$phase"
exec 9>"$artifact/.canonical-official-supervisor.lock"
flock -n 9 || { echo 'Another module supervisor is active' >&2; exit 2; }
[[ ! -e "$artifact/$prefix-supervisor.exit" && ! -e "$artifact/$prefix-BAAB.log" ]] || { echo 'Refusing to overwrite previous evidence' >&2; exit 2; }
python3 -B "$tools/official_package_check.py" --grant-file "$CANONICAL_FRESH_GRANT" "${accuracy_args[@]}" > "$artifact/$prefix-prelock-proof.json"
container_grant="${CANONICAL_FRESH_GRANT/#\/home\/syk\//\/root\/}"
run_round() {
  local round="$1"
  printf '%q ' env SGLANG_MUSE_CONTAINER="$SGLANG_MUSE_CONTAINER" "$wrapper" --gpu-lock --gpu "$gpu" --workdir "$container_artifact" -- env CANONICAL_VERSION="$CANONICAL_VERSION" CANONICAL_FRESH_GRANT="$container_grant" ACCURACY_ONLY="${ACCURACY_ONLY:-0}" bash "$container_tools/run_baab_container.sh" "$round" > "$artifact/$round.command"
  printf '\n' >> "$artifact/$round.command"
  set +e
  bash "$artifact/$round.command" > "$artifact/$round.log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" > "$artifact/$round.exit"
  tail -n 6 "$artifact/$round.log"
  [[ "$status" == 0 ]] || return "$status"
  printf '%q ' env SGLANG_MUSE_CONTAINER="$SGLANG_MUSE_CONTAINER" "$wrapper" --workdir "$container_artifact" -- bash -c 'export ONEAPI_DEVICE_SELECTOR=level_zero:cpu LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; exec taskset -c 96-111 python -B "$1" "$2"' _ "$container_tools/analyze.py" "$round" > "$artifact/$round-analysis.command"
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
(set -e; run_round "$prefix-BAAB")
status=$?
set -e
if [[ "$status" == 3 ]]; then
  echo '>5% variation/drift: separate BAAB-retest; not automatic acceptance'
  set +e
  (set -e; run_round "$prefix-BAAB-retest")
  status=$?
  set -e
  if [[ "$status" == 0 ]]; then status=3; fi
fi
printf '%s\n' "$status" > "$artifact/$prefix-supervisor.exit"
echo 'Runtime wrappers ended; review current v3 logs and compiler/grant/source proofs'
exit "$status"
