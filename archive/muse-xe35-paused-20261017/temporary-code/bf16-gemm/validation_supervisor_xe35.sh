#!/usr/bin/env bash
set -euo pipefail
[[ "${BATCHES:-8}" == 8 ]] || { echo 'BATCHES must be 8 before GPU work' >&2; exit 2; }
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/bf16-gemm
container_artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/bf16-gemm
wrapper=/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
printf '%s\n' "waiting for actual full build exit; no GPU lock held while waiting"
while [[ ! -f "$artifact/full-build.exit" ]]; do sleep 30; done
build_status=$(cat "$artifact/full-build.exit")
if [[ "$build_status" != 0 ]]; then
  echo "full default build failed: $build_status; no GPU acceptance run"
  python3 "$artifact/finalize_manifest.py"
  exit "$build_status"
fi
python3 "$artifact/wait_gpu_barrier.py" --phase official
run_round_xe35() {
  round="$1"
  printf '%s\n' "env SGLANG_MUSE_CONTAINER=$SGLANG_MUSE_CONTAINER $wrapper --gpu-lock --gpu 4 --workdir $container_artifact -- bash $container_artifact/run_baab_container_xe35.sh $round" > "$artifact/$round.command"
  set +e
  bash "$artifact/$round.command" > "$artifact/$round.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$artifact/$round.exit"
  tail -n 8 "$artifact/$round.log"
  if [[ "$status" != 0 ]]; then return "$status"; fi
  printf '%s\n' "env SGLANG_MUSE_CONTAINER=$SGLANG_MUSE_CONTAINER $wrapper --workdir $container_artifact -- bash -c 'export LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib:\${LD_LIBRARY_PATH:-}; exec python $container_artifact/analyze.py $round'" > "$artifact/$round-analysis.command"
  set +e
  bash "$artifact/$round-analysis.command" > "$artifact/$round-analysis.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$artifact/$round-analysis.exit"
  tail -n 14 "$artifact/$round-analysis.log"
  return "$status"
}
set +e
(set -e; run_round_xe35 BAAB)
status=$?
set -e
if [[ "$status" == 3 ]]; then
  echo '>5% difference or source drift: repeat matched BAAB, no automatic acceptance'
  set +e
  (set -e; run_round_xe35 BAAB-retest)
  status=$?
  set -e
fi
if [[ -f "$artifact/BAAB-analysis.exit" && "$(cat "$artifact/BAAB-analysis.exit")" == 3 && "$status" == 0 ]]; then
  echo "Repeat completed, but initial >5% finding requires manual analysis; not declaring performance pass"
  status=3
fi
python3 "$artifact/finalize_manifest.py"
echo 'Runtime processes ended; GPU4 wrapper locks released'
exit "$status"
