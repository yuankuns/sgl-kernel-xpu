#!/usr/bin/env bash
set -euo pipefail
root=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/moe-permutation
stage="$root/source-native-icpx-v1-accuracy"
[[ "${BATCHES:-8}" == 8 ]] || exit 2
[[ "${ZE_AFFINITY_MASK:-}" == 10 ]] || exit 2
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
for label in historical fresh; do
  if [[ "$label" == historical ]]; then export PYTHONPATH="$root/source/site"; else export PYTHONPATH="$root/fresh-source-icpx-v1/site"; fi
  [[ ! -e "$stage/$label.log" && ! -e "$stage/$label.json" ]] || exit 2
  printf '%q ' env "PYTHONPATH=$PYTHONPATH" taskset -c 96-111 python -B "$root/source_accuracy_case.py" "$label" --grant-file "$stage/fresh-grant.json" > "$stage/$label.command"
  printf '\n' >> "$stage/$label.command"
  set +e
  bash "$stage/$label.command" > "$stage/$label.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$stage/$label.exit"
  tail -n 3 "$stage/$label.log"
  [[ "$status" == 0 ]] || exit "$status"
done
