#!/usr/bin/env bash
set -euo pipefail
[[ "${ZE_AFFINITY_MASK:-}" == 4 && "${BATCHES:-8}" == 8 ]] || exit 2
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/moe-permutation
evidence="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
evidence_name="${evidence##*/}"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
for cycle in 1 2 3; do
round=canonical-v4-icpx-formal-${evidence_name#formal-v4-perf-}-round${cycle}-BAAB
labels=(source port port source)
for index in 0 1 2 3; do
  label="${labels[$index]}"
  name="$round-$index-$label"
  if [[ "$label" == port ]]; then export PYTHONPATH="$artifact/canonical-v4-icpx/site"; else export PYTHONPATH="$artifact/fresh-source-icpx-v1/site"; fi
  [[ ! -e "$evidence/$name.log" && ! -e "$evidence/$name.json" ]] || exit 2
  command=(taskset -c 96-111 python -B "$evidence/run_case.py" "$label" "$name" --canonical-version canonical-v4-icpx --grant-file "$evidence/fresh-grant.json" --warmup 5 --calls 100 --repeat 30)
  printf '%q ' env "PYTHONPATH=$PYTHONPATH" "${command[@]}" > "$evidence/$name.command"
  printf '\n' >> "$evidence/$name.command"
  set +e
  "${command[@]}" > "$evidence/$name.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$evidence/$name.exit"
  tail -n 3 "$evidence/$name.log"
  [[ "$status" == 0 ]] || exit "$status"
done
done
