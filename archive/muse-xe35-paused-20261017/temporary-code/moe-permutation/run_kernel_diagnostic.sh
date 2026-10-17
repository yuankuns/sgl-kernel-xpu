#!/usr/bin/env bash
set -euo pipefail
[[ "${ZE_AFFINITY_MASK:-}" == 10 ]] || { echo "wrong GPU lease; expected 10" >&2; exit 2; }
[[ "${BATCHES:-8}" == 8 ]] || { echo 'BATCHES must be 8 before GPU work' >&2; exit 2; }
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/moe-permutation
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
round="${1:-kernel-diagnostic-BAAB}"
labels=(source diagnostic-port diagnostic-port source)
echo 'PORT-KERNEL DIAGNOSTIC ONLY; NOT OFFICIAL INTEGRATION ACCEPTANCE'
for index in 0 1 2 3; do
  label="${labels[$index]}"
  name="$round-$index-$label"
  export PYTHONPATH="$artifact/$label/site"
  printf '%s\n' "env PYTHONPATH=$PYTHONPATH taskset -c 96-111 python $artifact/run_case.py $label $name --diagnostic" > "$artifact/$name.command"
  set +e
  taskset -c 96-111 python "$artifact/run_case.py" "$label" "$name" --diagnostic > "$artifact/$name.log" 2>&1
  status=$?
  set -e
  echo "$status" > "$artifact/$name.exit"
  tail -n 3 "$artifact/$name.log"
  [[ "$status" == 0 ]] || exit "$status"
done
set +e
taskset -c 96-111 python "$artifact/analyze.py" "$round" > "$artifact/$round-analysis.log" 2>&1
status=$?
set -e
echo "$status" > "$artifact/$round-analysis.exit"
tail -n 14 "$artifact/$round-analysis.log"
exit "$status"
