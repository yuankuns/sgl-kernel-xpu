#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4
directory="$artifact/official-performance-v4-gpu6-v1"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "$BASHPID" > "$directory/container-runtime.pid"
grep Cpus_allowed_list /proc/self/status > "$directory/container-affinity.txt"
for phase in $(seq 0 11); do
  [[ ! -f "$directory/performance-maintenance-hold.json" ]] || exit 125
  kind=after
  site="$artifact/canonical-v4-icpx/site"
  if (( phase % 4 == 0 || phase % 4 == 3 )); then
    kind=source
    site="$artifact/fresh-source-icpx-native-v2/site"
  fi
  export PYTHONPATH="$site:$artifact"
  set +e
  /opt/gfx-deps/venv/bin/python "$directory/graph_phase.py" --serving-batch 8 --isl 4096 --osl 16 --tp 1 --ep 1 --kind "$kind" --phase "$phase" --warmup 5 --calls 8 --repeats 31 > "$directory/phase-$phase-$kind.log" 2>&1
  status=$?
  set -e
  echo "$status" > "$directory/phase-$phase-$kind.exit"
  (( status == 0 )) || { tail -c 4500 "$directory/phase-$phase-$kind.log"; exit "$status"; }
done
set +e
/opt/gfx-deps/venv/bin/python "$directory/analyze.py" > "$directory/performance-analysis.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/performance-analysis.exit"
tail -c 5000 "$directory/performance-analysis.log"
exit "$status"
