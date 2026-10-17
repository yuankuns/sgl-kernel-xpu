#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4
evidence="$artifact/official-after-precision-v4-gpu11-v2"
prepared="$artifact/official-after-precision-v4-v1"
site="$artifact/canonical-v4-icpx/site"
source_tests=/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014/tests
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTHONPATH="$site:$artifact:$source_tests"
python=/opt/gfx-deps/venv/bin/python
echo "$BASHPID" > "$evidence/container-runtime.pid"
grep Cpus_allowed_list /proc/self/status > "$evidence/container-affinity.txt"
for round_id in 0 1 2 3; do
  set +e
  "$python" "$prepared/after_accuracy_only.py" --round "$round_id" --evidence "$evidence" > "$evidence/round-$round_id-after.log" 2>&1
  status=$?
  set -e
  echo "$status" > "$evidence/round-$round_id-after.exit"
  if (( status != 0 )); then tail -c 5000 "$evidence/round-$round_id-after.log"; exit "$status"; fi
done
set +e
"$python" "$prepared/compare_after_accuracy.py" --evidence "$evidence" > "$evidence/source-after-paired-precision.log" 2>&1
paired_status=$?
set -e
echo "$paired_status" > "$evidence/source-after-paired-precision.exit"
set +e
"$python" -m pytest -p no:cacheprovider -p conftest -p cooldown -v --run-long-tests "$prepared/test_original_40_normal_package.py" > "$evidence/original-40-after.log" 2>&1
status=$?
set -e
echo "$status" > "$evidence/original-40-after.exit"
tail -c 1500 "$evidence/original-40-after.log"
if (( status != 0 )); then exit "$status"; fi
grep -Eq '40 passed' "$evidence/original-40-after.log" || exit 1
exit "$paired_status"
