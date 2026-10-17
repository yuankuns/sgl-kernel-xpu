#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4
evidence="$artifact/${SOURCE_ACCURACY_TAG:?}"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
python=/opt/gfx-deps/venv/bin/python
cd "$artifact"
historical="$artifact/historical-source-normal-v1/site"
fresh="$artifact/fresh-source-icpx-native-v2/site"
for round_id in 0 1 2 3; do
  kind=fresh
  site="$fresh"
  if [[ "$round_id" == 0 || "$round_id" == 3 ]]; then kind=historical; site="$historical"; fi
  set +e
  PYTHONPATH="$site" "$python" "$artifact/source_accuracy_only.py" --kind "$kind" --round "$round_id" --evidence "$evidence" > "$evidence/round-$round_id-$kind.log" 2>&1
  status=$?
  set -e
  echo "$status" > "$evidence/round-$round_id-$kind.exit"
  if (( status != 0 )); then tail -c 5000 "$evidence/round-$round_id-$kind.log"; exit "$status"; fi
done
set +e
"$python" "$artifact/compare_source_accuracy.py" --evidence "$evidence" > "$evidence/source-paired-precision.log" 2>&1
paired_status=$?
set -e
echo "$paired_status" > "$evidence/source-paired-precision.exit"
for kind in historical fresh; do
  site="$fresh"
  if [[ "$kind" == historical ]]; then site="$historical"; fi
  set +e
  PYTHONPATH="$site:$artifact" "$python" -m pytest -p no:cacheprovider -p cooldown -v --run-long-tests /root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014/tests/test_mxfp4_moe_w4a4.py > "$evidence/original-40-$kind.log" 2>&1
  status=$?
  set -e
  echo "$status" > "$evidence/original-40-$kind.exit"
  tail -c 1200 "$evidence/original-40-$kind.log"
  if (( status != 0 )); then exit "$status"; fi
  grep -Eq '40 passed' "$evidence/original-40-$kind.log" || exit 1
done
exit "$paired_status"
