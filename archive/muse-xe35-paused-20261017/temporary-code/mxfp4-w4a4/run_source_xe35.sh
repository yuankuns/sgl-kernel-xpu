#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
if [[ ! -d "$artifact/source-venv" ]]; then
  cp -a "$artifact/port-venv" "$artifact/source-venv"
fi
python="$artifact/source-venv/bin/python"
"$python" -m pip install --ignore-installed --no-deps "$artifact"/source-wheels/*.whl > "$artifact/source-install.log" 2>&1
set +e
"$python" -m pytest -p no:cacheprovider -p cooldown --run-long-tests -v /root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014/tests/test_mxfp4_moe_w4a4.py > "$artifact/source-tests.log" 2>&1
status=$?
echo "$status" > "$artifact/source-tests.exit"
set -e
if [[ "$status" != 0 ]]; then
  tail -35 "$artifact/source-tests.log"
  exit "$status"
fi
set +e
"$python" compare_production.py --implementation source --round 0 > "$artifact/BAAB-0-source.log" 2>&1
status=$?
echo "$status" > "$artifact/BAAB-0-source.exit"
set -e
tail -c 2500 "$artifact/BAAB-0-source.log"
exit "$status"
