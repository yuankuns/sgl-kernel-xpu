#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
python="$artifact/port-venv/bin/python"
"$python" -c 'import hashlib,json,pathlib,sgl_kernel; root=pathlib.Path(sgl_kernel.__file__).parent; print(json.dumps({"package":str(root),"binaries":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.glob("*.so"))}},indent=2))' > "$artifact/port-binary-provenance.json"
set +e
"$python" -m pytest -p no:cacheprovider -p cooldown --run-long-tests -v /root/worktrees/sgl-kernel-xpu-up/pr/muse-mxfp4-w4a4-xe35/tests/test_mxfp4_moe_w4a4.py > "$artifact/port-tests.log" 2>&1
status=$?
echo "$status" > "$artifact/port-tests.exit"
set -e
if [[ "$status" != 0 ]]; then
  tail -c 4000 "$artifact/port-tests.log"
  exit "$status"
fi
for suffix in log exit; do
  if [[ -f "$artifact/BAAB-0-source.$suffix" ]]; then mv "$artifact/BAAB-0-source.$suffix" "$artifact/BAAB-0-source-initial-before-full.$suffix"; fi
done
for round_id in 0 1 2 3; do
  implementation=port
  python="$artifact/port-venv/bin/python"
  if [[ "$round_id" == 0 || "$round_id" == 3 ]]; then
    implementation=source
    python="$artifact/source-venv/bin/python"
  fi
  set +e
  "$python" compare_production.py --implementation "$implementation" --round "$round_id" > "$artifact/BAAB-$round_id-$implementation.log" 2>&1
  status=$?
  echo "$status" > "$artifact/BAAB-$round_id-$implementation.exit"
  set -e
  if [[ "$status" != 0 ]]; then
    tail -c 4000 "$artifact/BAAB-$round_id-$implementation.log"
    exit "$status"
  fi
done
set +e
"$artifact/port-venv/bin/python" summarize.py > "$artifact/summary-0.log" 2>&1
status=$?
echo "$status" > "$artifact/summary-0.exit"
set -e
cat "$artifact/summary-0.log"
if [[ "$status" != 5 ]]; then exit "$status"; fi
for round_id in 4 5 6 7; do
  implementation=port
  if [[ "$round_id" == 4 || "$round_id" == 7 ]]; then implementation=source; fi
  set +e
  "$artifact/$implementation-venv/bin/python" compare_production.py --implementation "$implementation" --round "$round_id" --iterations 100 > "$artifact/BAAB-$round_id-$implementation.log" 2>&1
  status=$?
  echo "$status" > "$artifact/BAAB-$round_id-$implementation.exit"
  set -e
  if [[ "$status" != 0 ]]; then tail -c 4000 "$artifact/BAAB-$round_id-$implementation.log"; exit "$status"; fi
done
set +e
"$artifact/port-venv/bin/python" summarize.py --offset 4 > "$artifact/summary-4.log" 2>&1
status=$?
echo "$status" > "$artifact/summary-4.exit"
cat "$artifact/summary-4.log"
exit "$status"
