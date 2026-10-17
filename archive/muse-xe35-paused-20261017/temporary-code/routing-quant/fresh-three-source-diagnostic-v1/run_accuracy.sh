#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/routing-quant
directory="$artifact/fresh-three-source-diagnostic-v1"
helpers=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4
source_tests=/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014/tests
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTHONPATH="$artifact/fresh-source-icpx-native-v2/site:$helpers"
python=/opt/gfx-deps/venv/bin/python
echo "$BASHPID" > "$directory/container-runtime.pid"
grep Cpus_allowed_list /proc/self/status > "$directory/container-affinity.txt"
set +e
"$python" "$directory/accuracy_capture.py" --serving-batch 8 --isl 4096 --osl 16 --tp 1 --ep 1 > "$directory/independent-four-component-accuracy.log" 2>&1
capture_status=$?
set -e
echo "$capture_status" > "$directory/independent-four-component-accuracy.exit"
set +e
"$python" -m pytest -p no:cacheprovider -p cooldown -vv --showlocals \
  "$source_tests/test_per_token_group_quant_mxfp4.py::TestPerTokenGroupQuantFP4XPU::test_fused_swiglu_matches_separate_dtype_surface[dtype0-0.5-2.40625]" \
  "$source_tests/test_per_token_group_quant_mxfp4.py::TestPerTokenGroupQuantFP4XPU::test_fused_swiglu_cpu_reference[256-dtype0]" \
  "$source_tests/test_per_token_group_quant_mxfp4.py::TestPerTokenGroupQuantFP4XPU::test_fused_swiglu_cpu_reference[8192-dtype0]" > "$directory/original-three-pytest.log" 2>&1
pytest_status=$?
set -e
echo "$pytest_status" > "$directory/original-three-pytest.exit"
tail -c 3000 "$directory/original-three-pytest.log"
if (( capture_status != 0 )); then exit "$capture_status"; fi
exit "$pytest_status"
