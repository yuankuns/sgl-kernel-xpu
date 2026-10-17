#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/routing-quant
directory="$artifact/fresh-source-full-accuracy-v1"
source_tests=/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014/tests
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 ONEAPI_DEVICE_SELECTOR=level_zero:cpu
export PYTHONPATH="$artifact/fresh-source-icpx-native-v2/site"
set +e
taskset -c 64-71 /opt/gfx-deps/venv/bin/python "$artifact/check_source_precision_package.py" --kind fresh --output "$directory/normal-source-precheck.json" > "$directory/normal-source-precheck.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/normal-source-precheck.exit"
(( status == 0 )) || exit "$status"
set +e
taskset -c 64-71 /opt/gfx-deps/venv/bin/python "$directory/cpu_driver.py" > "$directory/CPU-driver-provenance.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/CPU-driver-provenance.exit"
(( status == 0 )) || exit "$status"
set +e
taskset -c 64-71 /opt/gfx-deps/venv/bin/python -m pytest --collect-only -q -p no:cacheprovider "$source_tests/test_moe_routing.py" "$source_tests/test_per_token_group_quant_mxfp4.py" "$source_tests/test_topk_sigmoid.py" "$source_tests/test_routing_benchmark_scope.py" > "$directory/CPU-collect-original.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/CPU-collect-original.exit"
exit "$status"
