#!/usr/bin/env bash
set -euo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/routing-quant
directory="$artifact/fresh-source-full-accuracy-v1"
container_artifact="${artifact/\/home\/syk/\/root}"
set +e
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --workdir "$container_artifact/fresh-source-full-accuracy-v1" -- bash -c 'artifact="$1"; source /opt/gfx-deps/env.sh; export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 ONEAPI_DEVICE_SELECTOR=level_zero:cpu PYTHONPATH="$artifact/fresh-source-icpx-native-v2/site:$artifact"; taskset -c 64-71 /opt/gfx-deps/venv/bin/python "$artifact/check_source_precision_package.py" --kind fresh --output "$artifact/fresh-source-full-accuracy-v1/normal-source-precheck.json" > "$artifact/fresh-source-full-accuracy-v1/normal-source-precheck.log" 2>&1; status=$?; echo "$status" > "$artifact/fresh-source-full-accuracy-v1/normal-source-precheck.exit"; ((status==0)) || exit "$status"; taskset -c 64-71 /opt/gfx-deps/venv/bin/python "$artifact/fresh-source-full-accuracy-v1/cpu_driver.py" > "$artifact/fresh-source-full-accuracy-v1/CPU-driver-provenance.log" 2>&1; status=$?; echo "$status" > "$artifact/fresh-source-full-accuracy-v1/CPU-driver-provenance.exit"; ((status==0)) || exit "$status"; source_tests=/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014/tests; taskset -c 64-71 /opt/gfx-deps/venv/bin/python -m pytest --collect-only -q -p no:cacheprovider "$source_tests/test_moe_routing.py" "$source_tests/test_per_token_group_quant_mxfp4.py" "$source_tests/test_topk_sigmoid.py" "$source_tests/test_routing_benchmark_scope.py" > "$artifact/fresh-source-full-accuracy-v1/CPU-collect-original.log" 2>&1; status=$?; echo "$status" > "$artifact/fresh-source-full-accuracy-v1/CPU-collect-original.exit"; exit "$status"' bash "$container_artifact" > "$directory/CPU-preflight.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/CPU-preflight.exit"
(( status == 0 )) || exit "$status"
taskset -c 64-71 python3 "$directory/grant.py" > "$directory/grant-preparation.log" 2>&1
