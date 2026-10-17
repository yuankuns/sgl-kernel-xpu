#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/routing-quant
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -d "$artifact/port-venv" ]]; then
  cp -a /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4/port-venv "$artifact/port-venv"
fi
export PATH="$artifact/port-venv/bin:$PATH"
export CC=gcc CXX=g++
python /root/agent-skills/xpu-build-supervisor/scripts/run-incremental-build.py --max-jobs 8 --jobs 2 --build-dir "$artifact/source-quant-build" --state-dir "$artifact/source-quant-build-state" -- bash -c '
  set -euo pipefail
  export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
  source=/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014
  cmake -S "$source" -B "$1/source-quant-build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$1/port-venv/bin/python" -DPYTHON3_EXECUTABLE="$1/port-venv/bin/python" -DDPCPP_SYCL_TARGET=cri -DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/deps/sycl-tla-7b56 -DSYCL_COMPILER="$1/source_icx_monitor.sh" -DSYCL_EXECUTABLE="$1/source_icx_monitor.sh"
  cmake --build "$1/source-quant-build" --parallel "$BUILD_JOBS" --target sgl-ops-sycl-per_token_group_quant_fp4
' bash "$artifact" > "$artifact/source-quant-build.log" 2>&1
set +e
python /root/agent-skills/xpu-build-supervisor/scripts/run-incremental-build.py --max-jobs 8 --jobs 2 --build-dir "$artifact/source-native" --state-dir "$artifact/source-build-state" -- python "$artifact/prepare_source.py" > "$artifact/source-package.log" 2>&1
status=$?
echo "$status" > "$artifact/source-package.exit"
set -e
if [[ "$status" != 0 ]]; then
  tail -c 3000 "$artifact/source-package.log"
  exit "$status"
fi
if [[ ! -d "$artifact/source-venv" ]]; then
  cp -a "$artifact/port-venv" "$artifact/source-venv"
fi
"$artifact/source-venv/bin/python" -m pip install --force-reinstall --no-deps "$artifact"/source-wheels/*.whl > "$artifact/source-install.log" 2>&1
