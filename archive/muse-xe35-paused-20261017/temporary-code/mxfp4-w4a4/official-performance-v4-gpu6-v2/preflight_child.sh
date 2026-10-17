#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4
directory="$artifact/official-performance-v4-gpu6-v2"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export ONEAPI_DEVICE_SELECTOR=level_zero:cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH="$artifact/fresh-source-icpx-native-v2/site"
set +e
taskset -c 32-47 /opt/gfx-deps/venv/bin/python "$artifact/check_source_precision_package.py" --kind fresh --output "$directory/CPU-normal-source.json" > "$directory/CPU-normal-source.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/CPU-normal-source.exit"
(( status == 0 )) || exit "$status"
export PYTHONPATH="$artifact/canonical-v4-icpx/site"
set +e
taskset -c 32-47 /opt/gfx-deps/venv/bin/python "$artifact/check_canonical_package.py" --module mxfp4-w4a4 --version canonical-v4-icpx --output "$directory/CPU-normal-after.json" > "$directory/CPU-normal-after.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/CPU-normal-after.exit"
(( status == 0 )) || exit "$status"
set +e
taskset -c 32-47 /opt/gfx-deps/venv/bin/python "$directory/cpu_provider.py" > "$directory/CPU-driver-provider.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/CPU-driver-provider.exit"
(( status == 0 )) || exit "$status"
export PYTHONPATH="$artifact"
set +e
taskset -c 32-47 /opt/gfx-deps/venv/bin/python "$directory/verify_precision_fixtures.py" > "$directory/CPU-five-shape-precision-fixtures.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/CPU-five-shape-precision-fixtures.exit"
tail -c 1600 "$directory/CPU-five-shape-precision-fixtures.log"
exit "$status"
