#!/usr/bin/env bash
set -euo pipefail
root=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
sidecar="$root/mxfp4-w4a4/compiler-support-icpx-v1"
run="${1:-v2}"
[[ "$run" =~ ^v[2-9][0-9]*$ ]] || exit 2
export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
printf 'configure-only PID=%s affinity=%s; no full build and no GPU\n' "$$" "$(taskset -pc $$)"
command=(cmake -S "$sidecar/source" -B "$sidecar/build-intelllvm-$run" -G Ninja "-DCMAKE_CXX_COMPILER:FILEPATH=$(command -v icpx)" "-DCMAKE_C_COMPILER:FILEPATH=$(command -v icx)" "-DSYCL_COMPILER:FILEPATH=$(command -v icpx)" -DDPCPP_SYCL_TARGET=cri -DCMAKE_BUILD_TYPE=Release -DPYTHON3_EXECUTABLE=/opt/gfx-deps/venv/bin/python -DSKBUILD_SABI_COMPONENT=Development.SABIModule -DSKBUILD_SABI_VERSION=3.10 "-DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL=$root/deps/sycl-tla-7b56" -DSGL_BUILD_MEM_FILE_LIMIT_GIB=8)
[[ ! -e "$sidecar/build-intelllvm-$run" ]] || exit 125
printf '%q ' "${command[@]}" > "$sidecar/configure-$run.command"
printf '\n' >> "$sidecar/configure-$run.command"
set +e
timeout --signal=TERM --kill-after=15s 300s "${command[@]}"
status=$?
printf '%s\n' "$status" > "$sidecar/configure-$run.exit"
exit "$status"
