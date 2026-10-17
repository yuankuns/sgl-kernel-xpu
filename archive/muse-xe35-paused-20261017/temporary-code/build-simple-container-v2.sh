#!/usr/bin/env bash
set -euo pipefail
umask 000
root=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
export MAX_JOBS=1
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
for component in memory position logits-vocab; do
  source="/root/worktrees/sgl-kernel-xpu-up/pr/muse-${component}-xe35"
  artifact="$root/$component"
  build="$artifact/full-default-build"
  mkdir -p "$artifact"
  exec 8>"$artifact/build.lock"
  flock 8
  set +e
  cmake -S "$source" -B "$build" -G Ninja \
    -DDPCPP_SYCL_TARGET=cri -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON3_EXECUTABLE=/opt/gfx-deps/venv/bin/python \
    -DSKBUILD_SABI_COMPONENT=Development.SABIModule -DSKBUILD_SABI_VERSION=3.10 \
    -DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL="$root/deps/sycl-tla-7b56" \
    -DSGL_BUILD_MEM_FILE_LIMIT_GIB=8 > "$artifact/configure-v2.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$artifact/configure-v2.exit"
  if (( status != 0 )); then tail -50 "$artifact/configure-v2.log"; exit "$status"; fi
  printf '%s\n' "cmake --build $build --parallel 8" > "$artifact/full-default-build-v2.command"
  set +e
  cmake --build "$build" --parallel 8 > "$artifact/full-default-build-v2.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$artifact/full-default-build-v2.exit"
  if (( status != 0 )); then tail -50 "$artifact/full-default-build-v2.log"; exit "$status"; fi
  python "$root/stage_simple_official_v2.py" "$source" "$build" "$artifact" \
    > "$artifact/official-stage-v2.log" 2>&1
  printf '%s\n' 0 > "$artifact/official-stage-v2.exit"
  flock -u 8
done
