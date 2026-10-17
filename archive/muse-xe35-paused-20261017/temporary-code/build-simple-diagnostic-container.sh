#!/usr/bin/env bash
set -euo pipefail
umask 000
root=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
export MAX_JOBS=1
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
for component in memory position logits-vocab; do
  artifact="$root/$component"
  exec 8>"$artifact/build.lock"
  flock 8
  for pair in source-baseline:source-baseline-build native-diagnostic:native-diagnostic-build; do
    source_name="${pair%:*}"
    build_name="${pair#*:}"
    set +e
    cmake -S "$artifact/$source_name" -B "$artifact/$build_name" -G Ninja \
      > "$artifact/$build_name-configure.log" 2>&1
    configure_status=$?
    set -e
    printf '%s\n' "$configure_status" > "$artifact/$build_name-configure.exit"
    if (( configure_status != 0 )); then tail -c 5000 "$artifact/$build_name-configure.log"; exit "$configure_status"; fi
    set +e
    cmake --build "$artifact/$build_name" --parallel 2 > "$artifact/$build_name.log" 2>&1
    status=$?
    set -e
    printf '%s\n' "$status" > "$artifact/$build_name.exit"
    if (( status != 0 )); then tail -c 5000 "$artifact/$build_name.log"; exit "$status"; fi
  done
  python "$root/stage_simple_diagnostic.py" "$component"
  flock -u 8
done
