#!/usr/bin/env bash
set -euo pipefail
lane="${1:?lane 0..7 is required}"
shift
if [[ ! "$lane" =~ ^[0-7]$ ]] || (( $# == 0 )); then
  printf '%s\n' 'usage: build-lane.sh LANE COMMAND [ARG...]' >&2
  exit 2
fi
start=$((lane * 16))
end=$((start + 15))
export CMAKE_BUILD_PARALLEL_LEVEL=8
export MAX_JOBS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
printf 'build lane=%s cpus=%s-%s max_build_jobs=8\n' "$lane" "$start" "$end" >&2
exec taskset -c "$start-$end" "$@"
