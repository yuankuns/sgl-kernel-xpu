#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
tree=/root/worktrees/sgl-kernel-xpu-up/pr/muse-residual-norm-xe35
[[ "$#" == 1 && "$1" == canonical-v4-icpx ]] || exit 2
version="$1"
validation="canonical-validation-${version#canonical-}"
evidence="$artifact/$validation"
mkdir "$evidence"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$evidence/cache"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python="$artifact/source-full-venv/bin/python"
before_python="$artifact/source-before-icpx-bounded-v1/isolated-source-venv/bin/python"
stage="$artifact/$version/site"
run_case() {
  local name="$1"
  shift
  printf '%q ' "$@" > "$evidence/$name.command"
  printf '\n' >> "$evidence/$name.command"
  set +e
  "$@" > "$evidence/$name.log" 2>&1
  local status=$?
  set -e
  echo "$status" > "$evidence/$name.exit"
  return "$status"
}
cd "$artifact"
test "$(cat "$artifact/source_accuracy.exit")" = 0
test "$(cat "$artifact/diagnostic_compare_accuracy.exit")" = 0
run_case before_provenance "$python" -B "$artifact/check_canonical_package_v3.py" --mode before-provenance --build-version "$version"
run_case package_before env PYTHONPATH="$stage" "$python" -B "$artifact/check_canonical_package_v3.py" --build-version "$version"
run_case source_accuracy env -u PYTHONPATH "$before_python" -B "$tree/benchmark/bench_fused_norm_residual.py" \
  --provider source --accuracy-only --output-dir "$evidence/source-outputs"
run_case port_accuracy env PYTHONPATH="$stage" "$python" -B "$artifact/check_canonical_package_v3.py" \
  --build-version "$version" --mode accuracy --output-dir "$evidence/port-outputs"
run_case compare_accuracy env -u PYTHONPATH "$python" -B "$artifact/check_canonical_package_v3.py" --build-version "$version" --mode compare
run_case port_pytest env PYTHONPATH="$stage" "$python" -B -m pytest -q "$tree/tests/test_norm.py" \
  -k xe35 -o "cache_dir=$evidence/pytest-cache"
for round in 1 2; do
  for pair in B1:port A1:source A2:source B2:port; do
    name="${pair%%:*}"
    provider="${pair##*:}"
    if [[ "$provider" == port ]]; then
      run_case "round${round}_perf_$name" env PYTHONPATH="$stage" "$python" -B "$artifact/check_canonical_package_v3.py" --build-version "$version" --mode perf
    else
      run_case "round${round}_perf_$name" env -u PYTHONPATH "$before_python" -B "$tree/benchmark/bench_fused_norm_residual.py" \
        --provider source --iterations 256 --samples 31 --cpu-submit-trace
    fi
  done
done
run_case package_after env PYTHONPATH="$stage" "$python" -B "$artifact/check_canonical_package_v3.py" --build-version "$version"
