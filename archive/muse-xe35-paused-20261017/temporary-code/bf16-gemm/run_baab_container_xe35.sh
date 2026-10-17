#!/usr/bin/env bash
set -euo pipefail
accuracy_args=()
expected_gpu=4
if [[ "${ACCURACY_ONLY:-0}" == 1 ]]; then expected_gpu=10; accuracy_args=(--accuracy-only); fi
[[ "${ZE_AFFINITY_MASK:-}" == "$expected_gpu" ]] || { echo "wrong GPU phase/lease" >&2; exit 2; }
[[ "${BATCHES:-8}" == 8 ]] || { echo 'BATCHES must be 8 before GPU work' >&2; exit 2; }
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/bf16-gemm
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
version="${CANONICAL_VERSION:-canonical-v1}"
[[ "$version" != */* ]] || exit 2
round="${1:-$version-BAAB}"
export PYTHONDONTWRITEBYTECODE=1
grant_args=()
labels=(source port port source)
for index in 0 1 2 3; do
  label="${labels[$index]}"
  name="$round-$index-$label"
  if [[ "$label" == port ]]; then export PYTHONPATH="$artifact/$version/site"; else export PYTHONPATH="$artifact/source/site"; fi
  [[ ! -e "$artifact/$name.log" && ! -e "$artifact/$name.json" ]] || { echo "refusing to overwrite raw evidence" >&2; exit 2; }
  printf '%q ' env "PYTHONPATH=$PYTHONPATH" taskset -c 0-15 python "$artifact/run_case.py" "$label" "$name" --canonical-version "$version" "${accuracy_args[@]}" "${grant_args[@]}" > "$artifact/$name.command"
  printf '\n' >> "$artifact/$name.command"
  set +e
  taskset -c 0-15 python "$artifact/run_case.py" "$label" "$name" --canonical-version "$version" "${accuracy_args[@]}" "${grant_args[@]}" > "$artifact/$name.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$artifact/$name.exit"
  tail -n 3 "$artifact/$name.log"
  [[ "$status" == 0 ]] || exit "$status"
done
export PYTHONPATH="$artifact/$version/site"
set +e
taskset -c 0-15 python -m pytest /root/worktrees/sgl-kernel-xpu-up/pr/muse-bf16-gemm-xe35/tests/test_bf16_gemm_nt_xpu.py \
  -q -p no:cacheprovider > "$artifact/$round-pytest.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$artifact/$round-pytest.exit"
tail -n 5 "$artifact/$round-pytest.log"
exit "$status"
