#!/usr/bin/env bash
set -euo pipefail
root=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/bf16-gemm
round="$root/v4-after-accuracy-prewave-v1"
[[ "${BATCHES:-8}" == 8 && "${ZE_AFFINITY_MASK:-}" == 4 ]] || exit 2
export PYTHONPATH="$root/canonical-v4-icpx/site"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
[[ ! -e "$round/after.log" && ! -e "$round/pytest.log" ]] || exit 2
printf '%q ' env "PYTHONPATH=$PYTHONPATH" taskset -c 0-15 python -B "$round/after_case.py" > "$round/after.command"
printf '\n' >> "$round/after.command"
set +e
bash "$round/after.command" > "$round/after.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$round/after.exit"
tail -n 3 "$round/after.log"
[[ "$status" == 0 ]] || exit "$status"
printf '%q ' env "PYTHONPATH=$PYTHONPATH" taskset -c 0-15 python -B -m pytest /root/worktrees/sgl-kernel-xpu-up/pr/muse-bf16-gemm-xe35/tests/test_bf16_gemm_nt_xpu.py -q -p no:cacheprovider > "$round/pytest.command"
printf '\n' >> "$round/pytest.command"
set +e
bash "$round/pytest.command" > "$round/pytest.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$round/pytest.exit"
tail -n 4 "$round/pytest.log"
exit "$status"
