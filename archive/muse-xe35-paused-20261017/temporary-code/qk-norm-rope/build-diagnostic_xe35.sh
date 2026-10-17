#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
set +e
/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --workdir /root/worktrees/sgl-kernel-xpu-up/pr/muse-qk-norm-rope-xe35 -- \
  bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 2 \
  bash -c '
    export LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib:/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
    export MAX_JOBS=1
    root=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/qk-norm-rope
    target=/root/worktrees/sgl-kernel-xpu-up/pr/muse-qk-norm-rope-xe35
    exec python /root/agent-skills/xpu-build-supervisor/scripts/run-incremental-build.py \
      --jobs 1 --build-dir "$root/diagnostic/build" -- bash -c '\''
        set -euo pipefail
        root="$1"
        target="$2"
        cmake -S "$root/diagnostic" -B "$root/diagnostic/build" -G Ninja \
          -DCMAKE_BUILD_TYPE=Release -DOFFICIAL_TARGET="$target" \
          -DPORT_KERNEL="$target/build/src/libsgl-ops-sycl-FusedQKNormRope_xe35.so"
        cmake --build "$root/diagnostic/build" --parallel "$BUILD_JOBS"
      '\'' bash "$root" "$target"
  ' > "$root/diagnostic/build.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/diagnostic/build.exit"
tail -35 "$root/diagnostic/build.log"
exit "$status"
