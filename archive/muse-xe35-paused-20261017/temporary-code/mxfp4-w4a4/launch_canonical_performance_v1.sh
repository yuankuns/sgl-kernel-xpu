#!/usr/bin/env bash
set -euo pipefail
module="${1:?module required}"
shift
case "$module" in mxfp4-w4a4) gpu=6 ;; routing-quant) gpu=7 ;; *) exit 2 ;; esac
phase=performance
version=canonical-v4-icpx
while (( $# )); do
  case "$1" in
    --serving-batch) [[ "${2:-}" == 8 ]] || { echo 'Only batch8; rejected before GPU.' >&2; exit 2; }; shift 2 ;;
    --isl) [[ "${2:-}" == 4096 ]] || exit 2; shift 2 ;;
    --osl) [[ "${2:-}" == 16 ]] || exit 2; shift 2 ;;
    --tp|--ep) [[ "${2:-}" == 1 ]] || exit 2; shift 2 ;;
    --phase) case "${2:-}" in precision|performance) phase="$2" ;; *) exit 2 ;; esac; shift 2 ;;
    --version) [[ "${2:-}" =~ ^canonical-v[1-9][0-9]*(-icpx)?$ ]] || exit 2; version="$2"; shift 2 ;;
    *) exit 2 ;;
  esac
done
[[ "$version" =~ ^canonical-v[1-9][0-9]*-icpx$ ]] || { echo 'Old non-icpx canonical runtimes are retired; no GPU work.' >&2; exit 125; }
[[ "$version" != canonical-v3-icpx ]] || { echo 'Failed canonical-v3-icpx is retired; no GPU work.' >&2; exit 125; }
if [[ "$phase" == precision ]]; then gpu=8; fi
root=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
artifact="$root/$module"
common="$root/mxfp4-w4a4"
evidence="$artifact/$version"
[[ -f "$evidence/site/sgl_kernel/__init__.py" ]] || exit 125
[[ "$(cat "$evidence/default-all-build.exit")" == 0 ]] || exit 125
[[ "$(cat "$evidence/supervisor.exit")" == 0 ]] || exit 125
[[ -f "$common/run_canonical_validation_v1.sh" ]] || exit 125
container_root=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
container_artifact="$container_root/$module"
set +e
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --workdir "$container_artifact" -- taskset -c 64-79 env PYTHONDONTWRITEBYTECODE=1 "PYTHONPATH=$container_artifact/$version/site" LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib bash -c '"$1/port-venv/bin/python" "$2/check_canonical_package.py" --module "$3" --version "$4" --output "$1/$4/normal-package-check.json"' bash "$container_artifact" "$container_root/mxfp4-w4a4" "$module" "$version" > "$evidence/normal-package-check.log" 2>&1
status=$?
echo "$status" > "$evidence/normal-package-check.exit"
set -e
[[ "$status" == 0 ]] || { tail -c 5000 "$evidence/normal-package-check.log"; exit "$status"; }
set +e
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --workdir "$container_artifact" -- taskset -c 64-79 env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib "$container_artifact/source-venv/bin/python" "$container_root/mxfp4-w4a4/check_source_baseline.py" --module "$module" --version "$version" > "$evidence/source-baseline-check.log" 2>&1
status=$?
echo "$status" > "$evidence/source-baseline-check.exit"
set -e
[[ "$status" == 0 ]] || { tail -c 5000 "$evidence/source-baseline-check.log"; exit "$status"; }
python3 "$common/wait_canonical_performance_grant.py" --module "$module" --phase "$phase" --version "$version"
set +e
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --gpu-lock --gpu "$gpu" --workdir "$container_artifact" -- taskset -c 64-79 bash "$container_root/mxfp4-w4a4/run_canonical_validation_v1.sh" "$module" "$phase" "$version" > "$evidence/$phase-runtime.log" 2>&1
status=$?
echo "$status" > "$evidence/$phase-runtime.exit"
tail -c 5000 "$evidence/$phase-runtime.log"
printf '\nGPU%s canonical normal-package runtime released; true exit=%s\n' "$gpu" "$status"
exit "$status"
