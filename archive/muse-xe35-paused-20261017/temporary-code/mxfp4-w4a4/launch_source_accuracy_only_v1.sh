#!/usr/bin/env bash
set -euo pipefail
while (( $# )); do
  case "$1" in
    --serving-batch) [[ "${2:-}" == 8 ]] || { echo 'Reject non-B8 before GPU'; exit 2; }; shift 2 ;;
    --isl) [[ "${2:-}" == 4096 ]] || exit 2; shift 2 ;;
    --osl) [[ "${2:-}" == 16 ]] || exit 2; shift 2 ;;
    --tp|--ep) [[ "${2:-}" == 1 ]] || exit 2; shift 2 ;;
    *) exit 2 ;;
  esac
done
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4
tag=historical-fresh-source-precision-v3
evidence="$artifact/$tag"
container_artifact="${artifact/\/home\/syk/\/root}"
mkdir "$evidence"
echo "$BASHPID" > "$evidence/host-supervisor.pid"
python3 "$artifact/verify_source_accuracy_grant.py" --prepare "$evidence"
for kind in historical fresh; do
  site="$container_artifact/fresh-source-icpx-native-v2/site"
  if [[ "$kind" == historical ]]; then site="$container_artifact/historical-source-normal-v1/site"; fi
  set +e
  SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --workdir "$container_artifact" -- bash -c 'site="$1"; artifact="$2"; kind="$3"; tag="$4"; source /opt/gfx-deps/env.sh; export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; exec taskset -c 64-79 env ONEAPI_DEVICE_SELECTOR=level_zero:cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$site" /opt/gfx-deps/venv/bin/python "$artifact/check_source_precision_package.py" --kind "$kind" --output "$artifact/$tag/package-$kind.json"' bash "$site" "$container_artifact" "$kind" "$tag" > "$evidence/package-$kind.log" 2>&1
  status=$?
  set -e
  echo "$status" > "$evidence/package-$kind.exit"
  if (( status != 0 )); then cat "$evidence/package-$kind.log"; exit "$status"; fi
done
python3 "$artifact/verify_source_accuracy_grant.py" --seal "$evidence"
set +e
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --gpu-lock --gpu 8 --workdir "$container_artifact" -- bash -c 'artifact="$1"; tag="$2"; source /opt/gfx-deps/env.sh; export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; export SOURCE_ACCURACY_TAG="$tag"; exec taskset -c 64-79 bash "$artifact/run_source_accuracy_only_v1.sh"' bash "$container_artifact" "$tag" > "$evidence/gpu-runtime.log" 2>&1
status=$?
set -e
echo "$status" > "$evidence/gpu-runtime.exit"
tail -c 4000 "$evidence/gpu-runtime.log"
printf 'GPU8 historical/fresh SOURCE precision released; true exit=%s\n' "$status"
exit "$status"
