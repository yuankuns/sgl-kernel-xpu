#!/usr/bin/env bash
set -euo pipefail
kind="${1:?source or after}"
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/routing-quant
directory="$artifact/full-source-after-post-v1"
container_artifact="${artifact/\/home\/syk/\/root}"
if [[ "$kind" == source ]]; then
  evidence="$artifact/fresh-source-full-accuracy-v1"
  cores=64-71
else
  [[ "$kind" == after ]] || exit 2
  evidence="$artifact/official-after-accuracy-v4-gpu11-v1"
  cores=72-79
fi
echo "$BASHPID" > "$directory/host-post-$kind.pid"
while [[ ! -f "$evidence/host-launch.exit" ]]; do sleep 10; done
set +e
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --workdir "$container_artifact/full-source-after-post-v1" -- bash -c 'artifact="$1"; kind="$2"; cores="$3"; source /opt/gfx-deps/env.sh; export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; export ONEAPI_DEVICE_SELECTOR=level_zero:cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; directory="$artifact/full-source-after-post-v1"; if [[ "$kind" == source ]]; then export PYTHONPATH="$artifact/fresh-source-icpx-native-v2/site"; taskset -c "$cores" /opt/gfx-deps/venv/bin/python "$artifact/check_source_precision_package.py" --kind fresh --output "$directory/package-post-source.json" > "$directory/package-post-source.log" 2>&1; else export PYTHONPATH="$artifact/canonical-v4-icpx/site"; taskset -c "$cores" /opt/gfx-deps/venv/bin/python "$artifact/../mxfp4-w4a4/check_canonical_package.py" --module routing-quant --version canonical-v4-icpx --output "$directory/package-post-after.json" > "$directory/package-post-after.log" 2>&1; fi; status=$?; echo "$status" > "$directory/package-post-$kind.exit"; ((status==0)) || exit "$status"; taskset -c "$cores" /opt/gfx-deps/venv/bin/python "$directory/audit_provider.py" --kind "$kind" > "$directory/provider-post-$kind.log" 2>&1; status=$?; echo "$status" > "$directory/provider-post-$kind.exit"; exit "$status"' bash "$container_artifact" "$kind" "$cores" > "$directory/package-provider-supervisor-$kind.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/package-provider-supervisor-$kind.exit"
(( status == 0 )) || exit "$status"
set +e
taskset -c "$cores" python3 "$directory/classify_terminal.py" --kind "$kind" > "$directory/CPU-classification-$kind.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/CPU-classification-$kind.exit"
tail -c 3500 "$directory/CPU-classification-$kind.log"
exit "$status"
