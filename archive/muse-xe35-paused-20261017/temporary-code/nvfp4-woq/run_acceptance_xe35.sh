#!/usr/bin/env bash
set -uo pipefail
printf '%s\n' 'Disabled old GPU9 official performance launcher: GPU9 accuracy only; formal NV performance requires explicit main GPU5 grant after FMHA' >&2
exit 2
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq
for name in BATCHES BATCH BATCH_SIZE SERVING_BATCH; do
  if [[ -v "$name" && "${!name}" != 8 ]]; then
    printf '%s must be 8 before GPU acquisition\n' "$name" >&2
    exit 2
  fi
done
for specification in ISL:4096 OSL:16 TP:1 EP:1; do
  name=${specification%:*}
  value=${specification#*:}
  if [[ -v "$name" && "${!name}" != "$value" ]]; then
    printf '%s must be %s before GPU acquisition\n' "$name" "$value" >&2
    exit 2
  fi
done
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --gpu-lock --gpu 9 --workdir /root -- bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 5 bash -c '
  set -euo pipefail
  artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq
  target=/root/worktrees/sgl-kernel-xpu-up/pr/muse-nvfp4-woq-xe35
  export BATCHES=8 ISL=4096 OSL=16 TP=1 EP=1
  export LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}
  python="$artifact/target-venv/bin/python"
  unset PYTHONPATH
  run_step_xe35() {
    name="$1"
    shift
    set +e
    "$@" >"$artifact/$name.log" 2>&1
    status=$?
    printf "%s\n" "$status" >"$artifact/$name.exit"
    tail -n 40 "$artifact/$name.log"
    set -e
    return "$status"
  }
  run_step_xe35 package-gpu9-preflight "$python" -c "import os,sys,pathlib,torch,sgl_kernel; assert sgl_kernel.is_xe3_arch(); assert pathlib.Path(sgl_kernel.__file__).parent == pathlib.Path(sys.prefix)/\"lib/python3.12/site-packages/sgl_kernel\"; print(sys.prefix,sgl_kernel.__file__); print(torch.__version__,torch.xpu.get_device_name(0),os.environ.get(\"ZE_AFFINITY_MASK\"))"
  run_step_xe35 original-accuracy-gpu9 "$python" -m pytest -q "$target/tests/test_nvfp4_woq_moe.py" --junitxml="$artifact/original-accuracy-gpu9.xml"
  for suite in tails production; do
    controls=()
    rounds=3
    if [[ "$suite" == tails ]]; then controls+=(--tail-only); rounds=1; fi
    labels=(port source source port)
    for ((round=0; round<rounds; round++)); do
      for phase in 0 1 2 3; do
        label="${labels[$phase]}"
        unset PYTHONPATH
        if [[ "$label" == source ]]; then export PYTHONPATH="$artifact/native-diagnostic/source/site"; fi
        run_step_xe35 "official-$suite-r$round-p$phase-$label" "$python" "$artifact/run_official_case.py" \
          "$label" --run "official-gpu9-$suite" --round "$round" --phase "$phase" "${controls[@]}"
      done
    done
    unset PYTHONPATH
    run_step_xe35 "official-$suite-analysis" "$python" "$artifact/analyze_official_acceptance.py" "official-gpu9-$suite"
  done
' >"$artifact/acceptance-gpu9.log" 2>&1
status=$?
printf '%s\n' "$status" >"$artifact/acceptance-gpu9.exit"
tail -n 50 "$artifact/acceptance-gpu9.log"
exit "$status"
