#!/usr/bin/env bash
set -euo pipefail
root=/root/worktrees/sgl-kernel-xpu-up/pr/muse-fmha-xe35
art=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha
source_site=/root/worktrees/sgl-kernel-xpu/review-validation-artifacts/fmha-before-after-20261017/site-after
site="$art/site-port-diagnostic"
export MAX_JOBS=1
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$site"
cp -a "$source_site/sgl_kernel" "$site/"
for kind in decode prefill; do
  cp "$root/build-xe35/src/libsgl-ops-sycl-fmha_${kind}_64_bf16.so" "$site/sgl_kernel/"
done
ninja_pid=
for candidate in $(pgrep -x ninja); do
  if [[ $(readlink "/proc/$candidate/cwd") == "$root/build-xe35" ]]; then ninja_pid=$candidate; fi
done
if [[ -n $ninja_pid ]]; then
  printf 'Pausing existing Ninja pid=%s; reserving one of eight job slots\n' "$ninja_pid"
  kill -STOP "$ninja_pid"
  trap 'kill -CONT "$ninja_pid" 2>/dev/null || true' EXIT
  while (( $(ps -o stat= --ppid "$ninja_pid" | grep -cv '^[[:space:]]*Z' || true) >= 8 )); do sleep 2; done
fi
abi=$(python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
torch_root=/opt/gfx-deps/venv/lib/python3.12/site-packages/torch
c++ -shared -fPIC -O2 -std=c++20 -fvisibility=hidden -D_GLIBCXX_USE_CXX11_ABI="$abi" \
  -I"$torch_root/include" -I"$torch_root/include/torch/csrc/api/include" -I"$root/include" \
  "$art/diagnostic/xe35/fmha/abi_forward.cpp" -Wl,-z,defs -Wl,-rpath,'$ORIGIN' \
  -L"$site/sgl_kernel" -lsgl-ops-sycl-fmha_decode_64_bf16 -lsgl-ops-sycl-fmha_prefill_64_bf16 \
  -L"$torch_root/lib" -ltorch -ltorch_cpu -lc10 -lc10_xpu \
  -o "$site/sgl_kernel/libsgl-ops-sycl-flash_attentionXe35.so"
python "$art/stage_diagnostic.py"
readelf -d "$site/sgl_kernel/libsgl-ops-sycl-flash_attentionXe35.so"
nm -D -C "$site/sgl_kernel/libsgl-ops-sycl-flash_attentionXe35.so" | grep -E 'mha_fwd|dispatch_xe35'
if [[ -n $ninja_pid ]]; then
  kill -CONT "$ninja_pid"
  printf 'Resumed same Ninja pid=%s\n' "$ninja_pid"
  trap - EXIT
fi
