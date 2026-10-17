#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/bf16-gemm
repo=/root/worktrees/sgl-kernel-xpu-up/pr/muse-bf16-gemm-xe35
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1
exec 8>"$artifact/diagnostic-port/.abi-forward-build.lock"
flock 8
[[ "$(cat "$artifact/targeted-diagnostic.exit")" == 0 ]]
site="$artifact/diagnostic-port/site/sgl_kernel"
payload=libsgl-ops-sycl-bf16_gemm_nt_port_payload.so
cp "$artifact/targeted-diagnostic-build/src/$payload" "$site/$payload"
torch_root=$(python -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent)')
abi=$(python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
c++ -O2 -std=c++20 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI="$abi" \
  -I"$repo/include" -I"$torch_root/include" -I"$torch_root/include/torch/csrc/api/include" \
  "$artifact/diagnostic/xe35/bf16_gemm_abi_forward.cpp" \
  -L"$site" -Wl,--no-as-needed -l:"$payload" -L"$torch_root/lib" -ltorch_cpu -ltorch -lc10 \
  -Wl,--no-undefined -Wl,-rpath,'$ORIGIN' -Wl,-soname,libsgl-ops-sycl-bf16_gemm_nt.so \
  -o "$site/libsgl-ops-sycl-bf16_gemm_nt.so"
nm -D --defined-only -C "$site/$payload" | grep bf16_gemm_nt_xpu_xe35
nm -D --defined-only -C "$site/libsgl-ops-sycl-bf16_gemm_nt.so" | grep bf16_gemm_nt_xpu
readelf -d "$site/libsgl-ops-sycl-bf16_gemm_nt.so" | grep -E 'NEEDED|SONAME|RUNPATH'
sha256sum "$site/$payload" "$site/libsgl-ops-sycl-bf16_gemm_nt.so" "$site/"common_ops*.so > "$artifact/diagnostic-native.sha256"
