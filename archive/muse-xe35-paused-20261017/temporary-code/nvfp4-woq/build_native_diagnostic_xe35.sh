#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq
target=/root/worktrees/sgl-kernel-xpu-up/pr/muse-nvfp4-woq-xe35
export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1
export LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}
exec 8>"$artifact/native-diagnostic-stage.lock"
flock 8
bash "$artifact/audit_port_compile_xe35.sh"
python="$artifact/target-venv/bin/python"
"$python" "$artifact/stage_native_diagnostic.py"
torch_root=$("$python" -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent)')
abi=$("$python" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
site="$artifact/native-diagnostic/port/site/sgl_kernel"
for pair in gemm:blockwise_moe_nvfp4_woq prep:nvfp4_woq; do
  adapter=${pair%:*}
  name=${pair#*:}
  c++ -O2 -std=c++20 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI="$abi" \
    -I"$target/include" -I"$torch_root/include" -I"$torch_root/include/torch/csrc/api/include" \
    "$artifact/native-diagnostic/xe35/${adapter}_source_api.cpp" \
    -L"$site" -Wl,--no-as-needed -l:"libsgl-ops-sycl-${name}_port_payload.so" \
    -L"$torch_root/lib" -ltorch_cpu -ltorch -lc10 \
    -Wl,--no-undefined -Wl,-rpath,'$ORIGIN' -Wl,-soname,"libsgl-ops-sycl-${name}.so" \
    -o "$site/libsgl-ops-sycl-${name}.so"
done
for label in source port; do
  PYTHONPATH="$artifact/native-diagnostic/$label/site" "$python" -c \
    'import sgl_kernel,torch; print(sgl_kernel.__file__,sgl_kernel.common_ops.__file__); print(torch.ops.sgl_kernel.nvfp4_woq_bf16_grouped_mm)'
  find "$artifact/native-diagnostic/$label/site/sgl_kernel" -type f -name '*.so' -exec sha256sum {} + \
    >"$artifact/diagnostic-$label-binaries.sha256"
done
for pair in nvfp4_woq:prep blockwise_moe_nvfp4_woq:grouped_mm; do
  name=${pair%:*}
  nm -D --defined-only -C "$site/libsgl-ops-sycl-${name}_port_payload.so" | grep nvfp4_woq
  nm -D --defined-only -C "$site/libsgl-ops-sycl-${name}.so" | grep nvfp4_woq
done
nm -D --defined-only -C "$site/libsgl-ops-sycl-nvfp4_woq_scale_layout.so" | grep mxfp4_moe_scale_to_mn_major_xe35
readelf -d "$site/libsgl-ops-sycl-blockwise_moe_nvfp4_woq.so"
readelf -d "$site/libsgl-ops-sycl-blockwise_moe_nvfp4_woq_port_payload.so"
printf '%s\n' 'READY GPU9 CPU80-95: native kernel diagnostic only; NOT official full package acceptance; final barrier GPUs4-11' >"$artifact/diagnostic.READY"
