#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq
printf '%s\n' 'Recover exact source NVFP4-only 512-GRF policy; one targeted job precedes full outer8/link1'
set +e
bash "$artifact/audit_port_compile_xe35.sh" >"$artifact/native-grf512-targeted-build-v2.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" >"$artifact/native-grf512-targeted-build-v2.exit"
if [[ "$status" != 0 ]]; then
  tail -n 40 "$artifact/native-grf512-targeted-build-v2.log"
  exit "$status"
fi
sha256sum "$artifact/port-compile-audit/src/libsgl-ops-sycl-blockwise_moe_nvfp4_woq_port_payload.so" >"$artifact/native-grf512-targeted-build-v2.sha256"
printf '%s\n' 'Targeted payload built, NOT staged over live GPU9; resume default full unchanged features'
exec bash "$artifact/build_target_jobs8_link1_v1.sh"
