# NVFP4 weight-only grouped GEMM validation

The fused-experts API defaults to `compute_mode="auto"`, which selects BF16.
Pass `compute_mode="bf16"` or `compute_mode="mxfp8"` explicitly to choose
computation precision. Process environment no longer chooses this precision.
FP16 activations remain unsupported; tests check rejection before device work.

```bash
MAX_JOBS=1 cmake --build build --target sgl-ops-sycl-blockwise_moe_nvfp4_woq sgl-ops-sycl-nvfp4_woq sgl-ops-sycl-blockwise_moe_mxfp8 sgl-ops-sycl-mxfp4_moe_scale_layout -j2
python -m pytest tests/test_nvfp4_woq_moe.py -v
python -m pytest tests/test_nvfp4_woq_moe.py --run-long-tests -v
python -m pytest tests/test_nvfp4_woq_moe.py --run-long-tests -m long_test -v
python benchmark/bench_nvfp4_woq_group_gemm.py --groups 256 --m 1 --n 8192 --k 4096
CUTLASS=/root/worktrees/cutlass_internal/muse-w4a4-e87870b bash benchmark/ctimer/build_nvfp4_woq_ctimer.sh
python benchmark/ctimer/run_nvfp4_woq_ctimer.py --suite bf16 --groups 256 --routed-rows 64 --shapes 1:8192:4096 --randomize --skip-bf16-copy --warmup 5 --iterations 21
```

Configure with `DPCPP_SYCL_TARGET=cri` and `MAX_JOBS=1` before building to limit
device-link parallelism. The Xe35/SYCL platform, profiling, ABI, and fixed 512-GRF
compiler settings remain. The standalone ctimer's compiler-version definition
matches the repository's compatibility definition; it is not a runtime tuning
switch and requires re-audit when changing toolchains.

Long cases use the `long_test` pytest marker and `--run-long-tests`, not an
environment selector. Benchmark shapes are explicit CLI data; no serving-batch
environment guard or per-shape sleep remains. The ctimer no longer advertises
cache/tile options that set environment variables without any kernel consumer.
It measures device events through the kernel entry; the group benchmark retains
its existing event/graph timing method.

The shared MXFP4 scale-relayout helper checks tensor shapes without copying
device row counts to the host. Callers must provide nonnegative counts, an input
surface covering their sum, and `sum(round_up(count, 4))` output rows. The former
environment-controlled D2H diagnostic is removed to keep graph/collective safety.

Muse-Spark shapes: 256 experts, top-k 8, hidden/intermediate 8192/4096,
gate/up `N=8192,K=8192`, down `N=8192,K=4096`, 64 decode routed rows and
262144 prefill routed rows. Validation artifacts and source-aligned baseline
libraries live in `review-validation-artifacts/cleanup-20261016/muse_nvfp4_woq/`
under the host's centralized worktree area. Untracked `review_logs/` are historical
user artifacts; their old environment/cache/cooldown commands are not current.
