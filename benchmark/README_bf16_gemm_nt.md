# BF16 GEMM validation

Run on a CRI device with the matching oneAPI/PyTorch environment. The operator,
tests, and benchmarks have generic names; model dimensions are workload data.

```bash
MAX_JOBS=1 cmake --build build --target sgl-ops-sycl-bf16_gemm_nt -j2
python -m pytest tests/test_bf16_gemm_nt_xpu.py -v
python benchmark/bench_bf16_gemm_nt.py --m 1 8 9 16 17 32 33 64 --n 32768 --k 8192
python benchmark/bench_bf16_gemm_nt_dispatch.py --shape 8192 8192 --shape 18432 8192 --m 1 8 9 32 33 --output bf16_dispatch.csv
```

Configure with `DPCPP_SYCL_TARGET=cri` and `MAX_JOBS=1` before building to limit
device-link parallelism. The Xe35 target macro and fixed 256-GRF compiler setting
are platform/code-generation configuration, not environment tuning switches.
The benchmark measures XPU events around graph replay and reports time per call.
There is no per-run/shape sleep or `--cooldown` option. Synchronization, warmup,
and event boundaries remain. Historical files in `benchmark/results/` describe
older measurements and do not specify the current validation procedure.

For a targeted-library test shim rather than a full `common_ops` rebuild, see
`review-validation-artifacts/cleanup-20261016/muse_bf16_gemm/` in the host's
centralized worktree area. Builds use their own build-tree locks and a group-wide
parallelism budget, not a global build lock; device runs use the container wrapper
and both GPU locks.
