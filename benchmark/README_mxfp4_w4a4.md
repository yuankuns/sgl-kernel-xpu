# MXFP4 W4A4 grouped GEMM validation

```bash
MAX_JOBS=1 cmake --build build --target sgl-ops-sycl-moe_grouped_mm_mxfp4_w4a4 sgl-ops-sycl-mxfp4_moe_scale_layout sgl-ops-sycl-blockwise_moe_mxfp4 -j2
python -m pytest tests/test_mxfp4_moe_w4a4.py -v
python -m pytest tests/test_mxfp4_moe_w4a4.py --run-long-tests -v
python -m pytest tests/test_mxfp4_moe_w4a4.py --run-long-tests -m long_test -v
python benchmark/bench_mxfp4_decode.py --projection both --warmup 10 --iterations 50
python benchmark/bench_mxfp4_main_compare.py --rows 256 257 --rounds 2
```

Configure with `DPCPP_SYCL_TARGET=cri` and `MAX_JOBS=1` before building to limit
device-link parallelism. The Xe35/SYCL platform and fixed 512-GRF code-generation
settings remain. Long cases use `long_test`/`--run-long-tests`, not environment
selection. Both benchmarks use XPU device events, preserve synchronization and
warmup, and have no per-run/shape sleep. The comparison benchmark requires a
properly registered main op rather than an undeclared local registration module.

Both scale relayout and grouped GEMM validate shape metadata without copying
device counts to the host. Callers must provide nonnegative expert counts whose
sum fits the activation surface and allocate `sum(round_up(count, 4))` A-scale
rows. The former environment-controlled D2H diagnostic is removed: it broke
graph capture and could drain unmatched collectives. Tests retain shape rejection
coverage and validate ragged counts using a correctly padded scale surface.

Muse-Spark shape metadata: 256 experts, top-k 8, batch 8, hidden/intermediate
4096/4096, gate/up `N=8192,K=4096`, down `N=4096,K=4096`. Serving batch is
not an environment guard for these explicit benchmark shapes. Baseline sources,
libraries, hashes, and same-device comparison artifacts live in
`review-validation-artifacts/cleanup-20261016/muse_mxfp4_w4a4/` under the host's
centralized worktree area. The benchmark filenames and decode boundary test name
are generic; there are no build or CI references to the former benchmark names.
