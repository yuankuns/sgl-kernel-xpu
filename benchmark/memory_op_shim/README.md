# Memory target test loader

`memory_test_ops` registers only the four memory operations. It links
against locally built `sgl-ops-sycl-PagedAlloc` and
`sgl-ops-sycl-WriteReqToTokenPool` targets, so accuracy and device-time
benchmarks can run without building the unrelated `common_ops` dependencies.

```bash
cmake -S benchmark/memory_op_shim -B <shim-build> \
  -DPython3_EXECUTABLE=<python> \
  -DMEMORY_ALLOC_LIBRARY=<build>/src/libsgl-ops-sycl-PagedAlloc.so \
  -DMEMORY_WRITE_LIBRARY=<build>/src/libsgl-ops-sycl-WriteReqToTokenPool.so
cmake --build <shim-build> --target memory_test_ops -j1
python benchmark/bench_memory_compare.py \
  --test-library <shim-build>/libmemory_test_ops.so \
  --memory-source <sglang>/python/sglang/kernels/ops/memory
```

The historical tile-tuning scripts require temporary entry points that have
been removed from the production source. The shim registers only the four
production operations.

The benchmark loads SGLang's `allocator.py` and `common.py` directly when
`--memory-source` is supplied. This avoids importing server dependencies that
are absent from a minimal test image; the Triton kernel functions are unchanged.
