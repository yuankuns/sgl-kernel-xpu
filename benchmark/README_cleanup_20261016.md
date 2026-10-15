# QK normalization / RoPE cleanup, 2026-10-16

## Scope and preserved state

Generic source, test, and benchmark filenames already contain no model name;
the two model-named test functions now use `production_shapes` and `gated_stride`.
Shape descriptions may still identify MUSE Spark. Kernel source, build targets,
and the pre-existing dirty `FusedQKNormRopeXe35.h` are preserved byte-for-byte.
No CMake reference needs renaming. Historical untracked entry scripts and logs
remain byte-for-byte evidence, not current executable instructions: their old
environment CLI, test node IDs, and sleeps must not be reused for new runs.

## Environment-to-CLI migration

All benchmark/validator workload choices are explicit CLI arguments. Old
environment variables no longer select workloads or override the operator:

| Previous environment option | Current CLI |
| --- | --- |
| `MUSE_SPARK=1` | `--production-shapes` (`--production-shape` alias) |
| `PROVIDERS=...` | `--providers unfused sglang` (space-separated) |
| `NUM_CHUNKS`, `CHUNK_IDX` | `--num-chunks`, `--chunk-idx` |
| `SINGLE_CONFIG` | `--single-config 'batch,seq,nq,nk,nv,head_dim,is_neox,dtype;...'` (`--config` alias) |
| `MUSE_SPARK_PROFILE` | `--profile` in the main benchmark |
| `QK_IMPL=muse_spark_unfused` | `--impl unfused` in the boundary benchmark |
| `QK_TOKENS` | `--tokens` (space-separated list for boundary; scalar for graph) |
| `QK_VALIDATE_TOKENS` | `--tokens` in the validator |
| `QK_Q_HEADS`, `QK_K_HEADS`, `QK_TAIL_HEADS`, `QK_HEAD_DIM` | `--q-heads`, `--k-heads`, `--tail-heads`, `--head-dim` |
| `QK_SAMPLES`, `QK_WARMUP`, `QK_GRAPH_REPEATS` | `--samples`, `--warmup`, `--repeats` |
| `QK_PROFILE` | `--profile` in the boundary benchmark |
| `QK_DIAG_LIB_PATH` | Fused-path library hash diagnostics always enabled |
| `BATCHES`, `ISL`, `OSL`, `TP`, `EP` | No environment gate; production preset or explicit shape |

Active benchmarks have no sleeps. Device selection (`ZE_AFFINITY_MASK`, runtime
selector), loader paths, and compiler/platform settings are retained.
Provider name `muse_spark_unfused` becomes `unfused`; historical result labels are
not rewritten. The fixed production preset describes B8, ISL4096, OSL16,
TP1/EP1, BF16 Q128/K16/V16/D64, interleaved RoPE. Boundary/graph benchmarks also
cover the untouched gated tail (144 heads). Those layouts are not interchangeable.

## Macro and precision decisions

The kernel contains no `getenv` or environment tuning override.
`SYCL_DISPATCH_FLOATING_TYPES`, `LAUNCH_KERNEL`, and `LAUNCH_XE35_KERNEL` expand
dtype/head-dimension launch dispatch; retain them, along with required platform
compiler/ABI macros. Keep subgroup width, device EU/thread-count discovery, and
automatic launch selection. Keep
`SGL_PRE_SILICON`, `SYCL_INTEL_TARGET=35`, and `__SYCL_TARGET_INTEL_GPU_CRI__`
as required CRI PO platform definitions. Builds expose a pre-existing
`SYCL_DISPATCH_FLOATING_TYPES` redefinition warning from `Utils.h`; this is a
dispatch-helper name collision, not an environment override, and is not changed.
Shape-sensitive traffic thresholds (48/100/300 MiB),
4 KiB sparse-row alignment, and the baseline/wide-path crossover are suspicious
only as portability heuristics: they encode existing measured dispatch, not user
switches. They require revalidation on other hardware, not cleanup removal.
The pre-existing BF16 128D split edits are retained exactly.

Kernel arithmetic is unchanged. Tests cover BF16/FP16, both RoPE styles and
gated strides; the full prefill boundary validator uses BF16, rtol 8e-3 / atol
4e-3, and checks the untouched tail exactly. FP8 remains on the prior baseline
kernel path. Graph/boundary benchmarks and the prefill validator accept
`--dtype bf16|fp16`; FP8 performance validation is not implied.

## Reproduction commands

Use the coordinator's container/GPU launcher with a shared build-job total of
10 per agent; no global compile lock is used after the updated authorization.
The archived `run-entry-800-final.sh` and
`run-entry-735-final.sh` identify image `migration/sglang-muse-syk:rootfs`, the
wheel directory below, the old main/original worktrees, and GPU0's historical
lock. They are provenance only, since their invocations predate the CLI cleanup.
Use a prepared container with the worktree at `/workspace` for the existing
CMake cache, source `/opt/gfx-deps/env.sh`, and prepare the matching installed
`sgl_kernel` package. Commands inside that environment:

```sh
python /root/worktrees/sgl-kernel-xpu/review-validation-artifacts/cleanup-20261016/build_validation.py muse_qk_norm_rope
```

The helper builds only the target and shim in separate before/after artifact
trees using per-tree locks, `ninja -j10`, device link 1. Do not rebuild the old
cache with its historical device-link parallelism or replace an installed wheel.
For direct GPU commands inside the prepared container, select the artifact shim
package (switch `after-site` to `before-site` for the baseline), then verify the
emitted loaded-library hash. Keep identical GPU, dtype, layout and CLI settings:

```sh
export PYTHONPATH=/root/worktrees/sgl-kernel-xpu/review-validation-artifacts/cleanup-20261016/muse_qk_norm_rope/validation/after-site:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}
python -m pytest -q tests/test_fused_qk_norm_rope.py
python benchmark/validate_qk_norm_rope_b8.py --tokens 32768
python benchmark/validate_qk_norm_rope_b8.py --tokens 1439 --q-heads 32 --k-heads 8 --tail-heads 40 --head-dim 128
python benchmark/bench_qk_norm_rope_graph.py --tokens 32768 --repeats 32 --samples 21
python benchmark/bench_qk_norm_rope_boundary.py --tokens 1 8 799 800 832 863 864 32768 --samples 11
python benchmark/bench_qk_norm_rope_boundary.py --tokens 1439 1440 --q-heads 32 --k-heads 8 --tail-heads 40 --head-dim 128 --samples 11
python benchmark/bench_fused_qk_norm_rope.py --production-shapes --providers unfused sglang --profile
python benchmark/bench_fused_qk_norm_rope.py --providers torch sglang --num-chunks 2 --chunk-idx 0 --single-config '1,128,32,8,8,128,false,fp16;1,128,32,8,8,128,false,bf16'
```

Graph XPU events measure replay time divided by repeats; boundary XPU events
measure one op with reset outside timing. `--profile` in the main benchmark
measures fused device kernel events, not reference host time. Do not compare
these different timing methods as if they were identical.

## Baseline artifacts

`/home/syk/worktrees/sgl-kernel-xpu/review-validation-artifacts/cleanup-20261016/muse_qk_norm_rope/`
contains `before.patch` (including the dirty source edits), `before-head.txt`,
`before-status.txt`, complete `baseline/` snapshots, `baseline-files.txt`,
`baseline.sha256`, and `baseline-absent-files.txt` for newly added files.
`binary-baseline/`, `binary-sources.txt`, and `binary-baseline.sha256` preserve
the current target and the main/original targets found via entry scripts:

- Current target: `631078ca19b13dd8506c6b7ade7e35033ecf4774e26ab74e2eae05308ce6c375`.
- Main baseline: `57749179e38893c1710c0f382603f46524e2552e567355fa78e10a156e4f0d99`.
- Original WIP: `1b461d0174ef686fd7e904f3898a71698d5e1c96b3ee24fb07451c9ab89aaf46`.

Historical wheel source:
`/home/syk/muse-container-root/wheel-archive/sgl_kernel-muse-spark-cde14f01b3c44fd64504f2ced16520c16c2cac56/`.
Saved pre-audit binary hashes do not prove agreement with dirty source. After
validation was authorized, both variants were rebuilt from matched source in
isolated artifact build trees; the original build directory remains unchanged.

## Executed validation

The dedicated container is `codex-muse-residual-gfx-20261015`. Runs use the CRI
wrapper with `--workdir /root/... --gpu-lock --gpu 4` and additional host
`flock /tmp/muse-gpu-bench-4.lock`. Before entry, `sudo xpu-smi ps/stats -d 4`
confirmed no compute worker or large allocation; the only VLLM process was a
roughly 20 MiB `vllm bench serve` client. The external wrapper's initial wait is
unchanged; no shape cooldown is added. PCI ID is 8086:674f, SYCL driver
1.15.39229+15, compiler 2026.2.0, Torch 2.14.0a0+gitd35bb04. Model coverage is B8
decode/prefill only, never B64 serving.

The global compile lock was revoked by the coordinating user. Current builds
use per-build-tree locks, `ninja -j10`, and `MAX_JOBS=1` during configuration to
force device linking to one internal job; targeted registration shims build
serially. No wheel/common_ops build or installed-package replacement occurs.
The shim registers the production op and exposes the unchanged production
`fused_qk_norm_rope` Python wrapper. Both variants' kernel source hashes and the
dirty header hash match. Build log/exit/JSON are in
`validation/build-j10-incremental.*` and `validation/build-results.json`.

The complete original pytest file is additionally run unchanged from its
pre-cleanup snapshot, including FP8: **519 identical nodes pass before and
after, with zero failures or new failures**. Its source SHA256 is
`ea3d46fad87d80c9d2d50e3c68e4a8c36699b6f8737ab1522e0bc12d7239a858`.
Results are in `validation/strict-original-preserved-pytest-*`, JUnit XML, and
`validation/strict-original-summary.json`. This original-suite result is
separate from the additional coverage below; no production-tolerance cases
are substituted for original strict tests.

The current renamed test file separately passes 424 BF16/FP16 cases on each
variant, with its 96 FP8 cases deselected in that run. Prefill/tail validators
launch T32768/T32769 and D128 T1439 for both
dtypes, check a documented subset of reference rows, and check all untouched
tail elements exactly. These passes do not imply exhaustive full-row accuracy.
Existing pytest tolerances and reference functions are unchanged. The renamed
gated test gains FP16 coverage using its pre-existing rtol 8e-3 / atol 4e-3,
also already used by the production-shape test for both dtypes.

The initial new FP16 gated coverage tried a stricter rtol/atol 1e-3 and failed
on both kernels: 173 of 75,497,472 elements, greatest absolute difference among
failed elements 0.001953125. Before/after strict diagnostic logs each exit 1.
This is **not** a strict-precision pass. Existing basic FP16 tests still use
1e-3. The initial failure remains in `validation/accuracy-initial-pytest-before.log`.
Supplemental before/after strict failures and complete tensor, same-seed bitwise
comparison/error distributions are retained in `validation/precision-*` and
`validation/precision-full/`. The full CPU-reference check also exposes
pre-existing prefill tolerance violations not covered by sampled validators;
the collector records those failures without calling them passes or changing
the thresholds. Consult its baseline/final distributions and bitwise mismatch
counts rather than inferring no precision regression from the sampled passes.

Fourteen complete-output cases have identical seed-42 input hashes and **zero
before/after bitwise mismatches**, including decode, B8 prefill T32768, tail
T32769, gated T4096, D128 T1439, and D128/Q128/K16/tail144 T4266/T4267 around
the pre-existing dirty header's 300 MiB split boundary, for BF16 and FP16.
All baseline/final error distributions match exactly, and all untouched tails
are bitwise preserved. Default cases are in `validation/precision-full/`; the
dirty split-boundary cases are in `validation/precision-sparse-threshold/`.

The exhaustive CPU-reference diagnostics expose these identical **existing
failures**, which are not claimed as full-reference passes:

| Case | QK elements | Before/after production-bound violations | Global max absolute error |
| --- | ---: | ---: | ---: |
| T32768 D64 BF16 | 301989888 | 71 / 71 | 0.125 |
| T32769 D64 BF16 | 301999104 | 71 / 71 | 0.125 |
| T32768 D64 FP16 | 301989888 | 63 / 63 | 0.015625 |
| T32769 D64 FP16 | 301999104 | 63 / 63 | 0.015625 |

Distribution thresholds are evaluated as FP32
`abs(actual - reference) > atol + rtol * abs(reference)`; the standalone
FP16 strict pytest uses PyTorch's native dtype comparison, so near-boundary
diagnostic counts need not equal its 173 failing elements. The unchanged
original pytest suite passes, but these extra full-row diagnostics still fail
the stated CPU-reference bound in both binaries. Bitwise equality proves no
new output discrepancy for the covered cases, not universal reference accuracy.

GPU4 device timing uses XPU events around 64 captured calls, 31 samples per
process, reset outside timing, and production BF16 plus paired FP16. Initial
AB/BA decode BF16 showed +2.18% latency and triggered four more AB/BA rounds.
In those rounds decode deltas change sign; there is no stable >=1% regression.
Confirmation medians (before -> after, microseconds):

| Shape | Before | After | Latency change |
| --- | ---: | ---: | ---: |
| T8, D64, BF16 | 2.6580 | 2.6695 | +0.43% |
| T32768, D64, BF16 | 1644.7085 | 1642.3715 | -0.14% |
| T8, D64, FP16 | 2.6730 | 2.6790 | +0.22% |
| T32768, D64, FP16 | 1645.0145 | 1640.5305 | -0.27% |
| T1439, Q32/K8/tail40, D128, BF16 | 30.3520 | 30.3670 | +0.05% |

Raw initial and confirmation samples are retained separately in
`validation/performance-{initial,confirmation}-*.log`, with command/exit/hash
records and summary JSON. These are device graph timings, not whole-model
throughput or an optimization-speedup claim.

Host reproduction uses the artifact-only helpers below. The GPU helper acquires
both GPU4 locks and runs the wrapper; run one GPU mode at a time:

```sh
artifacts=/home/syk/worktrees/sgl-kernel-xpu/review-validation-artifacts/cleanup-20261016
bash "$artifacts/run_qk_residual_gpu.sh" muse_qk_norm_rope accuracy production-tolerance
bash "$artifacts/run_qk_residual_gpu.sh" muse_qk_norm_rope strict-original preserved-pytest
bash "$artifacts/run_qk_residual_gpu.sh" muse_qk_norm_rope performance confirmation 4
bash "$artifacts/run_qk_residual_gpu.sh" muse_qk_norm_rope precision full-error-distribution
bash "$artifacts/run_qk_residual_gpu.sh" muse_qk_norm_rope precision-sparse dirty-300m-boundary
```

Historical untracked `.sh`/`.py`, review libraries/logs and all validation
artifacts are excluded from the cleanup commit. Only maintained current
benchmark/validator/tests/docs belong in its cleanup file list. The pre-existing
dirty kernel header is preserved separately for the coordinator's branch review.
