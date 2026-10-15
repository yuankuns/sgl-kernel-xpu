# Residual normalization cleanup, 2026-10-16

## Scope and preserved state

Only this operator's tests, a generic benchmark, and this document change.
Kernel source, registration, CMake targets, and device configuration remain
unchanged. Existing untracked review scripts and logs are preserved byte-for-byte
as historical evidence, not current entry points. Their sleeps and experimental
`SGL_KERNEL_REVIEW_WG` assignments must not be carried into new validation runs.
The current kernel does not read that environment variable.

The generic test CLI no longer reads `BATCHES` or enforces model serving settings
(`--serving-batch`, `--isl`, `--osl`, `--tp`, `--ep`). Select actual tensor rows and
hidden size directly. `--post-mode`, `--round-pre-norm-input`, and `--output-fp32`
are operator semantics, not launch tuning. No active entry point adds sleeps.
MUSE Spark shapes are rows 8 (decode) and 32768 (B8 x ISL4096 prefill), H8192,
OSL16, TP1/EP1. They do not constrain the generic operator interface.

## Macro and precision decisions

No environment macro or runtime tuning override exists in `FusedNormResidual.cpp`.
Keep vector width, subgroup/workgroup templates, and automatic shape dispatch:
these are implementation/device configuration, not external switches. The
measured row cutoffs (128/256/384/768/3584) remain review-sensitive heuristics;
do not reinterpret them as universal hardware limits or alter them in cleanup.
Keep required upstream SYCL/compiler/ABI definitions and target flags.

The kernel currently requires BF16 branch/weights/output and FP32 stream,
alpha/beta/new-stream, with optional FP32 output. FP16 is not supported by its
contract and is not claimed here. Precision tests compare an independent CPU
reference, output at rtol/atol 2e-2, and residual error against the existing
BF16-ULP bound; optional FP32 output and hidden-size/row boundaries are covered.
No kernel arithmetic or precision setting changes.

## Reproduction commands

The existing container `codex-muse-residual-gfx-20261015` uses image
`migration/sglang-muse-syk:rootfs`, mounts this worktree at `/workspace`, the main
checkout at `/baseline`, and `/home/syk` at `/root`. It was stopped at audit time.
Use the coordinator's container launcher and GPU lock; do not start independent
GPU work. Source `/opt/gfx-deps/env.sh` in that environment. These are commands
inside the prepared container; reserve at most 10 build jobs for this agent:

```sh
python /root/worktrees/sgl-kernel-xpu/review-validation-artifacts/cleanup-20261016/build_validation.py muse_residual_norm
```

The helper builds only the target/shim with per-tree locks, `ninja -j10`, and
device link 1 in source-matched before/after artifact trees. Preserve the original
review registration and libraries as evidence, but use the new shim package
for validation. Switch `after-site` to `before-site` for the baseline. Verify
the benchmark's loaded-library hash; do not use the old cache's historical
device-link parallelism or assume an installed wheel points at the tested build.

```sh
export LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/root/worktrees/sgl-kernel-xpu/review-validation-artifacts/cleanup-20261016/muse_residual_norm/validation/after-site:${PYTHONPATH:-}
python -m pytest -q tests/test_fused_norm_residual.py
python benchmark/bench_fused_norm_residual.py --rows 8 32768 --hidden-size 8192 --round-pre-norm-input
python benchmark/bench_fused_norm_residual.py --rows 128 129 256 257 --hidden-size 8192
python benchmark/bench_fused_norm_residual.py --rows 64 65 384 385 --hidden-size 8200 --post-mode weight --output-fp32
```

The graph benchmark calls the registered production operator with preallocated,
identical input/output tensors; XPU events measure replay and divide by captured
iterations. It excludes allocations, setup, and CPU reference work. Run before
and after with identical arguments, precision mode, device, and timing method.
Historical profiler timings are not interchangeable with this graph timing.
The test file's existing `--profiler` CLI remains available for kernel-event
timing; it includes `run_op` output allocation outside the device kernels.

## Baseline artifacts

`/home/syk/worktrees/sgl-kernel-xpu/review-validation-artifacts/cleanup-20261016/muse_residual_norm/`
contains `before.patch`, `before-head.txt`, `before-status.txt`, complete file
snapshots in `baseline/`, `baseline-files.txt`, `baseline.sha256`, and a list of
new files without a pre-cleanup baseline in `baseline-absent-files.txt`.
`binary-baseline/` holds the current build, original/generic variants and targeted
registration; `binary-sources.txt` records original paths and
`binary-baseline.sha256` records hashes. Current operator hash:
`ebfc8f650280b1f57fef0790b526ddcee3680dd12c15f5dfed9065f7eb99bb69`.
Original review baseline hash:
`a31ad24944fac583b9b073fd8099a2ac27a256136bb28a03ef6d2f12b29c1303`.
An existing binary is preserved evidence, not proof that it matches current
source. The initial no-build restriction was later superseded by validation
authority; actual isolated-build and GPU results are recorded below.

## Executed validation

The dedicated container is `codex-muse-residual-gfx-20261015`. Runs use the CRI
wrapper with `--workdir /root/... --gpu-lock --gpu 4`, plus the extra host lock
`/tmp/muse-gpu-bench-4.lock`. GPU4 health/process checks found no compute worker
or large-memory task; a roughly 20 MiB `vllm bench serve` client is not a worker.
The external wrapper is unchanged; current scripts add no shape waits.
The device is CRI 8086:674f, driver 1.15.39229+15, compiler 2026.2.0, Torch
2.14.0a0+gitd35bb04. No B64 serving is tested.

Both source-matched before/after kernels and registration shims build with
exit 0. Current compile control is per-tree locking, `ninja -j10`, internal
device link 1 (`MAX_JOBS=1` at configuration), shim compile 1, and no global
compile lock. The canceled old-lock build and its stale shim lock recovery
remain in superseded logs; neither was an OOM. Successful final logs are
`validation/build-j10-final.log`, `.exit`, and `validation/build-results.json`.
The original build directory and pre-audit binaries remain untouched; separate
artifact trees resolve the old `/workspace` cache path mismatch.

Before/after each pass all 55 existing pytest cases without changing reference
or tolerances. Each also passes 15 production/tail CPU-reference checks:
rows 8/32768/32769, H8192; rows 385/257, H8200; post modes none/weight/input_scale,
both rounding semantics as applicable, plus optional FP32 output. Large cases
check six boundary/interior rows, not an exhaustive whole-tensor CPU reference.
The unsupported FP16 branch contract is explicitly rejected in both variants
rather than claiming paired FP16 implementation support.

GPU4 performance uses 64 captured calls per replay and 31 device-event samples
per process, production post mode none, pre-input rounding enabled, and the
default BF16/FP32 contract. Initial AB/BA decode/tail medians differed by
+1.07%/+1.32%, triggering four further AB/BA rounds. Confirmation medians:

| Shape | Before us | After us | Latency change |
| --- | ---: | ---: | ---: |
| rows8, H8192 | 4.444984 | 4.456789 | +0.27% |
| rows32768, H8192 | 3326.961266 | 3327.121188 | +0.005% |
| rows385, H8200 | 41.743977 | 41.721594 | -0.054% |

There is no stable >=1% regression across the confirmation rounds. Decode
is noisy: per-round changes include +8.40%, -1.34%, +0.11%, +0.42%; raw samples
and the initial result are preserved, not discarded or relabeled as speedup.
No kernel tuning is changed to chase this noise. Raw command/exit/hash/sample
records and initial/confirmation summaries are in `validation/performance-*`.

Artifact-only reproduction (one GPU mode at a time):

```sh
artifacts=/home/syk/worktrees/sgl-kernel-xpu/review-validation-artifacts/cleanup-20261016
bash "$artifacts/run_qk_residual_gpu.sh" muse_residual_norm accuracy
bash "$artifacts/run_qk_residual_gpu.sh" muse_residual_norm performance
bash "$artifacts/run_qk_residual_gpu.sh" muse_residual_norm performance confirmation 4
```

Commit only `benchmark/bench_fused_norm_residual.py`, this README, and the test
CLI cleanup. Preserve all original untracked review Python/C++/shell sources,
logs, libraries and artifact trees as historical evidence outside the commit.
