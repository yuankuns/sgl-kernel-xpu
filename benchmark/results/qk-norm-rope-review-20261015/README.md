# QK norm/RoPE Xe35 review: evidence and commit gate

**Commit scope:** this report and three directed benchmark/precision scripts.
Every new dispatch threshold discussed below is an **unshipped experiment**:
the 735-row wide entry and the 256 MiB, unaligned 300 MiB, and
alignment-gated 300 MiB BF16/128D sparse-row splits. The remote review branch
contains the original Xe35 implementation from `8598d82`, including its
hardware-wave guard and original conditional 100 MiB split, but none of these
new experimental thresholds. The worktree retains the alignment-gated 300 MiB
source experiment unstaged. Experiment binaries, patches, and raw logs remain
in the local evidence directory and are not Git commit contents.

## Paths and measurement

Muse Spark attention calls `sgl_kernel.fused_qk_norm_rope` from
`sglang_meta_common/models/muse_spark.py`. The checked XPU prefill setup uses
single-call eager mode; default `tc_piecewise` prefill graph is disabled.
B8/ISL4096/OSL16/TP1/EP1 gives 32768 QKV rows in this op. Main's equivalent
implementation is the SYCL fused kernel in `src/sycl/FusedQKNormRope.cpp`.
The other checked unfused path is PyTorch RMS norm plus existing SYCL cached
RoPE. No same-device equivalent Triton implementation was found.

Tests used GPU0 CRI `8086:674f` on `86-cri-d93-h`, Torch
`2.14.0a0+gitd35bb04`, BF16 contiguous QKV, int32 positions, interleaved
RoPE, `eps=1e-5`, `base=500000`. `bench_qk_norm_rope_boundary.py` times
one API call with XPU Events, restores QKV outside the interval, and prints
all raw microseconds, p50, MAD, p20 and p80. The paired old/candidate runs
used 10 warmups, 11 samples per provider/shape/pass, ABBA passes in two
opposite shape orders. `validate_qk_norm_rope_b8.py` checks selected rows
against BF16 reference (`rtol=8e-3`, `atol=4e-3`) and the untouched tail
exactly. GPU0 runs held host `flock /tmp/muse-gpu-bench.lock` throughout,
checked `xpu-smi stats -d 0` and `xpu-smi ps -d 0` idle on lock entry,
and cooled at least 2 seconds between shapes. XPU Events can include Python
submission gaps. Differences below 1% are ties; larger differences require
raw-sample and reverse-order support.

Reproduce with local scripts `run-split-restricted-final.sh`,
`run-split-middle.sh`, `run-split-main-reference.sh`,
`run-split-300-final.sh`, `run-split-refine.sh`,
`run-split-alt-heads-300.sh`, `run-split-stride-isolation.sh`, and
`run-split-aligned-final.sh` in the evidence directory below. All commands
and raw samples are retained in matching `.log` files.
Host builds used `flock /tmp/muse-op-build.lock` around
`cmake --build build --target sgl-ops-sycl-FusedQKNormRope -j1` in
`migration/sglang-muse-syk:rootfs`; see `wave-256-restricted-build.log`
and `split-300-build.log`. Main's same target was also built and run locally.
The archived wheel only carried package dependencies. No full wheel,
common_ops dependency build, or B64 serving test was run.

Library SHA-256: main
`57749179e38893c1710c0f382603f46524e2552e567355fa78e10a156e4f0d99`;
original HEAD `1b461d0174ef686fd7e904f3898a71698d5e1c96b3ee24fb07451c9ab89aaf46`;
256 MiB candidate `b865ac938908bd28f529bc3065759081553abd1829f06f4589416021718178ea`;
unaligned 300 MiB candidate `d52cdd31e25acc465704ec74b61cb2af7ff8339e37f6ef85e832f12456beac7a`;
alignment-gated 300 MiB candidate `631078ca19b13dd8506c6b7ade7e35033ecf4774e26ab74e2eae05308ce6c375`.

## Device time (microseconds)

Cells are pooled raw-sample **p50 / MAD**, with individual ordered passes in
the referenced logs. Main uses the same dtype/layout/GPU/event method;
its locally built path is much slower than the Xe35 branch at Muse B8.

| Shape | Main | Original HEAD | 256 MiB WIP | Unaligned 300 MiB WIP |
| --- | ---: | ---: | ---: | ---: |
| Muse BF16 64D Q128/K16/tail144, 32768 rows | 3960.260 / 5.260 | 1630.522 / 6.093 | 1623.229 / 6.771 | 1628.985 / 4.688 |
| BF16 128D Q32/K8/tail40, 15359 rows | 723.255 / 3.177 | 451.875 / 5.183 | 442.553 / 1.276 | 452.083 / 3.438 |
| Same layout, 15360 rows | 725.052 / 1.536 | 451.771 / 4.948 | 443.490 / 2.031 | 444.272 / 3.047 |
| Same layout, 16384 rows | 771.042 / 1.224 | 480.911 / 4.166 | 469.245 / 3.438 | 469.062 / 3.619 |

Muse main comes from `main-final.log` with 5 warmups/11 samples, while the
table's original/300 MiB cells come from `split-300-final.log` with 10
warmups/two passes. For a strict same-warmup Muse comparison,
`main-final.log` gives main 3960.260/5.260 and `original-final.log` gives
original HEAD 1631.302/4.583 (both 5 warmups/11 samples). The separate
four-pass `split-restricted-final.log` gives original/256 MiB
1630.052/5.625 versus 1623.229/6.771: a tie. The same-shape unfused
PyTorch/SYCL path was 7110.677 us p50 (`gated-unfused-bench.log`). These
are op times, not serving throughput. All 128D values in the table use
two reversed-shape passes in `split-300-final.log`.

For Q32/K8/128D, Q/K traffic is 20480 bytes per row. The 256 MiB
neighbors are 13107/13108: original HEAD pooled p50/MAD was
379.453/2.552 and 382.214/3.412 us; the 256 MiB candidate was
381.302/3.542 and 379.454/1.771 us (`split-restricted-final.log`). Both
are ties by the 1% rule. The coarse 8192 (160 MiB) and 16384 (320 MiB)
measurements bracketed a wide interval. Intermediate paired measurements
were:

| Traffic, rows | Original HEAD p50 / MAD | Split candidate p50 / MAD | Assessment |
| --- | ---: | ---: | --- |
| 240 MiB, 12288 | 354.270 / 3.594 | 353.646 / 4.271 | tie |
| 270 MiB, 13824 | 401.875 / 3.438 | 400.599 / 1.041 | tie |
| 275 MiB, 14080 | 408.672 / 3.776 | 405.261 / 2.475 | tie, 0.8% |
| 280 MiB, 14336 | 416.536 / 3.254 | 414.141 / 1.459 | tie, 0.6% |
| 285 MiB, 14592 | 427.135 / 2.916 | 421.615 / 2.188 | 1.3%, sample ranges overlap |
| 290 MiB, 14848 | 430.625 / 3.828 | 426.875 / 2.474 | tie, 0.9% |
| 295 MiB, 15104 | 440.338 / 4.635 | 435.860 / 1.589 | about 1%, variable |
| 300 MiB, 15360 | 450.260 / 4.010 | 443.385 / 1.588 | 1.5%, one pass ties |
| 320 MiB, 16384 | 479.479 / 3.334 | 472.917 / 3.542 | 1.4%, one pair ties |
| 480 MiB, 24576 | 732.240 / 6.484 | 704.140 / 3.073 | stable 3.8% |

The unaligned 300 MiB experiment activates the split at 15360 rows, and its
15359 left neighbor retains original HEAD speed. Yet the 256 MiB candidate
is 2.1% faster there in both provider orders. In the intermediate scan,
285 MiB exceeds 1% but 290 MiB returns to a tie. The data do not establish
a precise, monotonic eager crossover; neither 256 MiB nor 300 MiB is a
proved optimal general dispatch boundary. The former switches before a
stable gain, while the latter loses a stable WIP gain immediately to its
left. **No new sparse-row threshold is submitted as production code.**
The original 64D split and hardware-wave entry rule in HEAD remain the
reviewable production baseline.

## Alternate sparse-row layouts

The unaligned 300 MiB rule also selected two-chunk for Q33/K8/tail41 BF16
128D interleave, where each QKV row is 20,992 B. Its own 300 MiB traffic
neighbors are 14985/14986 rows. In two ABBA rounds with reversed shape
order (`split-alt-heads-300.log`), the right neighbor had original/unaligned
split p50/MAD **453.646/3.281 vs 486.797/3.412 us**, a stable 7.3%
regression. At 16384 rows, it was 496.485/3.984 vs 529.922/3.281 us.
Selected-row reference precision and untouched-tail checks passed for both
libraries at 14985, 14986 and 16384.

`split-stride-isolation.log` changed only the tail in Q32/K8/tail41:
its 20,736 B row stride made the 15360-row split regress from
469.479/3.047 to 491.224/3.151 us. Conversely, Q33/K8/tail55 restored
a 24,576 B row stride (six 4 KiB units) with the odd Q/K count; at 14986
rows, original/split were 479.505/2.526 vs 477.917/2.422 us, a tie.
This isolates 4 KiB row-stride alignment as a useful layout gate without
specializing fixed Muse head counts.

The **current unstaged experiment** applies the new BF16/128D split only
when `row_bytes % 4096 == 0`. `split-aligned-final.log` compares its locally
built `.so` with original HEAD under reversed provider/shape orders:

| BF16 layout and rows | Original HEAD p50 / MAD | Aligned candidate p50 / MAD | Assessment |
| --- | ---: | ---: | --- |
| Q32/K8/tail40, 16384; row 20,480 B | 482.005 / 3.541 | 469.870 / 2.813 | 2.5% gain |
| Q32/K8/tail41, 15360; row 20,736 B | 467.110 / 2.526 | 469.167 / 1.771 | tie |
| Q33/K8/tail41, 14986; row 20,992 B | 454.167 / 5.286 | 455.938 / 2.526 | tie |
| Q33/K8/tail55, 14986; row 24,576 B | 479.349 / 3.020 | 481.745 / 1.928 | tie |
| Muse Q128/K16/tail144, 32768 | 1624.557 / 9.947 | 1628.984 / 8.463 | tie |

The alignment gate removes the measured 4–7% regressions in the two
neighboring misaligned layouts while retaining the aligned 128D gain and
Muse B8 performance. It does **not** resolve the nonmonotonic traffic
crossover or the 300 MiB left-neighbor loss against the earlier 256 MiB
candidate, so the source remains WIP. The alignment-gated binary passed
the selected-row reference and exact untouched-tail checks at
Q32/K8/tail41 15360, Q33/K8/tail41 14985/14986,
Q32/K8/tail40 15360, Q33/K8/tail55 14986, and Muse 32768
(`split-aligned-final.log`).

## Precision, limits, and local evidence

Selected-row precision and exact tail preservation passed for main,
original HEAD, 256 MiB and 300 MiB libraries at 128D 15359/15360
(`split-300-final.log`); original/256 MiB also passed at 13107/13108,
and original/256 MiB at 16384. The 300 MiB library passed Muse 32768.
The extra split experiment is restricted to BF16, interleaved, 128D,
sparse rows with 4 KiB aligned row strides; it does not add a new path
for FP16, neox, 256D or non-sparse rows. Q33/K8 was tested at both
aligned and misaligned tail widths; other head counts remain unmeasured.

The earlier `kMinWideSubgroups=26460` Muse 734/735 entrance is also
unverified. `entry-small-eager.log` changes ranking with process/order;
`entry-735-final.log` has a 734-row 252.5 us outlier on the main launch.
`small-forced-graph.log` favors wide at 16–512 rows but shows a full
second-process 250 us eager plateau at 16/32; graph 1–8 results are mixed.
The 896 proposal regressed its 895-row left neighbor. Graph replay is a
separate timing mode, not an eager speedup. Keep the original Xe35
implementation for its repeatable Muse B8 operator gain. The new wide-entry
and sparse-row thresholds in this report remain unshipped until eager entry
and sparse-row boundaries are established.

Local persistent evidence directory:
`/home/syk/worktrees/sgl-kernel-xpu/muse-spark-qk-norm-rope-main-20261014/benchmark/results/qk-norm-rope-review-20261015/`.
All raw `.log` files, `raw-logs.tar.gz`, local run scripts, binaries,
`README-entry-735-experiment.md`, `README-split-256.md`, and the patches
`experimental-entry-735-before-wave-revert.patch` (SHA-256
`803e0b9d8e0a9aa1efce9d7447cb996c0e65ccd43829f64c105ce2bb1a43b9c1`),
`split-256-before-300.patch`, `split-300-uncommitted.patch`, and
`split-300-aligned-uncommitted.patch` are local
persistent evidence, **not included in this Git commit**. Worktree WIP and
all original logs remain intact.
# Current validation entry points (2026-10-16)

For the cleanup baseline, explicit benchmark CLI, renamed test nodes, and
no-sleep validation commands, see
[`../../README_cleanup_20261016.md`](../../README_cleanup_20261016.md).
The results and untracked scripts below are historical evidence; their old
environment-driven invocations must not be used with the current scripts.
