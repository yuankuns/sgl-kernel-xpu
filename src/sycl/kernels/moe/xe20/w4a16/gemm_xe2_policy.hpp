/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

namespace moe_w4a16 {
using namespace cute;

class xe_gemm_policy_base {
 public:
  using WGTile = Shape<_256, _256, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

  // Copy can be tuned for better performance. void => use make_block_2d_copy_*.
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  using GmemTiledCopyD = void;

  // 4-bit mainloop schedule. These defaults are what the hand-written policies
  // below were written against; w4a16_tile overrides them per tile.
  //   PrefetchDist    k-tiles of A and B prefetched ahead of the current one.
  //   StealChunk      output tiles claimed per work-stealing atomic.
  //   MainloopBarrier keep the per-k-tile work-group split barrier. Nothing in
  //                   the 4-bit mainloop is shared through SLM, so the barrier
  //                   is purely a scheduling device: it holds the subgroups in
  //                   lockstep so their prefetches stay timely. It is
  //                   load-bearing on a tile with several M-subgroups (dropping
  //                   it costs 34% compressed, 75% on real routing); on the
  //                   single-M-subgroup tiles, which already advance together,
  //                   it is worth at most 1.3% in either direction, so each of
  //                   those carries whichever sign it measured.
  static constexpr int PrefetchDist = 6;
  static constexpr int StealChunk = 1;
  static constexpr bool MainloopBarrier = true;
};

// Policy menu. The hand-written policies keep the *per-subgroup* tile at 32x32:
// the 4-bit mainloop needs its dequantised B fragment live alongside the
// accumulators, and a wider per-subgroup tile overruns the 256-GRF budget (a
// 256x256 / 8x4 variant compiles with ~190 spilled registers and runs at a
// third of the speed). So the work-group tile is scaled by adding or removing
// subgroups, not by widening them. The subgroup count per dimension must be a
// power of two -- cute's tile division rejects 3 and 6.
//
// Which one runs is decided per call by rows-per-expert and the GEMM shape; see
// select_w4a16_policy_id() in GroupGemmW4A16Xe20.cpp.

// A (BLK_M, BLK_N) work-group tile with its subgroup layout and its mainloop
// schedule. The subgroup tile is BLK_M/SgCountM x BLK_N/SgCountN.
//
// Measured on Arc Pro B60 over the whole tile space (GPT-OSS gemm1 geometry with
// the M padding removed, TFLOP/s), two rules decide the rate and both are
// properties of the *subgroup* tile, not of the work-group tile:
//
//   SgCountN such that the subgroup N tile is 16, i.e. exactly one dpas N-block
//   per subgroup. Every 32-wide variant loses 20-58%, whether or not it spills
//   (64x256_1x8 +58%, 64x512_1x16 +40%, 32x512_1x16 +44%, 32x256_1x8 +47%).
//
//   SgCountM == 1, i.e. BLK_M == SG_M. A second M-subgroup makes a ragged
//   expert tail leave whole subgroups out of bounds while they still run, and
//   what the rate is sensitive to is BLK_M (the per-work-group-tile B load and
//   dequantisation amortise over BLK_M rows): 64 rows measure 82.9, 32 rows
//   64.1, 16 rows 42.6. Above 64 the curve is flat (80 rows -0.7%, 96 rows
//   -2.5%) and the register cost is 2*SG_M GRFs, so 64 is the knee.
template <int BlkM, int BlkN, int SgCountM, int SgCountN, int PfDist = 2, int Steal = 1, bool Barrier = false>
class w4a16_tile : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<Int<BlkM>, Int<BlkN>, _32>;
  using SGLayout = Layout<Shape<Int<SgCountM>, Int<SgCountN>, _1>, Stride<Int<SgCountN>, _1, _0>>;

  static constexpr int PrefetchDist = PfDist;
  static constexpr int StealChunk = Steal;
  static constexpr bool MainloopBarrier = Barrier;
};

// The three small-avg_m tiles. All are 64-wide in N: at these row counts the
// 64-wide N tile beats a 256-wide one outright (the wide-N variant measures
// ~35 TFLOP/s against the 32-row tile's ~49).
//
// Their schedule was measured per tile, in the avg_m band each one serves, at a
// launch length that can resolve it: the 4-expert shapes those bands normally
// hit run 25-90 us and are host-launch-bound, so the sweep below holds
// rows-per-expert at 4/8/32 and widens the expert count to 128 instead
// (N=K=2880, median of 500, two interleaved reps agreeing within 0.1%, ms):
//
//   tile   pf3+none  pf3+barrier  pf6+none  pf6+barrier
//   8x64     1.2934      1.2992     1.2977       1.3024
//   16x64    1.3322      1.3492     1.3405       1.3646
//   32x64    1.5488      1.5331     1.5888       1.5707
//
// PrefetchDist 3 wins on all three (a deeper pipeline just holds more B in
// flight than these short-M tiles can consume). The barrier splits them: it
// costs 0.4%/1.3% on the 8- and 16-row tiles and pays 1.0% on the 32-row one.
// The same ordering appears, amplified 5-20x, when the tiles are forced onto
// avg_m=129/256 shapes, so it is a property of the tile and not of the shape.

// avg_m <= 4
using w4a16_policy_m_8_n_64 = w4a16_tile<8, 64, 1, 4, /*PfDist=*/3, /*Steal=*/1, /*Barrier=*/false>;

// avg_m <= 8
using w4a16_policy_m_16_n_64 = w4a16_tile<16, 64, 1, 4, /*PfDist=*/3, /*Steal=*/1, /*Barrier=*/false>;

// avg_m <= 32
using w4a16_policy_m_32_n_64 = w4a16_tile<32, 64, 1, 4, /*PfDist=*/3, /*Steal=*/1, /*Barrier=*/true>;

// Everything above avg_m = 32 runs a 64-row tile, and only BLK_N changes.
// w4a16_policy_m_64_n_128 keeps its name but not its subgroup layout: as a
// <_2,_4> tile its subgroups are 32 columns wide, which the SG_N rule above
// prices at +20-58%, so it is respelled 1x8 here and joined by a 256-wide
// sibling for long K.
//
// A 64-row tile bounds the M padding a ragged routing vector can cost at
// ceil(m/64)*64 -- 1.234x on the real GPT-OSS layer-0 histogram -- whereas the
// mean rows-per-expert the host can see says nothing about the tail: all three
// GPT-OSS prefill layers average 128 rows, but per-expert rows run 0..3573, so
// a 128-row tile that the mean scores as a perfect fit actually fills 0.61-0.82
// of itself. Both tiles below are single-M-subgroup, SG_N=16 tiles, so they can
// drop the mainloop barrier and run a short prefetch pipeline.
//
// GPT-OSS prefill GEMM1 (N=1472, K=2880): 82.9 TFLOP/s padding-free, 67.1 on
// the real TP=4 layer-0 routing vector.
using w4a16_policy_m_64_n_256 = w4a16_tile<64, 256, 1, 16, /*PfDist=*/2, /*Steal=*/1>;

// GPT-OSS prefill GEMM2 (N=2880, short K=736): 75.6 TFLOP/s padding-free, 60.5
// on the real TP=4 layer-0 routing vector. Half the N tile means twice the tiles
// per expert, so the work-stealing atomic is claimed four tiles at a time.
using w4a16_policy_m_64_n_128 = w4a16_tile<64, 128, 1, 8, /*PfDist=*/2, /*Steal=*/4>;

// Large avg_m, 32x32 subgroup tile. Kept as the measurement reference for the
// 128-row tile this menu used to select. On the real TP=4 layer-0 routing
// vectors it runs 49.3 TFLOP/s on GEMM1 and 45.1 on GEMM2, so the tile the
// selector picks instead beats it by 36% and 34%.
class w4a16_policy_m_128_n_128 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_128, _128, _32>;
  using SGLayout = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
};

}  // namespace moe_w4a16
