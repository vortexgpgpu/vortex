// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "constants.h"

// ---------------------------------------------------------------------------
// DTCU latency model parameters (see claude_doc "DTCU Latency Modeling").
// Shared by the compute datapath (dtcu.cpp) and the TMA engine (dtcu_tma.cpp).
// All are #ifndef-guarded so they can be overridden at build time via -D... .
//
// DTCU_MACS_PER_CYCLE: sustained multiply-accumulates per cycle of the DTCU
//   matrix array. Default 16 == one in-core TCU's raw throughput (NT=4), so the
//   DTCU's modeled advantage comes only from removing SIMT pipeline overhead
//   and NOT from also assuming a wider array (no double counting). Raise this to
//   model a physically wider unit.
// DTCU_COMPUTE_LATENCY: pipeline fill latency added per native tile (cycles).
// ---------------------------------------------------------------------------
#ifndef DTCU_MACS_PER_CYCLE
#define DTCU_MACS_PER_CYCLE 16 // In-core TCU also has 16 MACs/cycle (NT=4), so this models only SIMT overhead reduction, not a wider array.
#endif
#ifndef DTCU_COMPUTE_LATENCY
#define DTCU_COMPUTE_LATENCY 6
#endif

// Operand/accumulator buffer (scratchpad SRAM) model. Defaults borrow the in-core
// L1 dcache numbers (per postdoc suggestion):
//   DTCU_BUF_LATENCY = L1 dcache pipeline latency (1 cycle, sim/simx/socket.cpp).
//   DTCU_BUF_BW      = L1 dcache delivered 32-bit words/cycle
//                      = DCACHE_NUM_BANKS x (DCACHE_WORD_SIZE / 4)  (constants.h).
// (A scratchpad never misses, so we borrow L1's latency/bandwidth numbers rather
// than routing through the full cache_sim.)
#ifndef DTCU_BUF_LATENCY
#define DTCU_BUF_LATENCY 1
#endif
#ifndef DTCU_BUF_BW
#define DTCU_BUF_BW (DCACHE_NUM_BANKS * (DCACHE_WORD_SIZE / 4u))
#endif

// Address-generation (AGU) setup latency per cache-line-list build. Software (the
// descriptor) supplies the GMEM base pointers / dims; the DTCU's AGU only does the
// per-tile stride arithmetic (base + tile_idx*stride) and coalesces into cache
// lines -- modeled as a small per-tile setup. Per-element generation overlaps the
// memory requests (not a separate stall). Mirrors Virgo/Gemmini's controller (the
// SIMT core issues high-level commands; the matrix-unit HW computes tile addresses).
#ifndef DTCU_ADDRGEN_CYCLES
#define DTCU_ADDRGEN_CYCLES 3
#endif

// Max in-flight operand-prefetch requests (multiple-outstanding). Grounded in the
// L2 the DTCU connects to: L2_MSHR_SIZE outstanding misses. Bank-level concurrency
// (L2_NUM_BANKS) and contention are modeled by the L2 cache_sim automatically.
#ifndef DTCU_MAX_OUTSTANDING
#define DTCU_MAX_OUTSTANDING L2_MSHR_SIZE
#endif
