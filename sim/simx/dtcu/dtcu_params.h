// Copyright © 2019-2026
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

// Native-tile geometry + descriptor ABI, shared with the host and the device kernel.
#include "dtcu_cfg.h"

// v3.0 note: HW config arrives as -DVX_CFG_* flags (TOML -> ci/gen_config.py);
// there is no legacy "constants.h". The pre-v3 names used here were remapped:
//   DCACHE_NUM_BANKS -> VX_CFG_DCACHE_NUM_BANKS
//   DCACHE_WORD_SIZE -> (VX_CFG_XLEN / 8)   (dcache delivers one XLEN word/bank/cycle)
//   L2_MSHR_SIZE     -> VX_CFG_L2_MSHR_SIZE

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

// Operand-buffer fill (TMA write into the operand SRAM) latency model. A scratchpad
// never misses, so we borrow the L1 dcache pipeline latency (1 cycle) rather than
// routing through the full cache model. (Fill BANDWIDTH is not a separate parameter:
// fill and operand-read share the same banked scratchpad, so the fill rate is just
// DTCU_SMEM_BANKS words/cycle -- see buffer_fill_cycles_() in dtcu_tma.cpp.)
#ifndef DTCU_BUF_LATENCY
#define DTCU_BUF_LATENCY 1
#endif

// Accumulator SRAM model -- a SEPARATE physical SRAM from the operand scratchpad
// (mirrors Gemmini's distinct sp_* vs acc_* params and Virgo's spad-in-SHM vs private
// accumulator; Hopper keeps the accumulator in registers). It is matrix-unit-private
// and accessed as a sequential read-modify-write (accum[m*tile_n + n]), so -- unlike
// the operand B-column read -- it has NO bank-stride conflict: a plain BW model, no
// swizzle. DTCU_ACC_BANKS = fp32 words/cycle (conflict-free, so banks == delivered BW).
// Lab decision: fixed 2-bank physical SRAM (matches the DTCU_SMEM_BANKS=2 scratchpad),
// not array-width-scaled. Override -DDTCU_ACC_BANKS=... to model a wider accumulator.
#ifndef DTCU_ACC_BANKS
#define DTCU_ACC_BANKS 2
#endif
#ifndef DTCU_ACC_LATENCY
#define DTCU_ACC_LATENCY DTCU_BUF_LATENCY
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
// and contention are modeled by the L2 cache automatically.
#ifndef DTCU_MAX_OUTSTANDING
#define DTCU_MAX_OUTSTANDING VX_CFG_L2_MSHR_SIZE
#endif

// Native-tile geometry (DTCU_TILE_M / DTCU_TILE_N_MAX / DTCU_TILE_N_GRAN /
// DTCU_TILE_K_WORDS) is NOT defined here: it is part of the host/device/simulator
// contract and lives in sw/common/dtcu_cfg.h, included above. Only timing and
// microarchitecture parameters -- things the software side cannot observe -- belong
// in this file.

// Descriptor queue depth. The engine is asynchronous: dtensor_start hands over a
// descriptor and retires, so software must be able to keep more than one GEMM in
// flight or the issuing warp is forced to wait on the previous one. A pending entry is
// just a descriptor address, so depth is nearly free; it only has to be large enough
// that no core sharing the engine can be starved.
// The sharer set of each engine, which is what the depth has to cover:
//   cluster engine: every core in the cluster  -> VX_CFG_NUM_CORES (a per-CLUSTER count;
//                   constants.h derives NUM_SOCKETS from it as NUM_CORES / SOCKET_SIZE)
//   socket engine : the cores of one socket    -> VX_CFG_SOCKET_SIZE
//
// These two were previously swapped: the cluster engine, shared by NUM_CORES cores, was
// given NUM_SOCKETS*2 entries while the socket engine, shared by SOCKET_SIZE cores, was
// given NUM_CORES*2. At SOCKET_SIZE > 2 that made the cluster queue SMALLER than its own
// sharer set, which is exactly the starvation the comment above says depth exists to
// prevent -- and acceptance is not fair when it overflows: SimObjects tick in creation
// order, so the lowest-index core wins the last slot every time.
//
// Times two so a core can have one GEMM running and one queued behind it; +1 more is
// live in the engine itself (start() takes the non-busy path), giving 2N+1 in flight.
#ifndef DTCU_CLUSTER_QUEUE_DEPTH
#define DTCU_CLUSTER_QUEUE_DEPTH (VX_CFG_NUM_CORES * 2)
#endif
#ifndef DTCU_SOCKET_QUEUE_DEPTH
#define DTCU_SOCKET_QUEUE_DEPTH (VX_CFG_SOCKET_SIZE * 2)
#endif
// Single-engine builds (before the socket/cluster split) use the cluster depth.
#ifndef DTCU_QUEUE_DEPTH
#define DTCU_QUEUE_DEPTH DTCU_CLUSTER_QUEUE_DEPTH
#endif

// Operand SRAM bank count. The A/B operands share one banked SRAM; a bank delivers
// 1 word/cycle (the Vortex MemCrossBar rule), so a K-word operand vector takes
// (max words landing on one bank) cycles -- conflict-free = 1. Must be a power of two.
// Sweep parameter (RTL/FPGA pins the real value); banking matters only when the
// operand read becomes the bound (wider array / fewer banks).
#ifndef DTCU_SMEM_BANKS
#define DTCU_SMEM_BANKS 2
#endif

// Operand-SRAM swizzle toggle. 0 = naive interleave (a B column, stride DTCU_TILE_N_MAX,
// aliases to one bank -> conflict). 1 = XOR-permute the bank select (fold the high
// row/K bits in) so the column spreads across banks -- the Hopper-TMA swizzle idea.
// Applied identically at fill+read in HW (functional values unchanged); in this timing
// model it only changes the bank distribution. Rebuild with -DDTCU_SWIZZLE=1 to compare.
#ifndef DTCU_SWIZZLE
#define DTCU_SWIZZLE 0
#endif
