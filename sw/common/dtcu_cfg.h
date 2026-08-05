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

// DTCU host/device/simulator ABI: the GEMM descriptor layout and the native-tile
// geometry the engine derives from it. This is the single source of truth for all
// three builds; sw/common is the only directory on every include path
// (host CXXFLAGS, kernel VX_CFLAGS, and sim/simx CXXFLAGS all carry -I sw/common).
//
// It deliberately does NOT live in sw/kernel/include/vx_dtensor.h: that header pulls
// in vx_intrinsics.h, which is full of RISC-V `.insn` inline asm and cannot compile
// for x86. That is why the host tests used to hand-mirror this struct. vx_dtensor.h
// now includes this file and contributes only the dtensor_start/poll intrinsics --
// the same split vx_tensor.h (device intrinsics) / tensor_cfg.h (shared config) uses
// for the in-core TCU.
//
// Kept free of C++ constructs so the kernel can include it from C. The compile-time
// traits wrapper (dtcu_config_t) lives in tensor_cfg.h, which is C++-only.

#include <stdint.h>

// ---------------------------------------------------------------------------
// Native-tile geometry.
//
// The operand/accumulator SRAM is a FIXED-size physical buffer sized for the largest
// legal tile (like Hopper's fixed-capacity SMEM); a smaller tile (tile_n <
// DTCU_TILE_N_MAX) uses only the leading prefix.
//
// The three axes are NOT symmetric, and the asymmetry is the hardware contract:
//   tile_m : build-time fixed  (DTCU_TILE_M)
//   tile_n : runtime, descriptor-driven (shape_n_size * DTCU_TILE_N_GRAN)
//   tile_k : build-time fixed in 32-bit WORDS (DTCU_TILE_K_WORDS); the element count
//            depends on the input format, hence dtcu_tile_k() below.
//
// K is counted in words because the buffer is sized in bytes: one A row is
// DTCU_TILE_K_WORDS 32-bit words (32 B at the default 8), so a narrower input type
// fits proportionally more K elements in the same SRAM. This mirrors the in-core
// TCU's i_ratio (tensor_cfg.h: i_ratio = XB / sizeof(It), tileK = xtileK * i_ratio).
// ---------------------------------------------------------------------------
// There are TWO engines and their tiles differ, because the tile is sized by the
// capacity of whatever the engine writes D into:
//
//   CLUSTER  D -> L2 (1 MB)          64 x (16..128)   D tile up to 32 KB
//   SOCKET   D -> the socket's L1    32 x 16          D tile 2 KB
//
// The socket tile is small on purpose. Pipelining needs several D tiles resident at
// once (one being written, one being consumed, slack for a lagging consumer), and at
// SOCKET_SIZE=4 one 32 KB dcache serves four cores. 2 KB tiles leave four per consumer;
// 8 KB tiles would leave one, i.e. no double-buffering. It also lands between the
// in-core TCU's 16x16 fragment and the cluster tile, which is what makes the two
// engines a meaningful comparison rather than two names for the same thing.
//
// A consequence worth stating: with TILE_N_MAX == TILE_N_GRAN the socket engine has
// exactly one legal shape_n_size. Tile-shape freedom is what it trades for locality.
#ifndef DTCU_CLUSTER_TILE_M
#define DTCU_CLUSTER_TILE_M 64
#endif
#ifndef DTCU_CLUSTER_TILE_N_MAX
#define DTCU_CLUSTER_TILE_N_MAX 128
#endif
#ifndef DTCU_SOCKET_TILE_M
#define DTCU_SOCKET_TILE_M 32
#endif
#ifndef DTCU_SOCKET_TILE_N_MAX
#define DTCU_SOCKET_TILE_N_MAX 16
#endif

// Shared by both engines: the tile-N quantum and the K depth in 32-bit words.
#ifndef DTCU_TILE_N_GRAN
#define DTCU_TILE_N_GRAN 16
#endif
#ifndef DTCU_TILE_K_WORDS
#define DTCU_TILE_K_WORDS 8
#endif

// Which engine a descriptor is bound for. Chosen by WHICH START INSTRUCTION issued it
// (RISCV_CUSTOM2 funct3), not by a descriptor field -- the descriptor stays identical
// for both, so the same one can be replayed on either engine.
#define DTCU_ENGINE_SOCKET  0
#define DTCU_ENGINE_CLUSTER 1

// Legacy single-engine spellings, kept so code that predates the split still reads.
#ifndef DTCU_TILE_M
#define DTCU_TILE_M DTCU_CLUSTER_TILE_M
#endif
#ifndef DTCU_TILE_N_MAX
#define DTCU_TILE_N_MAX DTCU_CLUSTER_TILE_N_MAX
#endif

// The four helpers below are the ONLY place the geometry formulas live: the
// simulator calls them at runtime (it learns fmt_s only when a descriptor arrives)
// and tensor_cfg.h's dtcu_config_t calls the same ones at compile time. Under C++
// they are constexpr so both uses share one definition instead of two that happen
// to agree; under C they degrade to static inline for the device kernel.
#ifdef __cplusplus
#define DTCU_CFG_FN constexpr // implicitly inline
#else
#define DTCU_CFG_FN static inline
#endif

// Native tile K in ELEMENTS for an input element size of in_sz bytes.
// in_sz must divide 4 (the engine rejects sub-byte and non-power-of-two formats).
DTCU_CFG_FN uint32_t dtcu_tile_k(uint32_t in_sz) {
  return DTCU_TILE_K_WORDS * (4 / in_sz);
}

// Native tile N in elements for a descriptor's shape_n_size selector. Engine-agnostic:
// the quantum is shared, only the upper bound differs.
DTCU_CFG_FN uint32_t dtcu_tile_n(uint8_t shape_n_size) {
  return (uint32_t)shape_n_size * DTCU_TILE_N_GRAN;
}

// Inverse of dtcu_tile_n(): the selector that requests tile_n elements.
DTCU_CFG_FN uint8_t dtcu_shape_n_size(uint32_t tile_n) {
  return (uint8_t)(tile_n / DTCU_TILE_N_GRAN);
}

// Per-engine geometry accessors. Taking the engine as an argument keeps ONE formula
// per quantity; the engine only selects a bound.
DTCU_CFG_FN uint32_t dtcu_tile_m_of(int engine) {
  return (engine == DTCU_ENGINE_CLUSTER) ? DTCU_CLUSTER_TILE_M : DTCU_SOCKET_TILE_M;
}
DTCU_CFG_FN uint32_t dtcu_tile_n_max_of(int engine) {
  return (engine == DTCU_ENGINE_CLUSTER) ? DTCU_CLUSTER_TILE_N_MAX : DTCU_SOCKET_TILE_N_MAX;
}

// Whether tile_n is legal for *this* engine (granularity and its own capacity). The
// engine checks this when a descriptor arrives, so asking the socket engine for a
// cluster-sized tile is rejected rather than silently truncated.
DTCU_CFG_FN int dtcu_tile_n_valid_of(int engine, uint32_t tile_n) {
  return tile_n >= DTCU_TILE_N_GRAN
      && tile_n <= dtcu_tile_n_max_of(engine)
      && (tile_n % DTCU_TILE_N_GRAN) == 0;
}

// Legacy single-engine spelling (cluster bounds).
DTCU_CFG_FN int dtcu_tile_n_valid(uint32_t tile_n) {
  return dtcu_tile_n_valid_of(DTCU_ENGINE_CLUSTER, tile_n);
}

// ---------------------------------------------------------------------------
// Descriptor.
//
// The DTCU model reads these fields straight out of device memory, so the layout is
// an ABI: reordering or resizing a field feeds the engine garbage with no diagnostic
// (see the NT=32 story in cgo27_motivation/260718_moti_RFC.md). The static_assert
// below is the guard; it is C++-only, so C includers get the layout without it.
// ---------------------------------------------------------------------------

// Descriptor flag bits (dtensor_desc_t.flags).
#define DTENSOR_FLAG_ZERO_ACC 0x1 // zero-accumulate (no C preload)
#define DTENSOR_FLAG_NO_TMA   0x2 // disable TMA overlap: blocking loads/stores (timing only)
#define DTENSOR_FLAG_ALL      (DTENSOR_FLAG_ZERO_ACC | DTENSOR_FLAG_NO_TMA)

// 64-byte GEMM descriptor.
typedef struct {
  uint64_t ptrA;
  uint64_t ptrB;
  uint64_t ptrC;
  uint64_t ptrD;
  uint32_t ldmA; // leading dims in elements (not bytes)
  uint32_t ldmB;
  uint32_t ldmC;
  uint32_t ldmD;
  uint16_t M;
  uint16_t N;
  uint16_t K;
  uint8_t  fmt_s;        // source (A/B) element format id (tensor_cfg.h)
  uint8_t  fmt_d;        // dest (C/D) element format id
  uint8_t  flags;        // DTENSOR_FLAG_* bits above
  uint8_t  shape_n_size; // tile-N selector: tile_n = shape_n_size * DTCU_TILE_N_GRAN
  uint16_t shape_policy; // must be 0
  // Completion flag, written by the ENGINE (everything else is input). Software zeroes
  // it before submitting; the engine sets it to 1 only after every D line has been
  // acknowledged, so observing 1 implies the output is visible. It lives here rather
  // than in engine state because the consumer is typically a different core than the
  // submitter, and only the descriptor is addressable by both. Read it with
  // dtensor_check() -- a plain load would cache a stale copy in the reader's L1.
  uint32_t done;
} dtensor_desc_t;

// Byte offset of the completion flag, for the atomic accessor in vx_dtensor.h.
#define DTENSOR_DONE_OFFSET 60

#ifdef __cplusplus
static_assert(sizeof(dtensor_desc_t) == 64, "dtensor_desc_t must be 64 bytes");
#endif
