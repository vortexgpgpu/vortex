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
#ifndef DTCU_TILE_M
#define DTCU_TILE_M 64
#endif
#ifndef DTCU_TILE_N_MAX
#define DTCU_TILE_N_MAX 128
#endif
#ifndef DTCU_TILE_N_GRAN
#define DTCU_TILE_N_GRAN 16
#endif
#ifndef DTCU_TILE_K_WORDS
#define DTCU_TILE_K_WORDS 8
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

// Native tile N in elements for a descriptor's shape_n_size selector.
DTCU_CFG_FN uint32_t dtcu_tile_n(uint8_t shape_n_size) {
  return (uint32_t)shape_n_size * DTCU_TILE_N_GRAN;
}

// Inverse of dtcu_tile_n(): the selector that requests tile_n elements.
DTCU_CFG_FN uint8_t dtcu_shape_n_size(uint32_t tile_n) {
  return (uint8_t)(tile_n / DTCU_TILE_N_GRAN);
}

// Whether tile_n is a legal native tile-N (granularity and capacity).
DTCU_CFG_FN int dtcu_tile_n_valid(uint32_t tile_n) {
  return tile_n >= DTCU_TILE_N_GRAN
      && tile_n <= DTCU_TILE_N_MAX
      && (tile_n % DTCU_TILE_N_GRAN) == 0;
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
  uint32_t reserved2;
} dtensor_desc_t;

#ifdef __cplusplus
static_assert(sizeof(dtensor_desc_t) == 64, "dtensor_desc_t must be 64 bytes");
#endif
