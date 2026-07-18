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

#ifndef __VX_DTENSOR_H__
#define __VX_DTENSOR_H__

#include <stdint.h>
#include "vx_intrinsics.h" // RISCV_CUSTOM2

// -----------------------------------------------------------------------------
// Disaggregated Tensor Core (DTCU) control intrinsics.
//
// Encoding: opcode = RISCV_CUSTOM2 (EXT3, 0x5B), funct3 = 1 (start) / 2 (poll),
// funct7 = 0 (ignored by the decoder). This is a SEPARATE opcode space from the
// in-core TCU's RISCV_CUSTOM0/funct3=2 (WMMA/WGMMA/TCU_LD) group, so the DTCU and
// the in-core TCU can coexist in one binary — required by dtcu_compare.
//
// Software model: a single leader thread issues dtensor_start(&desc) once, then spins
// on dtensor_poll() until it returns non-zero. The DTCU engine (cluster-level) runs
// the whole tiled GEMM autonomously from the 64-byte descriptor.
// -----------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

// Descriptor flag bits (dtensor_desc_t.flags). Must match sim/simx/dtcu/dtcu.h.
#define DTENSOR_FLAG_ZERO_ACC 0x1 // zero-accumulate (no C preload)
#define DTENSOR_FLAG_NO_TMA   0x2 // disable TMA overlap: blocking loads/stores (timing only)

// 64-byte GEMM descriptor (must match sim/simx/dtcu/dtcu.h Dtcu::Desc exactly).
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
  uint8_t  shape_n_size; // tile-N selector: tile_n = shape_n_size * 16 (1..8)
  uint16_t shape_policy; // must be 0
  uint32_t reserved2;
} dtensor_desc_t;

// Fire the GEMM described by *desc_addr* (a device pointer to a dtensor_desc_t).
static inline void dtensor_start(uint64_t desc_addr) {
  __asm__ volatile (".insn r %[insn], 1, 0, x0, %[addr], x0"
    :
    : [insn] "i"(RISCV_CUSTOM2), [addr] "r"(desc_addr)
    : "memory");
}

// Returns non-zero once the in-flight GEMM has completed (and its D store drained).
static inline uint32_t dtensor_poll(void) {
  uint32_t done;
  __asm__ volatile (".insn r %[insn], 2, 0, %0, x0, x0"
    : "=r"(done)
    : [insn] "i"(RISCV_CUSTOM2)
    : "memory");
  return done;
}

#ifdef __cplusplus
}
#endif

#endif // __VX_DTENSOR_H__
