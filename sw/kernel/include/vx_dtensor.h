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
#include <dtcu_cfg.h>      // dtensor_desc_t + DTENSOR_FLAG_* (shared with host/simulator)
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

// dtensor_desc_t, DTENSOR_FLAG_*, and the native-tile geometry come from
// <dtcu_cfg.h>: the descriptor layout is an ABI the host, this kernel header, and
// the simulator all have to agree on, so it has exactly one definition. This header
// contributes only the intrinsics, which is why the config cannot live here -- the
// vx_intrinsics.h it includes is RISC-V inline asm and will not compile for x86.

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
