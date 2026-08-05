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

// Submit the GEMM described by *desc_addr* (a device pointer to a dtensor_desc_t).
//
// Non-blocking: the engine runs the GEMM after this instruction retires. Returns a
// 1-based ticket, or 0 when the descriptor queue is full -- the caller MUST check,
// because a rejected submission is not queued anywhere and its GEMM never runs.
//
// Before submitting, zero the descriptor's `done` field: the engine sets it once the
// output is visible, and a consumer distinguishes "finished" from "not started" by
// that transition alone.
static inline uint32_t dtensor_start(uint64_t desc_addr) {
  uint32_t ticket;
  __asm__ volatile (".insn r %[insn], 1, 0, %0, %[addr], x0"
    : "=r"(ticket)
    : [insn] "i"(RISCV_CUSTOM2), [addr] "r"(desc_addr)
    : "memory");
  return ticket;
}

// Non-zero once the GEMM described at *desc_addr* has finished AND its output is
// visible in memory.
//
// This is an ordinary atomic load, not a DTCU instruction, and that is the point:
// completion has to be observable by cores that never issued the descriptor, so it
// lives in memory rather than in engine state. The atomic matters -- a plain load
// installs the line in this core's L1 and every later read returns that stale copy,
// whereas an atomic access invalidates the local line and resolves at the last-level
// cache where the engine's write landed.
// It must be a read-modify-write, not an atomic *load*: RISC-V lowers an acquire load
// to `lw` plus a fence, which still hits the core's own cached copy. Only an AMO takes
// the cache's AmoProbe path, which invalidates that copy and resolves at the LLC.
// OR-with-zero is the read-only RMW -- it returns the value and changes nothing.
static inline uint32_t dtensor_check(uint64_t desc_addr) {
  uint32_t* flag = (uint32_t*)(uintptr_t)(desc_addr + DTENSOR_DONE_OFFSET);
  return __atomic_fetch_or(flag, 0u, __ATOMIC_ACQUIRE);
}

#ifdef __cplusplus
}
#endif

#endif // __VX_DTENSOR_H__
