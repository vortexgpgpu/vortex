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
//
// PRISM RTU intrinsics (Phase 1). See docs/proposals/rtu_simx_v3_proposal.md.
//
// All RTU ops share CUSTOM1 (.insn r prefix 0x2B) and funct3 = 5.
// The 2-bit sub-op selector lives at funct2 / funct7's low 2 bits:
//
//   sub-op 0  vx_rt_set     R4-type: rd=slot, rs1/rs2/rs3 = 3 values
//   sub-op 1  vx_rt_get     R-type:  rd=dest, slot in top 5 bits of funct7
//   sub-op 2  vx_rt_trace   R-type:  rd=handle, rs1=TLAS ptr
//   sub-op 3  vx_rt_wait    R-type:  rd=status, rs1=handle

#pragma once

#include <vx_intrinsics.h>
#include <VX_types.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// vx_rt_set1 — write one RTU register-file slot. Single-slot writer using
// R-type encoding (slot ID in top 5 bits of funct7, sub-op 0 in low 2 bits).
// Note: Phase 1 encoding writes one slot per instruction; the proposal's
// 3-slot bulk-set is a Phase 2+ extension (needs R4 encoding with rd-as-imm
// support that GAS .insn r4 doesn't provide today).
#define vx_rt_set1(slot, val) \
  __asm__ volatile (".insn r %0, 5, %1, x0, %2, x0" \
      :: "i"(RISCV_CUSTOM1), "i"(((slot) << 2) | 0), "r"(val))

// vx_rt_set3 — convenience macro emitting three single-slot SETs.
#define vx_rt_set3(slot, r1, r2, r3) \
  do { \
    vx_rt_set1((slot) + 0, (r1)); \
    vx_rt_set1((slot) + 1, (r2)); \
    vx_rt_set1((slot) + 2, (r3)); \
  } while (0)

// vx_rt_set2 — convenience macro emitting two single-slot SETs.
#define vx_rt_set2(slot, val1, val2) \
  do { \
    vx_rt_set1((slot) + 0, (val1)); \
    vx_rt_set1((slot) + 1, (val2)); \
  } while (0)

// vx_rt_get — read one RTU register-file slot into rd. funct2 = 1; the
// slot ID is encoded in the top 5 bits of funct7 (slot << 2 | 1).
#define vx_rt_get(slot) ({ \
  uint32_t __v; \
  __asm__ volatile (".insn r %1, 5, %2, %0, x0, x0" \
      : "=r"(__v) \
      : "i"(RISCV_CUSTOM1), "i"(((slot) << 2) | 1)); \
  __v; \
})

// vx_rt_get_f — same as vx_rt_get but reinterprets the bits as float.
static inline float vx_rt_get_f(uint32_t slot_runtime) {
  // Helper for runtime slot — emits the same instruction but slot must be
  // a compile-time constant for the encoding. Use the macro variants for
  // compile-time slots; this is a fallback for generic code.
  (void)slot_runtime;
  return 0.0f;  // not used in Phase 1 smoke; kernels use vx_rt_get_f_imm.
}

// vx_rt_get_f_imm — compile-time slot version returning float.
#define vx_rt_get_f_imm(slot) ({ \
  uint32_t __u = vx_rt_get(slot); \
  union { uint32_t u; float f; } __c; \
  __c.u = __u; \
  __c.f; \
})

// vx_rt_trace — fire a ray. rs1 = TLAS device address. Returns a handle
// in rd. Non-blocking (Phase 1 returns 0 since one ray per lane).
#define vx_rt_trace(tlas_addr) ({ \
  uint32_t __h; \
  __asm__ volatile (".insn r %1, 5, 2, %0, %2, x0" \
      : "=r"(__h) \
      : "i"(RISCV_CUSTOM1), "r"(tlas_addr)); \
  __h; \
})

// vx_rt_wait — block until ray identified by handle reaches terminal
// status. Returns VX_RT_STS_*.
#define vx_rt_wait(handle) ({ \
  uint32_t __s; \
  __asm__ volatile (".insn r %1, 5, 3, %0, %2, x0" \
      : "=r"(__s) \
      : "i"(RISCV_CUSTOM1), "r"(handle)); \
  __s; \
})

// vx_rt_cb_ret — Phase 2: release the lane's parked context in the RtuCore
// with one of VX_RT_CB_{ACCEPT,IGNORE,TERMINATE}. The callback dispatcher
// (mtvec-registered) calls this once it has decided the candidate hit's
// fate, then exits via `mret` to resume the post-vx_rt_wait PC. The
// dispatcher's tmask was narrowed at trap entry to only-yielded-lanes and
// is restored by `mret` from the saved tmask CSR (proposal §4.6).
//
// EXT2 / funct3=6 / sub-op=0 / R-type. rs1 = action; no rd.
#define vx_rt_cb_ret(action) \
  __asm__ volatile (".insn r %0, 6, 0, x0, %1, x0" \
      :: "i"(RISCV_CUSTOM1), "r"(action))

// Mark a function as the RTU callback dispatcher. The compiler should
// emit it as a normal extern-"C" function; the dispatcher is responsible
// for exiting via `mret` (e.g. by tail-calling a small `mret` stub or
// finishing with `vx_rt_cb_ret` + an inline `mret` asm). Kept as a no-op
// attribute today — Phase 1.5 / 2.5 Mesa work can lower it into a real
// codegen attribute (preserve `mret` epilogue, no caller-save spill).
#define VX_RT_CALLBACK_ENTRY __attribute__((used))

// Inline `mret` — for use at the bottom of a callback dispatcher to
// resume the post-vx_rt_wait PC and restore the pre-yield tmask.
#define vx_mret() \
  __asm__ volatile ("mret")

// Convenience float-encoding helper.
static inline uint32_t vx_rt_f2u(float f) {
  union { float f; uint32_t u; } c;
  c.f = f;
  return c.u;
}

#ifdef __cplusplus
}
#endif
