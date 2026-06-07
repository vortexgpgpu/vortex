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
// PRISM RTU intrinsics (Phase 1). See docs/proposals/rtu_simx_proposal.md.
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
// slot ID is encoded in the top 5 bits of funct7 (slot << 2 | 1). rs1 =
// x0 (no scoreboard dep). Use inside trap-context dispatchers (CHS / AHS
// / IS / MISS) where the regfile is already populated via
// apply_callback_payload(); ordinary kernel code AFTER vx_rt_wait must
// use vx_rt_get_after below to chain the scoreboard dep onto WAIT's rd.
#define vx_rt_get(slot) ({ \
  uint32_t __v; \
  __asm__ volatile (".insn r %1, 5, %2, %0, x0, x0" \
      : "=r"(__v) \
      : "i"(RISCV_CUSTOM1), "i"(((slot) << 2) | 1)); \
  __v; \
})

// §8.6 vx_rt_get_after — same op as vx_rt_get, but takes a
// "wait_status" register as rs1 to force a scoreboard dependency on
// the vx_rt_wait's rd. The SFU ignores rs1's value (the encoded slot
// still lives in funct7); rs1 only exists so the scoreboard stalls
// this read until vx_rt_wait actually writes back its status word.
//
// vx_rt_wait does NOT writeback until the matching TERMINAL drains
// (the trace is parked in RtuUnit::wait_parked_), so vx_rt_get_after
// is guaranteed to read post-TERMINAL hit attrs — even on the
// post-mret path coming out of an AHS/CHS/IS/MISS dispatcher.
//
// Kernel idiom:
//   uint32_t h   = vx_rt_trace(scene);
//   uint32_t sts = vx_rt_wait(h);
//   float    t   = vx_rt_get_f_imm_after(VX_RT_HIT_T, sts);
// The "memory" clobber makes this getter a compiler ordering barrier so a
// callback-written memory load placed after it in source is not hoisted above
// it; combined with in-order issue (the getter stalls on wait_status until the
// trace retires) this guarantees such a load observes the dispatcher's stores.
#define vx_rt_get_after(slot, wait_status) ({ \
  uint32_t __v; \
  __asm__ volatile (".insn r %1, 5, %2, %0, %3, x0" \
      : "=r"(__v) \
      : "i"(RISCV_CUSTOM1), "i"(((slot) << 2) | 1), "r"(wait_status) \
      : "memory"); \
  __v; \
})

// §8.6 float-typed scoreboard-safe getter (compile-time slot constant).
#define vx_rt_get_f_imm_after(slot, wait_status) ({ \
  uint32_t __u = vx_rt_get_after((slot), (wait_status)); \
  union { uint32_t u; float f; } __c; \
  __c.u = __u; \
  __c.f; \
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

// `vx_mret()` (in vx_intrinsics.h) is the trap-return primitive an RTU
// callback dispatcher uses at its bottom to resume the post-vx_rt_wait PC
// and restore the pre-yield tmask.

// Convenience float-encoding helper.
static inline uint32_t vx_rt_f2u(float f) {
  union { float f; uint32_t u; } c;
  c.f = f;
  return c.u;
}

// ===========================================================================
// ISA ABI v2 — scope-partitioned single-issue trace (docs/proposals/
// rtu_isa_v2_proposal.md). The ~16-op vx_rt_set/get marshalling collapses to
// one trace + one wait. Encoding lives at CUSTOM1 / funct3 = 7 (additive — the
// Phase-1 funct3 = 5/6 path is untouched so existing kernels keep building):
//
//   funct2 = 0  vx_rt_trace2  R-type macro-op; rd = handle, rs1 = lane-packed
//                             config, ray window = f0..f7
//   funct2 = 1  vx_rt_wait2   R-type macro-op; rd = status, rs1 = handle;
//                             HW writes t/u/v -> f0..f2, IDs -> t3..t5
// ===========================================================================

// Per-thread ray geometry — the divergent operand that rides the f0..f7
// register window (no marshalling: it is the compiler's register allocation).
typedef struct {
  float origin[3];   // f0..f2
  float dir[3];      // f3..f5
  float tmin;        // f6
  float tmax;        // f7
} vx_ray_t;

// Hit attributes written back by vx_rt_wait2: floats to the FP file, IDs to
// the GP file (the type-split of proposal §5.2 — no fmv conversions).
typedef struct {
  float    t;
  float    u;
  float    v;
  uint32_t primitive_id;
  uint32_t geometry_index;
  uint32_t instance_id;
} vx_hit_t;

// vx_rt_trace2 — issue one ray in a single (macro) instruction.
//   scene_ptr / payload_ptr / ray_flags / cull_mask : per-trace warp-uniform
//     config, lane-packed into one register via an IMPLICIT vx_wgather
//     (lane0=scene, lane1=payload, lane2=flags, lane3=cull) — pure register
//     domain, no memory traffic; hoists out of a bounce loop when invariant.
//   ray : the per-thread geometry, pinned into the f0..f7 caller-saved window.
// Returns the async ray handle in rd. Non-blocking: the RtuCore traverses
// while the kernel runs independent work; vx_rt_wait2 is the sync point.
static inline __attribute__((always_inline))
uint32_t vx_rt_trace2(uint32_t scene_ptr, uint32_t payload_ptr,
                      uint32_t ray_flags, uint32_t cull_mask,
                      const vx_ray_t* ray) {
  // Pack config into the GATHERED operands (not the wgather self slot): the
  // self slot is write-suppressed and so is the one word the partial-warp
  // wgather fix can't materialise from a live lane. lane1=scene, lane2=payload,
  // lane3={flags,cull}; lane0 (self) unused. Keeps scene valid even when lane 0
  // is masked (callback/recursion-narrowed traces).
  uint32_t flags_cull = (ray_flags & 0xffffu) | (cull_mask << 16);
  uint32_t cfg = (uint32_t)vx_wgather(0u, scene_ptr, payload_ptr, flags_cull);
  register float r0 __asm__("f0") = ray->origin[0];
  register float r1 __asm__("f1") = ray->origin[1];
  register float r2 __asm__("f2") = ray->origin[2];
  register float r3 __asm__("f3") = ray->dir[0];
  register float r4 __asm__("f4") = ray->dir[1];
  register float r5 __asm__("f5") = ray->dir[2];
  register float r6 __asm__("f6") = ray->tmin;
  register float r7 __asm__("f7") = ray->tmax;
  uint32_t handle;
  // rd = handle, rs1 = cfg, rs2 = x0. The f0..f7 window rides the operand
  // list (read by HW convention, like the tensor unit's fragment window);
  // the encoding itself only names rd/rs1. Named operands (not %0/%1) keep
  // the field references stable across the long register-binding list.
  __asm__ volatile (".insn r %[op], 7, 0, %[hnd], %[cfg], x0"
    : [hnd]"=r"(handle)
    : [op]"i"(RISCV_CUSTOM1), [cfg]"r"(cfg),
      "f"(r0), "f"(r1), "f"(r2), "f"(r3),
      "f"(r4), "f"(r5), "f"(r6), "f"(r7));
  return handle;
}

// vx_rt_trace2_mas — multi-AS (divergent) trace (proposal §5.4). Each lane
// traces its OWN acceleration structure: the per-thread scene pointer rides
// rs2 directly (no wgather, so lanes keep distinct scenes), while the
// warp-uniform {payload, flags, cull} still lane-pack into rs1 via the implicit
// wgather. Use this only when the AS genuinely diverges across the warp (e.g.
// reformation grouping by per-lane SBT); the warp-uniform vx_rt_trace2 is
// cheaper for the common single-AS case. Encoding: funct3 = 7, funct2 = 2.
static inline __attribute__((always_inline))
uint32_t vx_rt_trace2_mas(uint32_t scene_ptr, uint32_t payload_ptr,
                          uint32_t ray_flags, uint32_t cull_mask,
                          const vx_ray_t* ray) {
  // rs1 carries only the uniform config; the scene lane (lane1) is unused.
  uint32_t flags_cull = (ray_flags & 0xffffu) | (cull_mask << 16);
  uint32_t cfg = (uint32_t)vx_wgather(0u, 0u, payload_ptr, flags_cull);
  register float r0 __asm__("f0") = ray->origin[0];
  register float r1 __asm__("f1") = ray->origin[1];
  register float r2 __asm__("f2") = ray->origin[2];
  register float r3 __asm__("f3") = ray->dir[0];
  register float r4 __asm__("f4") = ray->dir[1];
  register float r5 __asm__("f5") = ray->dir[2];
  register float r6 __asm__("f6") = ray->tmin;
  register float r7 __asm__("f7") = ray->tmax;
  uint32_t handle;
  // rd = handle, rs1 = cfg, rs2 = per-lane scene. The f0..f7 window rides the
  // operand list (read by HW convention), as in vx_rt_trace2.
  __asm__ volatile (".insn r %[op], 7, 2, %[hnd], %[cfg], %[scn]"
    : [hnd]"=r"(handle)
    : [op]"i"(RISCV_CUSTOM1), [cfg]"r"(cfg), [scn]"r"(scene_ptr),
      "f"(r0), "f"(r1), "f"(r2), "f"(r3),
      "f"(r4), "f"(r5), "f"(r6), "f"(r7));
  return handle;
}

// vx_rt_wait2 — block on the ray handle until terminal; return the status
// word in rd and write the hit attributes back to their natural register
// files. Replaces the entire vx_rt_get / vx_rt_get_after dance.
static inline __attribute__((always_inline))
uint32_t vx_rt_wait2(uint32_t handle, vx_hit_t* hit) {
  register float    ht __asm__("f0");
  register float    hu __asm__("f1");
  register float    hv __asm__("f2");
  register uint32_t hp __asm__("t3");
  register uint32_t hg __asm__("t4");
  register uint32_t hi __asm__("t5");
  uint32_t status;
  // rd = status, rs1 = handle. The hit window (f0..f2, t3..t5) rides the
  // output operand list; HW writes it by convention. Named operands keep
  // %[op]/%[sts]/%[hnd] stable regardless of the bound-register count.
  __asm__ volatile (".insn r %[op], 7, 1, %[sts], %[hnd], x0"
    : [sts]"=r"(status), "=f"(ht), "=f"(hu), "=f"(hv),
      "=r"(hp), "=r"(hg), "=r"(hi)
    : [op]"i"(RISCV_CUSTOM1), [hnd]"r"(handle));
  hit->t = ht;
  hit->u = hu;
  hit->v = hv;
  hit->primitive_id   = hp;
  hit->geometry_index = hg;
  hit->instance_id    = hi;
  return status;
}

#ifdef __cplusplus
}
#endif
