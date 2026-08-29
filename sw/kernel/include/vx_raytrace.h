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
// Ray-tracing intrinsics (CUSTOM1, funct3 = 6/7).
//
// The hit window is READ-ONLY to the shader: the RTU writes it, the shader reads
// it, and there is no window-write op. Everything a shader hands back to the RTU
// — an intersection shader's hit distance and hitAttribute — rides the operands
// of the instruction that hands it back.
//
// The callback-side helpers ride funct3 = 6 (funct2: 0 = cb_ret / continue,
// 2 = GETWF FP read, 3 = GETW GP read); the per-trace path (vx_rt_wtrace /
// vx_rt_wait, further down) rides funct3 = 7.

#pragma once

#include <vx_intrinsics.h>
#include <VX_types.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The hit window is written by the RTU and read by the shader — never the other
// way round. There is no window-write primitive: every value a shader hands back
// to the RTU (an intersection shader's t and hit attribute) rides the operands of
// the instruction that hands it back, so a trace is self-contained.

// vx_rt_get_attr — read one RTU-written window slot, chained on the status word
// that produced it (funct2=3; slot in funct7[6:2]). `tok` rides rs1 so the
// scoreboard stalls the read until the RTU has staged the slot; the hardware ignores
// its value. The "memory" clobber keeps a callback-written load from hoisting
// above it.
#define vx_rt_get_attr(slot, tok) ({ \
  uint32_t __v; \
  __asm__ volatile (".insn r %1, 6, %2, %0, %3, x1" \
      : "=r"(__v) \
      : "i"(RISCV_CUSTOM1), "i"(((slot) << 2) | 3), "r"(tok) \
      : "memory"); \
  __v; })

// vx_rt_cb_ret — release the lane's parked context in the RTU with one of
// VX_RT_CB_{ACCEPT,IGNORE,TERMINATE}, handing back the intersection shader's
// computed hit distance and attribute in the same instruction. The callback
// dispatcher (mtvec-registered) calls this once it has decided the candidate
// hit's fate, then exits via `mret` to resume the post-vx_rt_wait PC. The
// dispatcher's tmask was narrowed at trap entry to only-yielded-lanes and is
// restored by `mret` from the saved tmask CSR.
//
// CUSTOM1 / funct3=6 / sub-op=0 / R4-type. rs1 = action, rs2 = t (FP),
// rs3 = hit attribute (GP); no rd. An IGNORE ignores t/attr.
#define vx_rt_cb_ret(action, t, attr) \
  __asm__ volatile (".insn r4 %0, 6, 0, x0, %1, %2, %3" \
      :: "i"(RISCV_CUSTOM1), "r"(action), "f"(t), "r"(attr) : "memory")

// Mark a function as the RTU callback dispatcher. The compiler should emit it
// as a normal extern-"C" function; the dispatcher is responsible for exiting
// via `mret` (e.g. by tail-calling a small `mret` stub or finishing with
// `vx_rt_cb_ret` + an inline `mret` asm). A no-op attribute today; a future
// Mesa codegen pass can lower it into a real attribute (preserve `mret`
// epilogue, no caller-save spill).
#define VX_RT_CALLBACK_ENTRY __attribute__((used))

// `vx_mret()` (in vx_intrinsics.h) is the trap-return primitive an RTU
// callback dispatcher uses at its bottom to resume the post-vx_rt_wait PC
// and restore the pre-yield tmask.

// ===========================================================================
// Warp-scoped single-issue trace: the per-slot marshalling collapses to one
// trace + one wait. Encoding lives at CUSTOM1 / funct3 = 7:
//
//   funct2 = 0  vx_rt_wtrace      R-type macro-op; rd = handle, rs1 = lane-packed
//                                config, ray window = f0..f7
//   funct2 = 1  vx_rt_wait       R-type macro-op; rd = status, rs1 = handle;
//                                HW writes t/u/v -> f0..f2, IDs -> t3..t5
// ===========================================================================

// Per-thread ray geometry — the divergent operand that rides the f0..f7
// register window (no marshalling: it is the compiler's register allocation).
typedef struct {
  float origin[3];   // f0..f2
  float dir[3];      // f3..f5
  float tmin;        // f6
  float tmax;        // f7
} vx_ray_t;

// The f0..f7 ray window is read by the trace decoder by HW convention (the
// decoders hardcode f0..f7 with no runtime check) and is fed by the
// register-pinned asm operands in vx_rt_wtrace below. Enforce that vx_ray_t is
// exactly the eight contiguous floats those operands map onto, so a field
// reorder or added padding fails to compile rather than silently scrambling
// the per-lane ray window streamed into the RTU.
_Static_assert(sizeof(vx_ray_t) == 8 * sizeof(float),
               "vx_ray_t must be exactly the 8-float f0..f7 ray window (no padding)");
_Static_assert(offsetof(vx_ray_t, origin) == 0,                 "vx_ray_t.origin -> f0..f2");
_Static_assert(offsetof(vx_ray_t, dir)    == 3 * sizeof(float), "vx_ray_t.dir -> f3..f5");
_Static_assert(offsetof(vx_ray_t, tmin)   == 6 * sizeof(float), "vx_ray_t.tmin -> f6");
_Static_assert(offsetof(vx_ray_t, tmax)   == 7 * sizeof(float), "vx_ray_t.tmax -> f7");

// Hit attributes written back by vx_rt_wait: floats to the FP file, IDs to the
// GP file (the type-split keeps floats and IDs in their natural files — no fmv
// conversions).
typedef struct {
  float    t;
  float    u;
  float    v;
  uint32_t primitive_id;
  uint32_t geometry_index;
  uint32_t instance_id;
  uint32_t instance_custom;   // gl_InstanceCustomIndexEXT (VK_INSTANCE_CUSTOM_INDEX)
} vx_hit_t;

// The struct field order (memory layout) is intentionally NOT the RTU register-
// window slot order: the ID window reads slots 21..24 =
// {primitive_id, instance_id, geometry_index, instance_custom}, whereas this
// struct groups geometry_index BEFORE instance_id. vx_rt_wait() marshals each
// slot into its named field individually, so the two orders are decoupled — do
// NOT bulk-copy the register window into this struct. The asserts pin the layout
// so a field reorder is caught at compile time.
_Static_assert(offsetof(vx_hit_t, t)               == 0 * sizeof(float), "vx_hit_t.t");
_Static_assert(offsetof(vx_hit_t, u)               == 1 * sizeof(float), "vx_hit_t.u");
_Static_assert(offsetof(vx_hit_t, v)               == 2 * sizeof(float), "vx_hit_t.v");
_Static_assert(offsetof(vx_hit_t, primitive_id)    == 3 * sizeof(float), "vx_hit_t.primitive_id");
_Static_assert(offsetof(vx_hit_t, geometry_index)  == 4 * sizeof(float), "vx_hit_t.geometry_index");
_Static_assert(offsetof(vx_hit_t, instance_id)     == 5 * sizeof(float), "vx_hit_t.instance_id");
_Static_assert(offsetof(vx_hit_t, instance_custom) == 6 * sizeof(float), "vx_hit_t.instance_custom");

// vx_rt_wtrace — issue one ray in a single warp-level (macro) instruction. The
// 'w' marks it warp-scoped: rs1 is the lane-packed warp-uniform config read
// across the warp's lanes, and the f0..f7 ray window is per-lane.
//   scene_ptr / payload_ptr / ray_flags / cull_mask : per-trace warp-uniform
//     config, lane-packed into one register via an IMPLICIT vx_wgather
//     (lane0=scene, lane1=payload, lane2=flags, lane3=cull) — pure register
//     domain, no memory traffic; hoists out of a bounce loop when invariant.
//   ray : the per-thread geometry, pinned into the f0..f7 caller-saved window.
// Returns the async ray handle in rd. Non-blocking: the RTU traverses
// while the kernel runs independent work; vx_rt_wait is the sync point.
static inline __attribute__((always_inline))
uint32_t vx_rt_wtrace(uint32_t scene_ptr, uint32_t payload_ptr,
                     uint32_t ray_flags, uint32_t cull_mask,
                     const vx_ray_t* ray) {
  // Pack config into the GATHERED operands (not the wgather self slot): the
  // self slot is write-suppressed and so is the one word the partial-warp
  // wgather can't materialise from a live lane. lane1=scene, lane2=payload,
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

// vx_rt_wait — block on the ray handle until terminal, return the status word,
// and write the hit attributes back to their natural register files. Emitted as
// TWO ops so it composes with callback-yielding traces:
//   (1) WAIT (funct2=1) — a SINGLE-OP block. It parks/revives exactly like the
//       register-file get path, so it survives the async callback trap (a parked
//       single op is revived by HW on terminal; a parking macro-op could not
//       have its writeback uops resumed after the trap flush).
//   (2) WAIT_WB (funct2=3) — a NON-blocking hit-window writeback macro-op,
//       scoreboard-chained on the status word, so it issues only after the
//       block retired (terminal). Running post-terminal, it never coincides
//       with a callback trap. Hit window: t/u/v -> f0..f2, IDs -> t3..t5.
// The "memory" clobber on the block keeps a callback-written memory load placed
// after the wait from hoisting above it.
static inline __attribute__((always_inline))
uint32_t vx_rt_wait(uint32_t handle, vx_hit_t* hit) {
  uint32_t status;
  __asm__ volatile (".insn r %[op], 7, 1, %[sts], %[hnd], x0"
    : [sts]"=r"(status)
    : [op]"i"(RISCV_CUSTOM1), [hnd]"r"(handle)
    : "memory");
  // (2a) hit t/u/v via an FP windowed read (slots HIT_T..HIT_BARY_V -> f0..f2),
  // chained on status (rs1) so it issues only after the block's terminal staged
  // the hit. Single register class -> reliable codegen.
  register float ht __asm__("f0");
  register float hu __asm__("f1");
  register float hv __asm__("f2");
  __asm__ volatile (".insn r %[op], 6, %[f7], %[w0], %[sts], x3"
    : [w0]"=f"(ht), "=f"(hu), "=f"(hv)
    : [op]"i"(RISCV_CUSTOM1), [f7]"i"(((VX_RT_HIT_T) << 2) | 2), [sts]"r"(status));
  // (2b) hit IDs via a GP windowed read (slots HIT_PRIMITIVE_ID..,+3 -> t3..t6):
  // 21 = primitive_id, 22 = instance_id, 23 = geometry_index,
  // 24 = instance_custom (gl_InstanceCustomIndexEXT).
  register uint32_t hp __asm__("t3");
  register uint32_t hi __asm__("t4");
  register uint32_t hg __asm__("t5");
  register uint32_t hc __asm__("t6");
  __asm__ volatile (".insn r %[op], 6, %[f7], %[w0], %[sts], x4"
    : [w0]"=r"(hp), "=r"(hi), "=r"(hg), "=r"(hc)
    : [op]"i"(RISCV_CUSTOM1), [f7]"i"(((VX_RT_HIT_PRIMITIVE_ID) << 2) | 3), [sts]"r"(status));
  hit->t = ht;
  hit->u = hu;
  hit->v = hv;
  hit->primitive_id    = hp;
  hit->instance_id     = hi;
  hit->geometry_index  = hg;
  hit->instance_custom = hc;
  return status;
}

// Loop condition of the candidate-return traversal: true while this lane must
// keep proceeding (rayQueryProceedEXT semantics), false once it reaches a
// terminal (DONE_HIT / DONE_MISS).
//
// It covers two cases. YIELD_ANYHIT / YIELD_PROC — the lane has a candidate to
// service, so the loop body reads it and decides an action. PENDING — the lane
// is still traversing and has NO candidate this iteration: the hardware yielded
// a batch that covers only some lanes (e.g. divergent-SBT reformation groups
// candidates by shader). A PENDING lane stays in the loop (it must not exit on
// stale data) and whatever action it computes is ignored by the RTU, which only
// applies actions to lanes with a pending candidate. Use vx_rt_sts_has_candidate
// when the loop body must distinguish the two.
static inline int vx_rt_sts_is_yield(uint32_t status) {
  return (status == VX_RT_STS_YIELD_ANYHIT)
      || (status == VX_RT_STS_YIELD_PROC)
      || (status == VX_RT_STS_PENDING);
}

// True only when this lane actually has a candidate to shade this iteration.
static inline int vx_rt_sts_has_candidate(uint32_t status) {
  return (status == VX_RT_STS_YIELD_ANYHIT) || (status == VX_RT_STS_YIELD_PROC);
}

// vx_rt_continue — resume traversal for a returned candidate. Hands back the
// per-lane action (VX_RT_CB_{ACCEPT,IGNORE,TERMINATE}) together with the hit
// distance and attribute an intersection shader computed for it, then blocks on
// the next response and returns its status + hit window, exactly like
// vx_rt_wait. `t`/`attr` ride the CONTINUE operands, so the candidate's verdict
// is one self-contained instruction and the shader never writes RTU state.
// An any-hit lane (which computes no t of its own) passes the candidate's own
// hit.t straight back. The natural ray-query loop is:
//
//     uint32_t sts = vx_rt_wait(h, &hit);
//     while (vx_rt_sts_is_yield(sts)) {
//         uint32_t action = /* any-hit / intersection decision from hit */;
//         sts = vx_rt_continue(h, action, t, attr, &hit);
//     }
//
// This is the Vulkan/DXR rayQueryProceedEXT shape, 1:1 with the hardware:
// the candidate is returned to the issuing warp (no trap, no dispatcher).
static inline __attribute__((always_inline))
uint32_t vx_rt_continue(uint32_t handle, uint32_t action, float t, uint32_t attr,
                        vx_hit_t* hit) {
  // CONTINUE: CUSTOM1 / funct3=6 / sub-op=0 / R4-type.
  // rs1 = action, rs2 = t (FP), rs3 = attr (GP); no rd.
  __asm__ volatile (".insn r4 %0, 6, 0, x0, %1, %2, %3"
      :: "i"(RISCV_CUSTOM1), "r"(action), "f"(t), "r"(attr) : "memory");
  return vx_rt_wait(handle, hit);
}

// ===========================================================================
// Callback-side register-window read. The trace/wait path
// collapsed the kernel's field-by-field marshalling; this does the same for the
// in-trap callback read path: a dispatcher that needs several
// contiguous float slots (e.g. the object-space ray an IS shader reads) issues
// ONE windowed read instead of N single-slot reads + N fmv. Encoding: CUSTOM1 /
// funct3 = 6 / funct2 = 2 (GETWF); the window start slot rides funct7[6:2] and
// the slot count rides the rs2 register-field index (an immediate). Values land
// in an FP register group with no int->float conversion.
// ===========================================================================

// Object-space ray staged by the RTU on an AHS/IS yield (slots
// VX_RT_OBJECT_RAY_ORIGIN..DIRECTION, six contiguous floats).
typedef struct {
  float origin[3];
  float dir[3];
} vx_objray_t;

// vx_rt_get_objray — read the six object-ray floats (VX_RT_OBJECT_RAY_ORIGIN..
// DIRECTION) into the f0..f5 window in one macro-op. Replaces the 6x single-slot read
// + 6x fmv an intersection-shader dispatcher would otherwise emit. Call inside
// a callback dispatcher (the regfile holds the candidate's object-space ray
// after the yield).
static inline __attribute__((always_inline))
void vx_rt_get_objray(vx_objray_t* out) {
  register float r0 __asm__("f0");
  register float r1 __asm__("f1");
  register float r2 __asm__("f2");
  register float r3 __asm__("f3");
  register float r4 __asm__("f4");
  register float r5 __asm__("f5");
  // rd = f0 (window base), rs2 = x6 (count = 6); funct7 = (start_slot << 2) | 2.
  __asm__ volatile (".insn r %[op], 6, %[f7], %[w0], x0, x6"
    : [w0]"=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3), "=f"(r4), "=f"(r5)
    : [op]"i"(RISCV_CUSTOM1),
      [f7]"i"(((VX_RT_OBJECT_RAY_ORIGIN) << 2) | 2));
  out->origin[0] = r0; out->origin[1] = r1; out->origin[2] = r2;
  out->dir[0]    = r3; out->dir[1]    = r4; out->dir[2]    = r5;
}

// vx_rt_wtrace_sync — fused trace+wait for the common ray-query case where the
// kernel needs the hit immediately (no independent work to overlap). This does
// NOT earn a dedicated opcode: a real fused instruction would have to PARK
// mid-macro-op (between arm and writeback),
// adding sequencer/scoreboard complexity, and it would forfeit the async
// overlap that is the whole point of the trace/wait split — all to save a
// single instruction fetch (the handle never leaves a register anyway). So the
// sync form is just the two macro-ops back to back; when the kernel DOES
// have independent work, it calls trace/wait separately and the compiler
// schedules that work into the gap.
static inline __attribute__((always_inline))
uint32_t vx_rt_wtrace_sync(uint32_t scene_ptr, uint32_t payload_ptr,
                          uint32_t ray_flags, uint32_t cull_mask,
                          const vx_ray_t* ray, vx_hit_t* hit) {
  uint32_t h = vx_rt_wtrace(scene_ptr, payload_ptr, ray_flags, cull_mask, ray);
  return vx_rt_wait(h, hit);
}

#ifdef __cplusplus
}
#endif
