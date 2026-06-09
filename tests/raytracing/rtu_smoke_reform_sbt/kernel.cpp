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
// PRISM RTU reformation divergent-SBT smoke kernel.
//
// All lanes trace ONE shared (warp-uniform) scene with vx_rt_wtrace; lane i
// aims a +z ray at tri i, so each lane gets a distinct sbt_idx from the tri it
// hits. The dispatcher reads VX_RT_HIT_SBT_IDX with a per-lane vx_rt_get and
// branches: sbt 0 -> ACCEPT, else IGNORE. The reformation engine narrows each
// CB_YIELD's tmask to lanes that share an sbt, so inside the trap the per-lane
// SBT branch is SIMT-coherent even though it is data-dependent across lanes.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

// Naked divergent dispatcher.
//   t0 ← vx_rt_get_after(VX_RT_HIT_SBT_IDX, sts)     (per-lane)
//   t1 ← (t0 == 0) ? ACCEPT : IGNORE                 (per-lane)
//   vx_rt_cb_ret(t1) ; mret
// Encoded inline so naked can stay stack-free.
__attribute__((naked, used))
static void rt_dispatcher_sbt(void) {
  __asm__ volatile (
    // vx_rt_get VX_RT_HIT_SBT_IDX → t0  (GETW funct3=6, funct2=3, slot in
    // funct7, rs2=x1 -> count=1).
    ".insn r %0, 6, %1, t0, x0, x1\n"
    // Per-lane: default to IGNORE, override to ACCEPT iff t0 == 0.
    "li t1, %2\n"
    "bnez t0, 1f\n"
    "li t1, %3\n"
    "1:\n"
    // vx_rt_cb_ret(t1)  (funct3=6, sub-op=0, rs1 = action, no rd).
    ".insn r %0, 6, 0, x0, t1, x0\n"
    "mret\n"
    :: "i"(0x2b),                                /* %0 = CUSTOM1 */
       "i"(((VX_RT_HIT_SBT_IDX) << 2) | 3),      /* %1 = GETW funct7 (sub3) */
       "i"(VX_RT_CB_IGNORE),                     /* %2 = default action */
       "i"(VX_RT_CB_ACCEPT)                      /* %3 = sbt==0 action */
  );
}

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = threadIdx.x;
  if (tid >= arg->num_lanes) return;

  csr_write(0x305, (uintptr_t)&rt_dispatcher_sbt);

  // One shared (warp-uniform) scene; the per-lane ray aims at tri `tid`, so
  // each lane gets a distinct sbt_idx from the tri it hits — divergence rides
  // the ray, not the scene pointer.
  uint32_t scene_addr = (uint32_t)(arg->scene_base_addr & 0xffffffffu);

  float ox = (float)tid * RTU_TRI_SPACING + RTU_RAY_XOFF;
  vx_ray_t ray = {
    { ox, RTU_RAY_Y, 0.f },
    { 0.f, 0.f, 1.f },
    arg->tmin, arg->tmax
  };

  uint32_t h   = vx_rt_wtrace(scene_addr, 0u, 0u, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[tid].status            = sts;
  results[tid].hit_t             = hit.t;
  results[tid].hit_u             = hit.u;
  results[tid].hit_v             = hit.v;
  results[tid].primitive_id      = hit.primitive_id;
  results[tid].pad               = 0;
}
