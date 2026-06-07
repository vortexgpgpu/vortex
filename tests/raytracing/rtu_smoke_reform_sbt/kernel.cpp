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
// PRISM RTU reformation divergent-SBT smoke kernel — Phase 3-A2 option B.
//
// Each lane traces a separate scene that holds a single non-opaque tri
// with a per-lane sbt_idx. The dispatcher reads VX_RT_HIT_SBT_IDX with
// a per-lane vx_rt_get and branches: sbt 0 → ACCEPT, else IGNORE. The
// reformation engine narrows each CB_YIELD's tmask to lanes that share
// an sbt — so inside the trap the per-lane SBT branch is SIMT-coherent
// (every active lane takes the same case) even though the branch is
// data-dependent across lanes.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

// Naked divergent dispatcher.
//   t0 ← vx_rt_get_after(VX_RT_HIT_SBT_IDX, sts)              (per-lane)
//   t1 ← (t0 == 0) ? ACCEPT : IGNORE                (per-lane)
//   vx_rt_cb_ret(t1) ; mret
// Encoded inline so naked can stay stack-free.
__attribute__((naked, used))
static void rt_dispatcher_sbt(void) {
  __asm__ volatile (
    // vx_rt_get VX_RT_HIT_SBT_IDX → t0  (funct3=5, funct2=1, slot in funct7).
    ".insn r %0, 5, %1, t0, x0, x0\n"
    // Per-lane: default to IGNORE, override to ACCEPT iff t0 == 0.
    "li t1, %2\n"
    "bnez t0, 1f\n"
    "li t1, %3\n"
    "1:\n"
    // vx_rt_cb_ret(t1)  (funct3=6, sub-op=0, rs1 = action, no rd).
    ".insn r %0, 6, 0, x0, t1, x0\n"
    "mret\n"
    :: "i"(0x2b),                                /* %0 = CUSTOM1 */
       "i"(((VX_RT_HIT_SBT_IDX) << 2) | 1),      /* %1 = vx_rt_get funct7 */
       "i"(VX_RT_CB_IGNORE),                     /* %2 = default action */
       "i"(VX_RT_CB_ACCEPT)                      /* %3 = sbt==0 action */
  );
}

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = threadIdx.x;
  if (tid >= arg->num_lanes) return;

  csr_write(0x305, (uintptr_t)&rt_dispatcher_sbt);

  // Per-lane scene_root: base + tid * RTU_SCENE_BYTES (one cache line per lane).
  uint32_t scene_addr = (uint32_t)(arg->scene_base_addr & 0xffffffffu)
                      + tid * RTU_SCENE_BYTES;

  vx_ray_t ray = {
    { arg->ray_origin[0],    arg->ray_origin[1],    arg->ray_origin[2] },
    { arg->ray_direction[0], arg->ray_direction[1], arg->ray_direction[2] },
    arg->tmin, arg->tmax
  };

  // Per-lane (divergent) scene: each lane traces its own AS, so the scene must
  // ride rs2 directly rather than the warp-uniform lane-packed config (§5.4).
  uint32_t h   = vx_rt_trace2_mas(scene_addr, 0u, 0u, 0xffu, &ray);
  uint32_t sts = vx_rt_wait(h);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[tid].status            = sts;
  results[tid].hit_t             = vx_rt_get_f_imm_after(VX_RT_HIT_T, sts);
  results[tid].hit_u             = vx_rt_get_f_imm_after(VX_RT_HIT_BARY_U, sts);
  results[tid].hit_v             = vx_rt_get_f_imm_after(VX_RT_HIT_BARY_V, sts);
  results[tid].primitive_id      = vx_rt_get_after(VX_RT_HIT_PRIMITIVE_ID, sts);
  results[tid].pad               = 0;
}
