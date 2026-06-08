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
// PRISM RTU AHS-callback smoke kernel — Phase 2.
//
// Each CTA: register an `mtvec` callback dispatcher (ACCEPT or IGNORE
// flavor based on cb_decision), load ray descriptor, vx_rt_trace +
// vx_rt_wait, read hit attrs, write result. The RtuCore fires an
// async trap during vx_rt_wait → PC → mtvec → dispatcher runs →
// vx_rt_cb_ret + mret → kernel resumes at the post-wait site.
//
// Proposal §4.6 (option-c): the dispatcher reuses the existing M-mode
// trap path (mtvec/mepc/mret) rather than a parallel PC-redirect
// fabric. The dispatcher exits via `mret`, which both restores PC
// (from mepc) and the pre-yield tmask (from mscratch_tmask).

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

// Naked dispatcher that ACCEPTs every yield. Reads no inputs (single
// non-opaque triangle hit -> always accept).
// EXT2 / funct3=6 / sub-op=0 / R-type: vx_rt_cb_ret rs1.
__attribute__((naked, used))
static void rt_dispatcher_accept(void) {
  __asm__ volatile (
    "li t0, %0\n"
    ".insn r %1, 6, 0, x0, t0, x0\n"
    "mret\n"
    :: "i"(VX_RT_CB_ACCEPT), "i"(0x2b)
  );
}

__attribute__((naked, used))
static void rt_dispatcher_ignore(void) {
  __asm__ volatile (
    "li t0, %0\n"
    ".insn r %1, 6, 0, x0, t0, x0\n"
    "mret\n"
    :: "i"(VX_RT_CB_IGNORE), "i"(0x2b)
  );
}

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid >= arg->num_lanes) return;

  // Register the callback dispatcher into mtvec (RISC-V CSR 0x305).
  uintptr_t handler = (arg->cb_decision == RTU_AHS_DECISION_ACCEPT)
                          ? (uintptr_t)&rt_dispatcher_accept
                          : (uintptr_t)&rt_dispatcher_ignore;
  csr_write(0x305, handler);

  // Assemble the ray descriptor.
  vx_ray_t ray = {
    {arg->ray_origin[0], arg->ray_origin[1], arg->ray_origin[2]},
    {arg->ray_direction[0], arg->ray_direction[1], arg->ray_direction[2]},
    arg->tmin,
    arg->tmax,
  };

  // Fire ray + wait for terminal. Ray flags = 0 (no OPAQUE override;
  // per-triangle flags drive AHS). No payload.
  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t h   = vx_rt_trace2(scene_lo, 0u, 0u, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait2(h, &hit);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[tid].status       = sts;
  results[tid].hit_t        = hit.t;
  results[tid].hit_u        = hit.u;
  results[tid].hit_v        = hit.v;
  results[tid].primitive_id = hit.primitive_id;
  results[tid].pad          = 0;
}
