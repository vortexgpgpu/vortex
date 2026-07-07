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
// PRISM RTU any-hit candidate instance-attribute smoke kernel — I1 guard.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

// Naked AHS dispatcher, entered via the M-mode trap (mtvec) during the
// vx_rt_wait candidate callback (see rtu_smoke_ahs_bvh / rtu_smoke_is for
// the trap-path pattern). It reads the CANDIDATE instance attributes from
// the register window and stashes them into the trace payload buffer, then
// ACCEPTs so the candidate commits.
//
// GETW reads one window slot into rd (funct3=6, funct7 = (slot<<2)|3):
//   VX_RT_PAYLOAD_PTR_LO      (25) -> (25<<2)|3 = 103   (capture buffer ptr)
//   VX_RT_HIT_INSTANCE_CUSTOM (24) -> (24<<2)|3 =  99
//   VX_RT_HIT_INSTANCE_ID     (22) -> (22<<2)|3 =  91
__attribute__((naked, used))
static void rt_ahs_custom_dispatcher(void) {
  __asm__ volatile (
    ".insn r 0x2b, 6, 103, t1, x0, x1\n"   // t1 = candidate-capture buffer ptr
    ".insn r 0x2b, 6, 99,  t2, x0, x1\n"   // t2 = candidate instance_custom
    ".insn r 0x2b, 6, 91,  t3, x0, x1\n"   // t3 = candidate instance_id
    "sw t2, 0(t1)\n"                        // cand->cand_instance_custom
    "sw t3, 4(t1)\n"                        // cand->cand_instance_id
    "li t4, %0\n"                           // t4 = CB_ACCEPT
    ".insn r 0x2b, 6, 0, x0, t4, x0\n"      // vx_rt_cb_ret(CB_ACCEPT)
    "mret\n"
    :: "i"(VX_RT_CB_ACCEPT)
  );
}

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  csr_write(0x305, (uintptr_t)&rt_ahs_custom_dispatcher);

  vx_ray_t ray = { {arg->ray_origin[0], arg->ray_origin[1], arg->ray_origin[2]},
                   {arg->ray_direction[0], arg->ray_direction[1], arg->ray_direction[2]},
                   arg->tmin, arg->tmax };

  // Ray flags = 0: the non-opaque triangle drives the opacity classifier so
  // the walker yields an AHS candidate callback. The candidate-capture buffer
  // rides the trace as the payload pointer (VX_RT_PAYLOAD_PTR_LO), so the
  // dispatcher can locate it from the window.
  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t cand_lo  = (uint32_t)(arg->cand_addr  & 0xffffffffu);
  uint32_t h   = vx_rt_wtrace(scene_lo, cand_lo, 0u, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].status          = sts;
  results[0].hit_t           = hit.t;
  results[0].instance_id     = hit.instance_id;
  results[0].instance_custom = hit.instance_custom;
}
