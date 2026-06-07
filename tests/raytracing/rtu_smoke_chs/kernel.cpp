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
// PRISM RTU Closest-Hit Shader smoke kernel — Phase 5.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

// Naked CHS dispatcher.
//   t0 ← vx_rt_get_after(VX_RT_HIT_T, sts)
//   t1 ← vx_rt_get_after(VX_RT_PAYLOAD_PTR_LO, sts)
//   t2 ← MAGIC ^ hit_t_bits ; *t1 = t2
//   vx_rt_cb_ret(VX_RT_CB_DONE) ; mret
//
// funct7 for vx_rt_get(slot) is (slot << 2) | 1:
//   VX_RT_HIT_T            (14) → 57
//   VX_RT_PAYLOAD_PTR_LO   (25) → 101
__attribute__((naked, used))
static void rt_chs_dispatcher(void) {
  __asm__ volatile (
    ".insn r 0x2b, 5, 57,  t0, x0, x0\n"   // t0 = hit_t bits
    ".insn r 0x2b, 5, 101, t1, x0, x0\n"   // t1 = payload pointer
    "li t2, %0\n"                           // t2 = MAGIC
    "xor t2, t2, t0\n"                      // t2 ^= hit_t bits
    "sw t2, 0(t1)\n"                        // *(payload) = result
    "li t3, %1\n"                           // t3 = CB_DONE
    ".insn r 0x2b, 6, 0, x0, t3, x0\n"      // vx_rt_cb_ret(t3)
    "mret\n"
    :: "i"(RTU_CHS_MAGIC),
       "i"(VX_RT_CB_DONE)
  );
}

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  // Register the CHS dispatcher in mtvec (CSR 0x305).
  csr_write(0x305, (uintptr_t)&rt_chs_dispatcher);

  // Assemble the ray descriptor.
  vx_ray_t ray = {
    {arg->ray_origin[0], arg->ray_origin[1], arg->ray_origin[2]},
    {arg->ray_direction[0], arg->ray_direction[1], arg->ray_direction[2]},
    arg->tmin,
    arg->tmax,
  };

  // The trace stages the payload pointer the CHS dispatcher reads via
  // vx_rt_get. Opt into CHS dispatch (Phase 5).
  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t payload  = (uint32_t)(arg->payload_addr & 0xffffffffu);
  uint32_t h = vx_rt_trace2(scene_lo, payload, VX_RT_FLAG_ENABLE_CHS, 0xffu, &ray);
  uint32_t sts = vx_rt_wait(h);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].status            = sts;
  results[0].hit_t             = vx_rt_get_f_imm_after(VX_RT_HIT_T, sts);
  results[0].primitive_id      = vx_rt_get_after(VX_RT_HIT_PRIMITIVE_ID, sts);
  // Read the payload the CHS wrote only AFTER a wait-dependent op (the gets
  // above) so in-order issue holds this load until the trace — and its CHS
  // callback store — have retired.
  uint32_t chs_payload = *(volatile uint32_t*)(uintptr_t)arg->payload_addr;
  results[0].chs_payload       = chs_payload;
}
