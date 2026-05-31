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
// PRISM RTU Miss Shader smoke kernel — Phase 5.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

// Naked MISS dispatcher: write MAGIC to the payload, then cb_ret(DONE).
//
// funct7 for vx_rt_get(VX_RT_PAYLOAD_PTR_LO = 25) is (25 << 2) | 1 = 101.
__attribute__((naked, used))
static void rt_miss_dispatcher(void) {
  __asm__ volatile (
    ".insn r 0x2b, 5, 101, t0, x0, x0\n"  // t0 = payload pointer
    "li t1, %0\n"                          // t1 = MAGIC
    "sw t1, 0(t0)\n"                       // *(payload) = MAGIC
    "li t2, %1\n"                          // t2 = CB_DONE
    ".insn r 0x2b, 6, 0, x0, t2, x0\n"     // vx_rt_cb_ret(t2)
    "mret\n"
    :: "i"(RTU_MISS_MAGIC),
       "i"(VX_RT_CB_DONE)
  );
}

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  csr_write(0x305, (uintptr_t)&rt_miss_dispatcher);

  vx_rt_set1(VX_RT_PAYLOAD_PTR_LO,
             (uint32_t)(arg->payload_addr & 0xffffffffu));

  vx_rt_set3(VX_RT_RAY_ORIGIN,
             vx_rt_f2u(arg->ray_origin[0]),
             vx_rt_f2u(arg->ray_origin[1]),
             vx_rt_f2u(arg->ray_origin[2]));
  vx_rt_set3(VX_RT_RAY_DIRECTION,
             vx_rt_f2u(arg->ray_direction[0]),
             vx_rt_f2u(arg->ray_direction[1]),
             vx_rt_f2u(arg->ray_direction[2]));
  vx_rt_set3(VX_RT_T_MIN,
             vx_rt_f2u(arg->tmin),
             vx_rt_f2u(arg->tmax),
             0u);

  vx_rt_set1(VX_RT_RAY_FLAGS, VX_RT_FLAG_ENABLE_MISS);

  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t h   = vx_rt_trace(scene_lo);
  uint32_t sts = vx_rt_wait(h);

  uint32_t hit_t_bits = vx_rt_get_after(VX_RT_HIT_T, sts);

  uint32_t miss_payload = *(volatile uint32_t*)(uintptr_t)arg->payload_addr;

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].status            = sts;
  *(uint32_t*)&results[0].hit_t = hit_t_bits;
  results[0].miss_payload      = miss_payload;
  results[0].pad               = 0;
}
