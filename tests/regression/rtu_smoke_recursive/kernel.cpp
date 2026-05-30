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
// PRISM RTU recursive-trace smoke kernel — Phase 12.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

// Naked CHS dispatcher with a recursive vx_rt_trace:
//   vx_rt_set1(VX_RT_RAY_FLAGS, 0)          // sub-trace owns its own flags
//   t0 ← vx_rt_get(VX_RT_HIT_ATTR_0)        // kernel-stashed sub_scene addr
//   t1 ← vx_rt_trace(t0)                    // submit sub-ray, returns handle
//   t2 ← vx_rt_wait(t1)                     // block for sub-ray TERMINAL
//   t3 ← vx_rt_get(VX_RT_PAYLOAD_PTR_LO)
//   *t3 = t2                                 // write sub_status to payload
//   vx_rt_cb_ret(CB_DONE); mret              // release parent
//
// Clearing RAY_FLAGS mirrors NVIDIA's per-TraceRay flag model — the
// dispatcher owns the sub-ray's behavior. Inheriting the parent's
// ENABLE_CHS would queue a recursive CB_YIELD that deadlocks against
// the in_async_trap gate (parent still mid-callback).
//
// funct7 for vx_rt_set1(slot) is (slot << 2) | 0:
//   VX_RT_RAY_FLAGS      (27) → 108
// funct7 for vx_rt_get(slot) is (slot << 2) | 1:
//   VX_RT_HIT_ATTR_0     (17) → 69
//   VX_RT_PAYLOAD_PTR_LO (25) → 101
//
// .insn r 0x2b, 5, 2, rd, rs1, x0   = vx_rt_trace rs1 -> rd
// .insn r 0x2b, 5, 3, rd, rs1, x0   = vx_rt_wait  rs1 -> rd
__attribute__((naked, used))
static void rt_chs_recursive(void) {
  __asm__ volatile (
    ".insn r 0x2b, 5, 108, x0, x0, x0\n"  // VX_RT_RAY_FLAGS = 0
    ".insn r 0x2b, 5, 69, t0, x0, x0\n"   // t0 = sub_scene addr
    ".insn r 0x2b, 5, 2,  t1, t0, x0\n"   // t1 = vx_rt_trace(t0)
    ".insn r 0x2b, 5, 3,  t2, t1, x0\n"   // t2 = vx_rt_wait(t1)
    ".insn r 0x2b, 5, 101, t3, x0, x0\n"  // t3 = payload pointer
    "sw t2, 0(t3)\n"                       // *payload = sub_status
    "li t4, %0\n"                          // t4 = CB_DONE
    ".insn r 0x2b, 6, 0, x0, t4, x0\n"     // vx_rt_cb_ret(t4)
    "mret\n"
    :: "i"(VX_RT_CB_DONE)
  );
}

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  // Register the CHS recursive dispatcher in mtvec.
  csr_write(0x305, (uintptr_t)&rt_chs_recursive);

  // Pass the sub-scene address through HIT_ATTR_0 (slot 17 is a
  // user attribute slot — the kernel can write it freely, the CHS
  // dispatcher reads it via vx_rt_get).
  vx_rt_set1(VX_RT_HIT_ATTR_0,
             (uint32_t)(arg->sub_scene_addr & 0xffffffffu));

  // Payload pointer for the CHS to write sub_status into.
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

  // Enable CHS for the parent ray (so the dispatcher fires).
  vx_rt_set1(VX_RT_RAY_FLAGS, VX_RT_FLAG_ENABLE_CHS);

  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t h   = vx_rt_trace(scene_lo);
  uint32_t sts = vx_rt_wait(h);

  uint32_t hit_t_bits = vx_rt_get(VX_RT_HIT_T);
  uint32_t sub_status = *(volatile uint32_t*)(uintptr_t)arg->payload_addr;

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].status              = sts;
  *(uint32_t*)&results[0].hit_t  = hit_t_bits;
  results[0].sub_status          = sub_status;
  results[0].pad                 = 0;
}
