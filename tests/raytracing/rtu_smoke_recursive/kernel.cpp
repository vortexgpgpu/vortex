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

static inline float u2f(uint32_t u) { float f; __builtin_memcpy(&f, &u, 4); return f; }

// CHS dispatcher firing a recursive ray via the v2 ISA (trace2 + wait2). Uses
// the M-mode interrupt attribute so the compiler saves/restores the registers
// the nested trace clobbers (the ray window + hit window). The sub-ray inherits
// the parent's world ray (read back from the regfile) but owns its own flags
// (0 -> no nested CHS yield, which would deadlock against the in_async_trap
// gate while the parent is still mid-callback).
__attribute__((interrupt("machine"), used))
void rt_chs_recursive(void) {
  // Read the payload pointer BEFORE the nested trace2 (which overwrites the
  // PAYLOAD_PTR_LO slot with its own payload arg).
  uint32_t payload   = vx_rt_get(VX_RT_PAYLOAD_PTR_LO);
  uint32_t sub_scene = vx_rt_get(VX_RT_HIT_ATTR_0);   // kernel-stashed sub-scene
  vx_ray_t ray = {
    { u2f(vx_rt_get(VX_RT_RAY_ORIGIN + 0)),
      u2f(vx_rt_get(VX_RT_RAY_ORIGIN + 1)),
      u2f(vx_rt_get(VX_RT_RAY_ORIGIN + 2)) },
    { u2f(vx_rt_get(VX_RT_RAY_DIRECTION + 0)),
      u2f(vx_rt_get(VX_RT_RAY_DIRECTION + 1)),
      u2f(vx_rt_get(VX_RT_RAY_DIRECTION + 2)) },
    u2f(vx_rt_get(VX_RT_T_MIN)), u2f(vx_rt_get(VX_RT_T_MAX))
  };
  uint32_t sub_h = vx_rt_wtrace(sub_scene, 0u, 0u, 0xffu, &ray);
  vx_hit_t sub_hit;
  uint32_t sub_status = vx_rt_wait(sub_h, &sub_hit);
  *(volatile uint32_t*)(uintptr_t)payload = sub_status;
  vx_rt_cb_ret(VX_RT_CB_DONE);
}

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  // Register the CHS recursive dispatcher in mtvec.
  csr_write(0x305, (uintptr_t)&rt_chs_recursive);

  // Pass the sub-scene address through HIT_ATTR_0 (slot 17 is a
  // user attribute slot — the kernel can write it freely, the CHS
  // dispatcher reads it via vx_rt_get).
  vx_rt_set(VX_RT_HIT_ATTR_0,
             (uint32_t)(arg->sub_scene_addr & 0xffffffffu));

  vx_ray_t ray = {
    { arg->ray_origin[0],    arg->ray_origin[1],    arg->ray_origin[2] },
    { arg->ray_direction[0], arg->ray_direction[1], arg->ray_direction[2] },
    arg->tmin, arg->tmax
  };

  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  // payload pointer for the CHS to write sub_status into; enable CHS for
  // the parent ray (so the dispatcher fires).
  uint32_t payload  = (uint32_t)(arg->payload_addr & 0xffffffffu);
  uint32_t h   = vx_rt_wtrace(scene_lo, payload, VX_RT_FLAG_ENABLE_CHS,
                              0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].status              = sts;
  results[0].hit_t               = hit.t;
  // Read the sub_status the recursive CHS wrote only AFTER a wait-dependent op
  // (the get above) so in-order issue holds this load until the parent trace —
  // and its CHS dispatcher's nested trace+wait — have retired.
  uint32_t sub_status = *(volatile uint32_t*)(uintptr_t)arg->payload_addr;
  results[0].sub_status          = sub_status;
  results[0].pad                 = 0;
}
