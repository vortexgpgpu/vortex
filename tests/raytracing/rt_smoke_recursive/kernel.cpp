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
// PRISM RTU recursive-trace smoke kernel — candidate-return CHS + nested trace.
//
// The parent ray enables CHS; its closest hit returns a candidate to this warp,
// which reads the parent's ray back out of the window and fires a nested
// vx_rt_wtrace/vx_rt_wait, writes the sub-ray's status to the payload, then
// CONTINUEs the parent with DONE. NOTE: nested tracing shares the single per-
// warp window context on RTL — recursion needs a per-warp context stack (a
// future extension); the RTL path is the tracked Phase-F item.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

static inline float u2f(uint32_t u) { float f; __builtin_memcpy(&f, &u, 4); return f; }

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  // Pass the sub-scene address through HIT_ATTR_0 (slot 17 is a user attribute
  // slot — the kernel writes it freely, the CHS loop body reads it).
  vx_gfx_set(VX_RT_HIT_ATTR_0,
             (uint32_t)(arg->sub_scene_addr & 0xffffffffu));

  vx_ray_t ray = {
    { arg->ray_origin[0],    arg->ray_origin[1],    arg->ray_origin[2] },
    { arg->ray_direction[0], arg->ray_direction[1], arg->ray_direction[2] },
    arg->tmin, arg->tmax
  };

  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  // payload pointer for the CHS to write sub_status into; enable CHS for
  // the parent ray (so its closest hit returns a candidate).
  uint32_t payload  = (uint32_t)(arg->payload_addr & 0xffffffffu);
  uint32_t h   = vx_rt_wtrace(scene_lo, payload, VX_RT_FLAG_ENABLE_CHS,
                              0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);
  while (vx_rt_sts_is_yield(sts)) {
    // CHS body: read the payload pointer + stashed sub-scene, reconstruct the
    // parent's world ray from the window, and fire the nested trace.
    uint32_t payload_p = vx_gfx_get(VX_RT_PAYLOAD_PTR_LO);
    uint32_t sub_scene = vx_gfx_get(VX_RT_HIT_ATTR_0);
    vx_ray_t sub_ray = {
      { u2f(vx_gfx_get(VX_RT_RAY_ORIGIN + 0)),
        u2f(vx_gfx_get(VX_RT_RAY_ORIGIN + 1)),
        u2f(vx_gfx_get(VX_RT_RAY_ORIGIN + 2)) },
      { u2f(vx_gfx_get(VX_RT_RAY_DIRECTION + 0)),
        u2f(vx_gfx_get(VX_RT_RAY_DIRECTION + 1)),
        u2f(vx_gfx_get(VX_RT_RAY_DIRECTION + 2)) },
      u2f(vx_gfx_get(VX_RT_T_MIN)), u2f(vx_gfx_get(VX_RT_T_MAX))
    };
    uint32_t sub_h = vx_rt_wtrace(sub_scene, 0u, 0u, 0xffu, &sub_ray);
    vx_hit_t sub_hit;
    uint32_t sub_status = vx_rt_wait(sub_h, &sub_hit);
    *(volatile uint32_t*)(uintptr_t)payload_p = sub_status;
    sts = vx_rt_continue(h, VX_RT_CB_DONE, &hit);
  }

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
