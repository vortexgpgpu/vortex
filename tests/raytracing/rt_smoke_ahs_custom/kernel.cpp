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
// PRISM RTU any-hit candidate instance-attribute smoke kernel — candidate-
// return loop.
//
// A non-opaque triangle returns an any-hit candidate to the warp. The kernel
// reads the CANDIDATE instance attributes (instance_custom, instance_id) from
// the register window, stashes them into the trace payload buffer, and ACCEPTs
// so the candidate commits.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  vx_ray_t ray = { {arg->ray_origin[0], arg->ray_origin[1], arg->ray_origin[2]},
                   {arg->ray_direction[0], arg->ray_direction[1], arg->ray_direction[2]},
                   arg->tmin, arg->tmax };

  // Ray flags = 0: the non-opaque triangle drives the opacity classifier so
  // the walker yields an AHS candidate callback. The candidate-capture buffer
  // rides the trace as the payload pointer (VX_RT_PAYLOAD_PTR_LO), so the
  // loop can locate it from the window.
  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t cand_lo  = (uint32_t)(arg->cand_addr  & 0xffffffffu);
  uint32_t h   = vx_rt_wtrace(scene_lo, cand_lo, 0u, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);
  while (vx_rt_sts_is_yield(sts)) {
    // Read the candidate instance attributes and the capture buffer pointer
    // (staged as the trace payload) from the register window, then stash.
    uint32_t cand_ptr    = vx_rt_get_attr(VX_RT_PAYLOAD_PTR_LO, sts);
    uint32_t cand_custom = vx_rt_get_attr(VX_RT_HIT_INSTANCE_CUSTOM, sts);
    uint32_t cand_inst   = vx_rt_get_attr(VX_RT_HIT_INSTANCE_ID, sts);
    uint32_t* cand = (uint32_t*)(uintptr_t)cand_ptr;
    cand[0] = cand_custom;   // cand->cand_instance_custom
    cand[1] = cand_inst;     // cand->cand_instance_id
    sts = vx_rt_continue(h, VX_RT_CB_ACCEPT, hit.t, 0u, &hit);
  }

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].status          = sts;
  results[0].hit_t           = hit.t;
  results[0].instance_id     = hit.instance_id;
  results[0].instance_custom = hit.instance_custom;
}
