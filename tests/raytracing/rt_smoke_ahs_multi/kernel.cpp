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
// RTU multi-candidate any-hit smoke kernel: every lane of the warp traces the
// same ray and ACCEPTs only its target primitive, IGNOREing every other
// candidate, so the lanes' walks diverge -- some end early, others keep being
// offered candidates (and read PENDING while they wait on the others).

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = threadIdx.x;
  if (tid >= arg->num_lanes) return;
  uint32_t target = ((const uint8_t*)(uintptr_t)arg->targets_addr)[tid];

  vx_ray_t ray = {
    {arg->ray_origin[0], arg->ray_origin[1], arg->ray_origin[2]},
    {arg->ray_direction[0], arg->ray_direction[1], arg->ray_direction[2]},
    arg->tmin,
    arg->tmax,
  };

  rtu_result_t* res = (rtu_result_t*)((uintptr_t)arg->results_addr) + tid;
  uint32_t n = 0;

  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t h = vx_rt_wtrace(scene_lo, 0u, 0u, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);
  while (vx_rt_sts_is_yield(sts)) {
    uint32_t action = VX_RT_CB_IGNORE;
    if (vx_rt_sts_has_candidate(sts)) {
      if (n < RTU_MULTI_MAX_OFFER)
        res->offered[n] = (uint8_t)hit.primitive_id;
      ++n;
      if (hit.primitive_id == target)
        action = VX_RT_CB_ACCEPT;
    }
    sts = vx_rt_continue(h, action, hit.t, 0u, &hit);
  }

  res->status       = sts;
  res->hit_t        = hit.t;
  res->primitive_id = hit.primitive_id;
  res->num_offered  = n;
}
