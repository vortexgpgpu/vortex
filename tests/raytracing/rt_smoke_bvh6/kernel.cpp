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
// PRISM RTU smoke test — CW-BVH6 kernel.
// Fires one primary ray against a scene_kind=3 (6-wide) BVH whose root
// internal node fans out to six triangle leaves at increasing depth.
// The width-generic walker must test all six children and commit the
// nearest opaque hit.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  vx_ray_t ray = { {arg->ray_origin[0], arg->ray_origin[1], arg->ray_origin[2]},
                   {arg->ray_direction[0], arg->ray_direction[1], arg->ray_direction[2]},
                   arg->tmin, arg->tmax };

  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t h   = vx_rt_wtrace(scene_lo, 0u, VX_RT_FLAG_OPAQUE, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].status         = sts;
  results[0].hit_t          = hit.t;
  results[0].hit_u          = hit.u;
  results[0].hit_v          = hit.v;
  results[0].primitive_id   = hit.primitive_id;
  results[0].geometry_index = hit.geometry_index;
}
