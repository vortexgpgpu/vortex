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
// §8.8 instanceCullMask smoke — ISA ABI v2. Three rays at the same
// origin/dir but different cull masks — the walker should skip instances
// whose mask doesn't overlap. Validates per-ray status, hit_t, and
// hit_instance_id.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  // Stage ray origin/direction/tmin/tmax once — only the cull mask
  // changes between the three rays.
  vx_ray_t ray;
  ray.origin[0] = arg->ray_origin[0];
  ray.origin[1] = arg->ray_origin[1];
  ray.origin[2] = arg->ray_origin[2];
  ray.dir[0]    = arg->ray_direction[0];
  ray.dir[1]    = arg->ray_direction[1];
  ray.dir[2]    = arg->ray_direction[2];
  ray.tmin      = arg->tmin;
  ray.tmax      = arg->tmax;

  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);

  // Each ray: fire+wait with a distinct cull mask, read results.
  for (uint32_t i = 0; i < RTU_NUM_RAYS; ++i) {
    uint32_t h = vx_rt_trace2(scene_lo, 0u, VX_RT_FLAG_OPAQUE, arg->ray_cull_mask[i], &ray);
    vx_hit_t hit;
    uint32_t sts = vx_rt_wait2(h, &hit);
    results[tid].rays[i].status          = sts;
    results[tid].rays[i].hit_t           = hit.t;
    results[tid].rays[i].hit_instance_id = hit.instance_id;
    results[tid].rays[i].pad             = 0;
  }
}
