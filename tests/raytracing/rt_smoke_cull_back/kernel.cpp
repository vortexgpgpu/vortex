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
// PRISM RTU smoke — §8.8 CULL_BACK_FACING kernel, ISA ABI v2.
// Fires two sequential traces against the same scene with
// CULL_BACK_FACING set:
//   ray 0: shoots +z → hits front face → expected HIT
//   ray 1: shoots -z from far side → hits back face → expected MISS (culled)

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

static inline uint32_t fire_ray(uint64_t scene_addr,
                                const float origin[3], const float dir[3],
                                float tmin, float tmax, vx_hit_t* hit) {
  vx_ray_t ray;
  ray.origin[0] = origin[0];
  ray.origin[1] = origin[1];
  ray.origin[2] = origin[2];
  ray.dir[0]    = dir[0];
  ray.dir[1]    = dir[1];
  ray.dir[2]    = dir[2];
  ray.tmin      = tmin;
  ray.tmax      = tmax;
  uint32_t scene_lo = (uint32_t)(scene_addr & 0xffffffffu);
  uint32_t h = vx_rt_wtrace(scene_lo, 0u,
                            VX_RT_FLAG_OPAQUE | VX_RT_FLAG_CULL_BACK_FACING,
                            0xffu, &ray);
  return vx_rt_wait(h, hit);
}

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  vx_hit_t front_hit;
  uint32_t front_sts = fire_ray(arg->scene_addr,
                                arg->front_origin, arg->front_dir,
                                arg->tmin, arg->tmax, &front_hit);

  vx_hit_t back_hit;
  uint32_t back_sts = fire_ray(arg->scene_addr,
                               arg->back_origin, arg->back_dir,
                               arg->tmin, arg->tmax, &back_hit);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].front_status = front_sts;
  results[0].back_status  = back_sts;
  results[0].front_t      = front_hit.t;
  results[0].pad          = 0;
}
