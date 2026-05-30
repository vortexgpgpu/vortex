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
// PRISM RTU smoke — §8.8 CULL_BACK_FACING kernel.
// Fires two sequential traces against the same scene with
// CULL_BACK_FACING set:
//   ray 0: shoots +z → hits front face → expected HIT
//   ray 1: shoots -z from far side → hits back face → expected MISS (culled)

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

static inline uint32_t fire_ray(uint64_t scene_addr,
                                const float origin[3], const float dir[3],
                                float tmin, float tmax) {
  vx_rt_set3(VX_RT_RAY_ORIGIN,
             vx_rt_f2u(origin[0]), vx_rt_f2u(origin[1]), vx_rt_f2u(origin[2]));
  vx_rt_set3(VX_RT_RAY_DIRECTION,
             vx_rt_f2u(dir[0]), vx_rt_f2u(dir[1]), vx_rt_f2u(dir[2]));
  vx_rt_set3(VX_RT_T_MIN,
             vx_rt_f2u(tmin), vx_rt_f2u(tmax), 0u);
  vx_rt_set1(VX_RT_RAY_FLAGS,
             VX_RT_FLAG_OPAQUE | VX_RT_FLAG_CULL_BACK_FACING);
  uint32_t scene_lo = (uint32_t)(scene_addr & 0xffffffffu);
  uint32_t h = vx_rt_trace(scene_lo);
  return vx_rt_wait(h);
}

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  uint32_t front_sts = fire_ray(arg->scene_addr,
                                arg->front_origin, arg->front_dir,
                                arg->tmin, arg->tmax);
  uint32_t front_t_bits = vx_rt_get(VX_RT_HIT_T);

  uint32_t back_sts = fire_ray(arg->scene_addr,
                               arg->back_origin, arg->back_dir,
                               arg->tmin, arg->tmax);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].front_status = front_sts;
  results[0].back_status  = back_sts;
  *(uint32_t*)&results[0].front_t = front_t_bits;
  results[0].pad          = 0;
}
