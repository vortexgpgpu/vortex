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

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

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
  vx_rt_set1(VX_RT_RAY_FLAGS, VX_RT_FLAG_OPAQUE);

  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t h   = vx_rt_trace(scene_lo);
  uint32_t sts = vx_rt_wait(h);

  uint32_t hit_t_bits = vx_rt_get(VX_RT_HIT_T);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].status              = sts;
  *(uint32_t*)&results[0].hit_t  = hit_t_bits;
  results[0].hit_u               = 0.f;
  results[0].hit_v               = 0.f;
  results[0].primitive_id        = 0;
  results[0].pad                 = 0;
}
