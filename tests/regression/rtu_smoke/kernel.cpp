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
// PRISM RTU smoke test — Phase 1 kernel.
//
// Each CTA: load ray descriptor → vx_rt_set the RTU register file →
// vx_rt_trace + vx_rt_wait → read hit attrs from the RTU register file →
// write result to per-lane slot. CPU oracle compares.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid >= arg->num_lanes) return;

  // Set ray origin (3 slots).
  uint32_t ox = vx_rt_f2u(arg->ray_origin[0]);
  uint32_t oy = vx_rt_f2u(arg->ray_origin[1]);
  uint32_t oz = vx_rt_f2u(arg->ray_origin[2]);
  vx_rt_set3(VX_RT_RAY_ORIGIN, ox, oy, oz);

  // Set ray direction (3 slots).
  uint32_t dx = vx_rt_f2u(arg->ray_direction[0]);
  uint32_t dy = vx_rt_f2u(arg->ray_direction[1]);
  uint32_t dz = vx_rt_f2u(arg->ray_direction[2]);
  vx_rt_set3(VX_RT_RAY_DIRECTION, dx, dy, dz);

  // Set tmin / tmax / padding.
  vx_rt_set3(VX_RT_T_MIN,
             vx_rt_f2u(arg->tmin),
             vx_rt_f2u(arg->tmax),
             0u);

  // Set ray flags.
  vx_rt_set1(VX_RT_RAY_FLAGS, VX_RT_FLAG_OPAQUE);

  // Fire ray, then wait for terminal status.
  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t h = vx_rt_trace(scene_lo);
  uint32_t sts = vx_rt_wait(h);

  // Read hit attributes.
  uint32_t hit_t_bits = vx_rt_get(VX_RT_HIT_T);
  uint32_t hit_u_bits = vx_rt_get(VX_RT_HIT_BARY_U);
  uint32_t hit_v_bits = vx_rt_get(VX_RT_HIT_BARY_V);
  uint32_t prim_id   = vx_rt_get(VX_RT_HIT_PRIMITIVE_ID);

  // Store per-lane result.
  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[tid].status       = sts;
  *(uint32_t*)&results[tid].hit_t = hit_t_bits;
  *(uint32_t*)&results[tid].hit_u = hit_u_bits;
  *(uint32_t*)&results[tid].hit_v = hit_v_bits;
  results[tid].primitive_id = prim_id;
  results[tid].pad          = 0;
}
