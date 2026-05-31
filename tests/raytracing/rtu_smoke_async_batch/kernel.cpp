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
// §8.6 async batch smoke kernel. Issues RTU_ASYNC_NUM_BATCH back-to-
// back vx_rt_trace calls — each one targets a different scene with a
// triangle at a unique z — then drains the handles with the matching
// vx_rt_wait + vx_rt_get_after pair. If any of the WAIT-handle ↔
// TERMINAL routings is wrong, hit_t for one or more rays will be
// off (or the same value will appear for multiple rays).

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid >= arg->num_lanes) return;

  // Stage the ray descriptor once — all rays share origin/dir; only
  // the scene (and thus the target triangle's z) differs across the
  // batch. tmin/tmax also identical.
  uint32_t ox = vx_rt_f2u(arg->ray_origin[0]);
  uint32_t oy = vx_rt_f2u(arg->ray_origin[1]);
  uint32_t oz = vx_rt_f2u(arg->ray_origin[2]);
  vx_rt_set3(VX_RT_RAY_ORIGIN, ox, oy, oz);
  uint32_t dx = vx_rt_f2u(arg->ray_direction[0]);
  uint32_t dy = vx_rt_f2u(arg->ray_direction[1]);
  uint32_t dz = vx_rt_f2u(arg->ray_direction[2]);
  vx_rt_set3(VX_RT_RAY_DIRECTION, dx, dy, dz);
  vx_rt_set3(VX_RT_T_MIN,
             vx_rt_f2u(arg->tmin),
             vx_rt_f2u(arg->tmax),
             0u);
  vx_rt_set1(VX_RT_RAY_FLAGS, VX_RT_FLAG_OPAQUE);

  // Issue RTU_ASYNC_NUM_BATCH traces back-to-back. No WAIT in
  // between — each ray runs async in the cluster's RtuCore.
  uint32_t h0 = vx_rt_trace((uint32_t)(arg->scene_addr[0] & 0xffffffffu));
  uint32_t h1 = vx_rt_trace((uint32_t)(arg->scene_addr[1] & 0xffffffffu));
  uint32_t h2 = vx_rt_trace((uint32_t)(arg->scene_addr[2] & 0xffffffffu));
  uint32_t h3 = vx_rt_trace((uint32_t)(arg->scene_addr[3] & 0xffffffffu));

  // Drain in declared order. Each WAIT blocks until its TERMINAL
  // delivers and applies the matching hit attrs into the regfile;
  // vx_rt_get_after gates on WAIT's rd so the read sees the post-
  // apply_response state.
  uint32_t s0 = vx_rt_wait(h0);
  uint32_t t0 = vx_rt_get_after(VX_RT_HIT_T, s0);
  uint32_t s1 = vx_rt_wait(h1);
  uint32_t t1 = vx_rt_get_after(VX_RT_HIT_T, s1);
  uint32_t s2 = vx_rt_wait(h2);
  uint32_t t2 = vx_rt_get_after(VX_RT_HIT_T, s2);
  uint32_t s3 = vx_rt_wait(h3);
  uint32_t t3 = vx_rt_get_after(VX_RT_HIT_T, s3);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[tid].rays[0].status = s0;
  *(uint32_t*)&results[tid].rays[0].hit_t = t0;
  results[tid].rays[1].status = s1;
  *(uint32_t*)&results[tid].rays[1].hit_t = t1;
  results[tid].rays[2].status = s2;
  *(uint32_t*)&results[tid].rays[2].hit_t = t2;
  results[tid].rays[3].status = s3;
  *(uint32_t*)&results[tid].rays[3].hit_t = t3;
}
