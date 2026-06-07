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
// §8.6 async batch smoke kernel — ISA ABI v2. Issues RTU_ASYNC_NUM_BATCH
// back-to-back vx_rt_trace2 calls — each one targets a different scene with
// a triangle at a unique z — then drains the handles with the matching
// vx_rt_wait2. If any of the WAIT-handle ↔ TERMINAL routings is wrong,
// hit_t for one or more rays will be off (or the same value will appear for
// multiple rays).

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid >= arg->num_lanes) return;

  // Stage the ray descriptor once — all rays share origin/dir; only
  // the scene (and thus the target triangle's z) differs across the
  // batch. tmin/tmax also identical.
  vx_ray_t ray;
  ray.origin[0] = arg->ray_origin[0];
  ray.origin[1] = arg->ray_origin[1];
  ray.origin[2] = arg->ray_origin[2];
  ray.dir[0]    = arg->ray_direction[0];
  ray.dir[1]    = arg->ray_direction[1];
  ray.dir[2]    = arg->ray_direction[2];
  ray.tmin      = arg->tmin;
  ray.tmax      = arg->tmax;

  // Issue RTU_ASYNC_NUM_BATCH traces back-to-back. No WAIT in
  // between — each ray runs async in the cluster's RtuCore.
  uint32_t h0 = vx_rt_trace2((uint32_t)(arg->scene_addr[0] & 0xffffffffu), 0u, VX_RT_FLAG_OPAQUE, 0xffu, &ray);
  uint32_t h1 = vx_rt_trace2((uint32_t)(arg->scene_addr[1] & 0xffffffffu), 0u, VX_RT_FLAG_OPAQUE, 0xffu, &ray);
  uint32_t h2 = vx_rt_trace2((uint32_t)(arg->scene_addr[2] & 0xffffffffu), 0u, VX_RT_FLAG_OPAQUE, 0xffu, &ray);
  uint32_t h3 = vx_rt_trace2((uint32_t)(arg->scene_addr[3] & 0xffffffffu), 0u, VX_RT_FLAG_OPAQUE, 0xffu, &ray);

  // Drain in declared order. Each WAIT blocks until its TERMINAL
  // delivers and applies the matching hit attrs.
  vx_hit_t hit0, hit1, hit2, hit3;
  uint32_t s0 = vx_rt_wait2(h0, &hit0);
  uint32_t s1 = vx_rt_wait2(h1, &hit1);
  uint32_t s2 = vx_rt_wait2(h2, &hit2);
  uint32_t s3 = vx_rt_wait2(h3, &hit3);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[tid].rays[0].status = s0;
  results[tid].rays[0].hit_t  = hit0.t;
  results[tid].rays[1].status = s1;
  results[tid].rays[1].hit_t  = hit1.t;
  results[tid].rays[2].status = s2;
  results[tid].rays[2].hit_t  = hit2.t;
  results[tid].rays[3].status = s3;
  results[tid].rays[3].hit_t  = hit3.t;
}
