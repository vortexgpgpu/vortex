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
// PRISM RTU Miss Shader smoke kernel — candidate-return loop.
//
// With MISS enabled a ray that finds no hit returns a miss candidate to the
// warp. The kernel reads the payload pointer from the register window, writes
// MAGIC, and resumes with vx_rt_continue(CB_DONE), which drains to a terminal
// status.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  vx_ray_t ray = {
    {arg->ray_origin[0], arg->ray_origin[1], arg->ray_origin[2]},
    {arg->ray_direction[0], arg->ray_direction[1], arg->ray_direction[2]},
    arg->tmin,
    arg->tmax,
  };

  // The trace stages the payload pointer the MISS loop body reads via
  // vx_gfx_get. Opt into MISS dispatch.
  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t payload  = (uint32_t)(arg->payload_addr & 0xffffffffu);
  uint32_t h   = vx_rt_wtrace(scene_lo, payload, VX_RT_FLAG_ENABLE_MISS, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);
  while (vx_rt_sts_is_yield(sts)) {
    // Read the payload pointer from the window, write MAGIC, then resume.
    uint32_t payload_ptr = vx_gfx_get_dep(VX_RT_PAYLOAD_PTR_LO, sts);
    *(uint32_t*)(uintptr_t)payload_ptr = RTU_MISS_MAGIC;
    sts = vx_rt_continue(h, VX_RT_CB_DONE, &hit);
  }

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].status            = sts;
  results[0].hit_t             = hit.t;
  // The MISS store landed in program order in the loop above, so this load
  // observes it.
  uint32_t miss_payload = *(volatile uint32_t*)(uintptr_t)arg->payload_addr;
  results[0].miss_payload      = miss_payload;
  results[0].pad               = 0;
}
