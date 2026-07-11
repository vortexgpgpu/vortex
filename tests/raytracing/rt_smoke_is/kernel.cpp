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
// PRISM RTU Intersection Shader smoke kernel — candidate-return loop.
//
// A procedural primitive returns a candidate to the warp (YIELD_PROC). The
// kernel reads VX_RT_CB_TYPE from the register window and writes MAGIC to the
// payload when it is VX_RT_CB_TYPE_PROC (else ~MAGIC as a sentinel), then
// resumes with vx_rt_continue(CB_ACCEPT).

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  vx_ray_t ray = {
    { arg->ray_origin[0],    arg->ray_origin[1],    arg->ray_origin[2] },
    { arg->ray_direction[0], arg->ray_direction[1], arg->ray_direction[2] },
    arg->tmin, arg->tmax
  };

  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t payload  = (uint32_t)(arg->payload_addr & 0xffffffffu);
  uint32_t h   = vx_rt_wtrace(scene_lo, payload, 0u, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);
  while (vx_rt_sts_is_yield(sts)) {
    // Read the candidate's callback type + payload pointer from the window.
    // Write MAGIC for a procedural candidate, a ~MAGIC sentinel otherwise.
    uint32_t cb_type     = vx_gfx_get_dep(VX_RT_CB_TYPE, sts);
    uint32_t payload_ptr = vx_gfx_get_dep(VX_RT_PAYLOAD_PTR_LO, sts);
    uint32_t val = (cb_type == VX_RT_CB_TYPE_PROC) ? (uint32_t)RTU_IS_MAGIC
                                                   : ~(uint32_t)RTU_IS_MAGIC;
    *(uint32_t*)(uintptr_t)payload_ptr = val;
    sts = vx_rt_continue(h, VX_RT_CB_ACCEPT, &hit);
  }

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].status            = sts;
  results[0].hit_t             = hit.t;
  // The IS store landed in program order in the loop above, so this load
  // observes it.
  uint32_t is_payload = *(volatile uint32_t*)(uintptr_t)arg->payload_addr;
  results[0].is_payload        = is_payload;
  results[0].pad               = 0;
}
