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
// PRISM RTU Closest-Hit Shader smoke kernel — candidate-return loop.
//
// With CHS enabled the closest opaque hit is returned to the warp as a
// candidate. The kernel reads the committed hit_t bits and the payload
// pointer from the register window, writes MAGIC ^ hit_t to the payload,
// and resumes with vx_rt_continue(CB_DONE) — the CHS is done shading the
// already-committed hit, so DONE just drains to a terminal status.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  // Assemble the ray descriptor.
  vx_ray_t ray = {
    {arg->ray_origin[0], arg->ray_origin[1], arg->ray_origin[2]},
    {arg->ray_direction[0], arg->ray_direction[1], arg->ray_direction[2]},
    arg->tmin,
    arg->tmax,
  };

  // The trace stages the payload pointer the CHS loop body reads via
  // vx_gfx_get. Opt into CHS dispatch.
  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t payload  = (uint32_t)(arg->payload_addr & 0xffffffffu);
  uint32_t h = vx_rt_wtrace(scene_lo, payload, VX_RT_FLAG_ENABLE_CHS, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);
  while (vx_rt_sts_is_yield(sts)) {
    // Read the committed hit_t bits and the payload pointer from the window,
    // write MAGIC ^ hit_t, then resume: DONE drains the CHS-shaded hit.
    uint32_t hit_t_bits  = vx_rt_get_attr(VX_RT_HIT_T, sts);
    uint32_t payload_ptr = vx_rt_get_attr(VX_RT_PAYLOAD_PTR_LO, sts);
    *(uint32_t*)(uintptr_t)payload_ptr = RTU_CHS_MAGIC ^ hit_t_bits;
    sts = vx_rt_continue(h, VX_RT_CB_DONE, hit.t, 0u, &hit);
  }

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].status            = sts;
  results[0].hit_t             = hit.t;
  results[0].primitive_id      = hit.primitive_id;
  // The CHS store landed in program order in the loop above, so this load
  // observes it.
  uint32_t chs_payload = *(volatile uint32_t*)(uintptr_t)arg->payload_addr;
  results[0].chs_payload       = chs_payload;
}
