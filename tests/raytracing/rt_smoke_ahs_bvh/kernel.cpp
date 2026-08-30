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
// PRISM RTU AHS-callback smoke kernel — candidate-return loop.
//
// Each CTA: load ray descriptor, vx_rt_wtrace + vx_rt_wait, then loop
// on the candidate-return status. A non-opaque triangle candidate is
// returned to the warp (YIELD_ANYHIT); the kernel picks ACCEPT or
// IGNORE from cb_decision and resumes traversal with vx_rt_continue.
// When the status is terminal the loop exits and the hit attrs are
// written out.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid >= arg->num_lanes) return;

  // Per-candidate action: ACCEPT or IGNORE based on cb_decision.
  uint32_t action = (arg->cb_decision == RTU_AHS_DECISION_ACCEPT)
                        ? VX_RT_CB_ACCEPT
                        : VX_RT_CB_IGNORE;

  // Assemble the ray descriptor.
  vx_ray_t ray = {
    {arg->ray_origin[0], arg->ray_origin[1], arg->ray_origin[2]},
    {arg->ray_direction[0], arg->ray_direction[1], arg->ray_direction[2]},
    arg->tmin,
    arg->tmax,
  };

  // Fire ray + wait for terminal. Ray flags = 0 (no OPAQUE override;
  // per-triangle flags drive AHS). No payload.
  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t h   = vx_rt_wtrace(scene_lo, 0u, 0u, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);
  while (vx_rt_sts_is_yield(sts)) {
    sts = vx_rt_continue(h, action, hit.t, 0u, &hit);
  }

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[tid].status       = sts;
  results[tid].hit_t        = hit.t;
  results[tid].hit_u        = hit.u;
  results[tid].hit_v        = hit.v;
  results[tid].primitive_id = hit.primitive_id;
  results[tid].pad          = 0;
}
