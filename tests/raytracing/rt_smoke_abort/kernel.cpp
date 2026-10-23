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
// PRISM RTU verdict smoke kernel: one two-lane trace per case, lane 0's
// candidates answered from the case's verdict script.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t lane = threadIdx.x;
  if (lane >= RTU_NUM_LANES) {
    return;
  }

  const rtu_case_t* cases = (const rtu_case_t*)(uintptr_t)arg->cases_addr;
  rtu_result_t* results   = (rtu_result_t*)(uintptr_t)arg->results_addr;

  vx_ray_t ray = {
    { arg->ray_origin[lane][0], arg->ray_origin[lane][1], arg->ray_origin[lane][2] },
    { 0.0f, 0.0f, 1.0f },
    0.001f, 1e30f
  };

  for (uint32_t c = 0; c < arg->num_cases; ++c) {
    uint32_t scene_lo = (uint32_t)(arg->scene_addr[cases[c].scene] & 0xffffffffu);
    uint32_t h = vx_rt_wtrace(scene_lo, 0u, 0u, 0xffu, &ray);
    vx_hit_t hit;
    uint32_t sts = vx_rt_wait(h, &hit);
    uint32_t first = sts;
    uint32_t n = 0;
    float first_t = 0.0f;
    while (vx_rt_sts_is_yield(sts)) {
      uint32_t action = VX_RT_CB_IGNORE;
      if (vx_rt_sts_has_candidate(sts)) {
        if (n == 0) {
          first_t = hit.t;
        }
        if (n < RTU_MAX_STEPS) {
          action = cases[c].action[n];
        }
        ++n;
      }
      sts = vx_rt_continue(h, action, hit.t, 0u, &hit);
    }
    rtu_result_t* r = &results[c * RTU_NUM_LANES + lane];
    r->first_status = first;
    r->num_cand     = n;
    r->first_cand_t = first_t;
    r->status       = sts;
    r->hit_t        = hit.t;
    r->geom         = vx_rt_get_attr(VX_RT_HIT_GEOMETRY_INDEX, sts);
  }
}
