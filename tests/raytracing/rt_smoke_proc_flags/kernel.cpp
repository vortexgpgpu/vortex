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
// PRISM RTU procedural-opacity smoke kernel: one trace per case, every
// candidate answered with the case's verdict (a box at the shader's t).

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  if (blockIdx.x != 0) {
    return;
  }

  const rtu_case_t* cases = (const rtu_case_t*)(uintptr_t)arg->cases_addr;
  rtu_result_t* results   = (rtu_result_t*)(uintptr_t)arg->results_addr;
  vx_ray_t ray = { { 0.25f, 0.25f, 0.0f }, { 0.0f, 0.0f, 1.0f }, 0.001f, 1e30f };

  for (uint32_t c = 0; c < arg->num_cases; ++c) {
    uint32_t scene_lo = (uint32_t)(arg->scene_addr[cases[c].scene] & 0xffffffffu);
    uint32_t h = vx_rt_wtrace(scene_lo, 0u, cases[c].ray_flags, 0xffu, &ray);
    vx_hit_t hit;
    uint32_t sts = vx_rt_wait(h, &hit);
    rtu_result_t* r = &results[c];
    r->first_status = sts;
    r->cand_geom = 0;
    uint32_t n = 0;
    while (vx_rt_sts_is_yield(sts)) {
      if (n == 0) {
        r->cand_geom = vx_rt_get_attr(VX_RT_HIT_GEOMETRY_INDEX, sts);
      }
      float t = (sts == VX_RT_STS_YIELD_PROC) ? RTU_IS_T : hit.t;
      ++n;
      sts = vx_rt_continue(h, cases[c].action, t, 0u, &hit);
    }
    r->num_cand = n;
    r->status   = sts;
    r->hit_t    = hit.t;
    r->geom     = vx_rt_get_attr(VX_RT_HIT_GEOMETRY_INDEX, sts);
  }
}
