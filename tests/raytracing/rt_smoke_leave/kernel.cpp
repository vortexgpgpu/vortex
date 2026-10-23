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
// PRISM RTU smoke kernel: lane 1 leaves its loop holding a candidate.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t lane = threadIdx.x;
  if (lane >= RTU_NUM_LANES) {
    return;
  }

  rtu_result_t* r = (rtu_result_t*)(uintptr_t)arg->results_addr + lane;
  vx_ray_t ray = { { 0.25f, 0.25f, 0.0f }, { 0.0f, 0.0f, 1.0f }, 0.001f, 1e30f };
  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t h = vx_rt_wtrace(scene_lo, 0u, 0u, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);
  uint32_t n = 0;
  float first_t = 0.0f;
  while (vx_rt_sts_is_yield(sts)) {
    if (vx_rt_sts_has_candidate(sts)) {
      if (n == 0) {
        first_t = hit.t;
      }
      ++n;
      if (lane == 1) {
        break;
      }
    }
    sts = vx_rt_continue(h, VX_RT_CB_IGNORE, hit.t, 0u, &hit);
  }
  r->num_cand     = n;
  r->first_cand_t = first_t;
  r->exit_status  = sts;
  uint32_t live   = vx_rt_get_attr(VX_RT_STATUS, sts);
  r->live_status  = live;
  r->live_t       = vx_rt_get_attr(VX_RT_HIT_T, live);
}
