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
// PRISM RTU smoke kernel: one trace per lane, every candidate accepted at the
// intersection shader's t.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t lane = threadIdx.x;
  if (lane >= RTU_NUM_LANES) {
    return;
  }

  rtu_result_t* r = (rtu_result_t*)(uintptr_t)arg->results_addr + lane;
  vx_ray_t ray = { { arg->ray_x[lane], 0.25f, 0.0f }, { 0.0f, 0.0f, 1.0f }, 0.001f, 1e30f };
  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t h = vx_rt_wtrace(scene_lo, 0u, 0u, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);
  r->first_status = sts;
  r->cand_geom = 0;
  uint32_t n = 0;
  while (vx_rt_sts_is_yield(sts)) {
    if (n == 0) {
      r->cand_geom = vx_rt_get_attr(VX_RT_HIT_GEOMETRY_INDEX, sts);
    }
    ++n;
    sts = vx_rt_continue(h, VX_RT_CB_ACCEPT, RTU_IS_T, 0u, &hit);
  }
  r->num_cand = n;
  r->status   = sts;
  r->hit_t    = hit.t;
  r->geom     = vx_rt_get_attr(VX_RT_HIT_GEOMETRY_INDEX, sts);
  r->inst     = hit.instance_id;
}
