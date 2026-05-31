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
// §8.8 instanceCullMask smoke. Three rays at the same origin/dir but
// different VX_RT_CULL_MASK values — the walker should skip instances
// whose mask doesn't overlap. Validates per-ray status, hit_t, and
// hit_instance_id.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  // Stage ray origin/direction/tmin/tmax once — only cull_mask
  // changes between the three rays.
  vx_rt_set3(VX_RT_RAY_ORIGIN,
             vx_rt_f2u(arg->ray_origin[0]),
             vx_rt_f2u(arg->ray_origin[1]),
             vx_rt_f2u(arg->ray_origin[2]));
  vx_rt_set3(VX_RT_RAY_DIRECTION,
             vx_rt_f2u(arg->ray_direction[0]),
             vx_rt_f2u(arg->ray_direction[1]),
             vx_rt_f2u(arg->ray_direction[2]));
  vx_rt_set3(VX_RT_T_MIN,
             vx_rt_f2u(arg->tmin),
             vx_rt_f2u(arg->tmax),
             0u);
  vx_rt_set1(VX_RT_RAY_FLAGS, VX_RT_FLAG_OPAQUE);

  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);

  // Each ray: set CULL_MASK, fire+wait, read results. We sequence
  // the three rays serially (set→trace→wait→get) rather than
  // batching, because each ray needs a distinct CULL_MASK staged
  // before its TRACE — and CULL_MASK is shared per-(warp,lane)
  // regfile state.
  for (uint32_t i = 0; i < RTU_NUM_RAYS; ++i) {
    vx_rt_set1(VX_RT_CULL_MASK, arg->ray_cull_mask[i]);
    uint32_t h   = vx_rt_trace(scene_lo);
    uint32_t sts = vx_rt_wait(h);
    uint32_t hit_t_bits     = vx_rt_get_after(VX_RT_HIT_T,           sts);
    uint32_t instance_id    = vx_rt_get_after(VX_RT_HIT_INSTANCE_ID, sts);
    results[tid].rays[i].status          = sts;
    *(uint32_t*)&results[tid].rays[i].hit_t = hit_t_bits;
    results[tid].rays[i].hit_instance_id = instance_id;
    results[tid].rays[i].pad             = 0;
  }
}
