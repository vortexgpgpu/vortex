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
// PRISM RTU reformation divergent-SBT smoke kernel — candidate-return loop.
//
// All lanes trace ONE shared (warp-uniform) scene with vx_rt_wtrace; lane i
// aims a +z ray at tri i, so each lane gets a distinct sbt_idx from the tri it
// hits. Each yielded candidate is returned to the warp; the kernel reads
// VX_RT_HIT_SBT_IDX per lane and branches: sbt 0 -> ACCEPT, else IGNORE. The
// reformation engine narrows each CB_YIELD's tmask to lanes that share an sbt,
// so the per-lane SBT branch is SIMT-coherent even though it is data-dependent
// across lanes.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = threadIdx.x;
  if (tid >= arg->num_lanes) return;

  // One shared (warp-uniform) scene; the per-lane ray aims at tri `tid`, so
  // each lane gets a distinct sbt_idx from the tri it hits — divergence rides
  // the ray, not the scene pointer.
  uint32_t scene_addr = (uint32_t)(arg->scene_base_addr & 0xffffffffu);

  float ox = (float)tid * RTU_TRI_SPACING + RTU_RAY_XOFF;
  vx_ray_t ray = {
    { ox, RTU_RAY_Y, 0.f },
    { 0.f, 0.f, 1.f },
    arg->tmin, arg->tmax
  };

  uint32_t h   = vx_rt_wtrace(scene_addr, 0u, 0u, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);
  while (vx_rt_sts_is_yield(sts)) {
    // Per-lane: read this lane's candidate sbt_idx, ACCEPT sbt 0, else IGNORE.
    uint32_t sbt_idx = vx_gfx_get_dep(VX_RT_HIT_SBT_IDX, sts);
    uint32_t action  = (sbt_idx == 0) ? VX_RT_CB_ACCEPT : VX_RT_CB_IGNORE;
    sts = vx_rt_continue(h, action, &hit);
  }

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[tid].status            = sts;
  results[tid].hit_t             = hit.t;
  results[tid].hit_u             = hit.u;
  results[tid].hit_v             = hit.v;
  results[tid].primitive_id      = hit.primitive_id;
  results[tid].pad               = 0;
}
