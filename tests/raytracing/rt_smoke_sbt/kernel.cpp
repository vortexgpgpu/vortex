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
// PRISM RTU SBT smoke kernel — candidate-return loop.
//
// Single primitive marked PROCEDURAL with per-tri sbt_idx = 1. The kernel
// builds a 2-record SBT in memory whose IS slot holds a distinct magic per
// record (MAGIC_0, MAGIC_1). A procedural candidate is returned to the warp;
// the kernel reads (sbt_idx, cb_type) from the register window, indexes the
// SBT the same way the fixed-function lookup would (record = sbt_idx, slot =
// cb_type - 1), loads the selected record's magic, and writes it to the
// payload — proving the runtime SBT lookup (not a hardcoded dispatch)
// selected the entry.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  // Build the SBT in memory: 2 records, each holds 4 slots (one per cb_type
  // at index cb_type-1). Only the IS slot is populated, with the record's
  // magic; the other slots stay 0.
  uint32_t* sbt = (uint32_t*)(uintptr_t)arg->sbt_addr;
  for (uint32_t i = 0; i < 8; ++i) sbt[i] = 0;   // 2 records × 4 slots
  sbt[0 * 4 + 1] = RTU_SBT_MAGIC_0;   // sbt[0].is
  sbt[1 * 4 + 1] = RTU_SBT_MAGIC_1;   // sbt[1].is

  vx_ray_t ray = {
    {arg->ray_origin[0], arg->ray_origin[1], arg->ray_origin[2]},
    {arg->ray_direction[0], arg->ray_direction[1], arg->ray_direction[2]},
    arg->tmin,
    arg->tmax,
  };

  // The trace stages the payload pointer the SBT lookup reads via vx_gfx_get.
  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t payload  = (uint32_t)(arg->payload_addr & 0xffffffffu);
  uint32_t h   = vx_rt_wtrace(scene_lo, payload, 0u, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);
  while (vx_rt_sts_is_yield(sts)) {
    // Runtime SBT lookup: index the record by the candidate's sbt_idx and its
    // cb_type slot, load the selected magic, write it to the payload.
    uint32_t cb_type     = vx_gfx_get_dep(VX_RT_CB_TYPE, sts);
    uint32_t sbt_idx     = vx_gfx_get_dep(VX_RT_HIT_SBT_IDX, sts);
    uint32_t payload_ptr = vx_gfx_get_dep(VX_RT_PAYLOAD_PTR_LO, sts);
    uint32_t magic = sbt[sbt_idx * 4 + (cb_type - 1)];
    *(uint32_t*)(uintptr_t)payload_ptr = magic;
    sts = vx_rt_continue(h, VX_RT_CB_ACCEPT, &hit);
  }

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].status            = sts;
  results[0].hit_t             = hit.t;
  // The SBT-selected store landed in program order in the loop above, so this
  // load observes it.
  uint32_t sbt_payload = *(volatile uint32_t*)(uintptr_t)arg->payload_addr;
  results[0].sbt_payload       = sbt_payload;
  results[0].pad               = 0;
}
