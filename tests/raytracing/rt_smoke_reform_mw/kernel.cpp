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
// PRISM RTU reformation multi-warp smoke kernel — Phase 3-A2.
//
// Launched as num_warps blocks × lanes_per_warp threads. The CTA
// dispatcher places each block on its own warp, so the host can drive
// two (or more) warps into vx_rt_wait simultaneously. Both warps fire
// the same ray at the same non-opaque tri → the per-core ahs_queue_
// holds entries from BOTH warps at the same time. Reformation must
// keep them separated by warp_id (no cross-warp CB_YIELD).

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__attribute__((naked, used))
static void rt_dispatcher_accept(void) {
  __asm__ volatile (
    "li t0, %0\n"
    ".insn r %1, 6, 0, x0, t0, x0\n"
    "mret\n"
    :: "i"(VX_RT_CB_ACCEPT), "i"(0x2b)
  );
}

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= arg->total_lanes) return;

  csr_write(0x305, (uintptr_t)&rt_dispatcher_accept);

  vx_ray_t ray = {
    { arg->ray_origin[0],    arg->ray_origin[1],    arg->ray_origin[2] },
    { arg->ray_direction[0], arg->ray_direction[1], arg->ray_direction[2] },
    arg->tmin, arg->tmax
  };

  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t h   = vx_rt_wtrace(scene_lo, 0u, 0u, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[tid].status            = sts;
  results[tid].hit_t             = hit.t;
  results[tid].hit_u             = hit.u;
  results[tid].hit_v             = hit.v;
  results[tid].primitive_id      = hit.primitive_id;
  results[tid].pad               = 0;
}
