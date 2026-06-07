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
// PRISM RTU smoke test — ISA ABI v2 kernel (rtu_isa_v2_proposal.md).
//
// Each lane: assemble the per-thread ray, issue ONE trace (config lane-packed
// into rs1, ray in the f0..f7 window) + ONE wait (hit attrs returned in
// registers). The whole ~16-op vx_rt_set/get marshalling collapses to two
// architectural instructions. CPU oracle compares.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  // Global thread id = (CTA index × CTA size) + lane-within-CTA.
  // Host launches with block_dim = num_threads_per_warp (queried via
  // VX_CAPS_NUM_THREADS), grid_dim = ceil(num_lanes / block_dim) so
  // every CTA fills exactly one warp. Lanes past num_lanes mask off.
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= arg->num_lanes) return;

  // The per-thread ray — the divergent operand the f0..f7 window carries.
  vx_ray_t ray;
  ray.origin[0] = arg->ray_origin[0];
  ray.origin[1] = arg->ray_origin[1];
  ray.origin[2] = arg->ray_origin[2];
  ray.dir[0]    = arg->ray_direction[0];
  ray.dir[1]    = arg->ray_direction[1];
  ray.dir[2]    = arg->ray_direction[2];
  ray.tmin      = arg->tmin;
  ray.tmax      = arg->tmax;

  // One trace + one wait. cull_mask = 0xff = no culling (matches the
  // Phase-1 default). payload = 0 (smoke test reads no payload).
  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t h = vx_rt_trace2(scene_lo, 0u, VX_RT_FLAG_OPAQUE, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait2(h, &hit);

  // Store per-lane result.
  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[tid].status       = sts;
  results[tid].hit_t        = hit.t;
  results[tid].hit_u        = hit.u;
  results[tid].hit_v        = hit.v;
  results[tid].primitive_id = hit.primitive_id;
  results[tid].pad          = 0;
}
