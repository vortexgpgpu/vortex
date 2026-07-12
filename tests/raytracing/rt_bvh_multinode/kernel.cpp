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
// PRISM RTU multi-node BVH kernel — one thread per ray. Each ray is traced
// against a host-built (vortex::raytrace::build_bvh_scene<4>) multi-level
// CW-BVH4 scene, exercising real internal-node traversal (box tests, ordered
// descent) rather than the single-leaf scan.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid >= arg->num_rays) return;

  const float* r = (const float*)((uintptr_t)arg->rays_addr) + tid * 8;
  vx_ray_t ray = { { r[0], r[1], r[2] },
                   { r[3], r[4], r[5] },
                   r[6], r[7] };

  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t h   = vx_rt_wtrace(scene_lo, 0u, VX_RT_FLAG_OPAQUE, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[tid].status       = sts;
  results[tid].hit_t        = hit.t;
  results[tid].hit_u        = hit.u;
  results[tid].hit_v        = hit.v;
  results[tid].primitive_id = hit.primitive_id;
  results[tid].pad          = 0;
}
