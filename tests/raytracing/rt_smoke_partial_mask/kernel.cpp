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
// RTU partial-warp trace smoke kernel: an all-lane trace of the empty scene,
// then a trace of the triangle scene from the warp's last lane alone.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

static uint32_t trace_one(uint32_t scene, float* t_out) {
  vx_ray_t ray = {{0.25f, 0.25f, 0.f}, {0.f, 0.f, 1.f}, 0.001f, 1e30f};
  uint32_t h = vx_rt_wtrace(scene, 0u, 0u, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);
  while (vx_rt_sts_is_yield(sts))
    sts = vx_rt_continue(h, VX_RT_CB_ACCEPT, hit.t, 0u, &hit);
  *t_out = hit.t;
  return sts;
}

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = threadIdx.x;
  if (tid >= arg->num_lanes) return;
  rtu_result_t* res = (rtu_result_t*)((uintptr_t)arg->results_addr) + tid;

  float t;
  res->first_status = trace_one((uint32_t)arg->empty_scene_addr, &t);
  res->second_status = 0xffffffffu;
  if (tid == arg->num_lanes - 1) {
    res->second_status = trace_one((uint32_t)arg->tri_scene_addr, &t);
    res->second_t = t;
  }
}
