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

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

// One ray per thread against one scene: every hit is opaque, so the trace
// ends on the committed hit.
__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= arg->count) return;

  const tie_ray_t& r = ((const tie_ray_t*)((uintptr_t)arg->rays_addr))[i];
  vx_ray_t ray = { {r.origin[0], r.origin[1], r.origin[2]},
                   {r.dir[0], r.dir[1], r.dir[2]},
                   r.tmin, r.tmax };
  uint32_t h = vx_rt_wtrace(arg->scene, 0u, VX_RT_FLAG_OPAQUE, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);

  tie_result_t* res = (tie_result_t*)((uintptr_t)arg->results_addr) + i;
  res->status      = sts;
  res->t           = __builtin_bit_cast(uint32_t, hit.t);
  res->u           = __builtin_bit_cast(uint32_t, hit.u);
  res->v           = __builtin_bit_cast(uint32_t, hit.v);
  res->prim        = hit.primitive_id;
  res->geom        = hit.geometry_index | (hit.back_facing ? VX_RT_HIT_BACK_FACING : 0u);
  res->inst_id     = hit.instance_id;
  res->inst_custom = hit.instance_custom;
}
