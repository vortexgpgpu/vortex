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
// rt_replay kernel — one recorded ray per thread. The host pads every
// warp-sized group to one (scene, flags, cull), so the trace config is uniform.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= arg->count) return;

  const replay_ray_t* rays = (const replay_ray_t*)((uintptr_t)arg->rays_addr);
  const replay_ray_t& r = rays[i];
  vx_ray_t ray = { {r.origin[0], r.origin[1], r.origin[2]},
                   {r.dir[0], r.dir[1], r.dir[2]},
                   r.tmin, r.tmax };
  uint32_t h = vx_rt_wtrace(r.scene, 0u, r.flags, r.cull, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);
  uint32_t yielded = 0;
  // A replayed ray was never decided by a shader; a candidate here is itself a
  // divergence. Resolve it as a closest-hit/miss dispatch would, and flag it.
  while (vx_rt_sts_is_yield(sts)) {
    yielded = REPLAY_STS_YIELDED;
    sts = vx_rt_continue(h, VX_RT_CB_IGNORE, hit.t, 0u, &hit);
  }

  replay_result_t* res = (replay_result_t*)((uintptr_t)arg->results_addr) + i;
  res->status      = sts | yielded;
  res->t           = __builtin_bit_cast(uint32_t, hit.t);
  res->u           = __builtin_bit_cast(uint32_t, hit.u);
  res->v           = __builtin_bit_cast(uint32_t, hit.v);
  res->prim        = hit.primitive_id;
  res->geom        = hit.geometry_index | (hit.back_facing ? VX_RT_HIT_BACK_FACING : 0u);
  res->inst_id     = hit.instance_id;
  res->inst_custom = hit.instance_custom;
}
