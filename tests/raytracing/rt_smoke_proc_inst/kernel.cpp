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
// Procedural candidates inside instances: the intersection shader must see the
// candidate's own gl_InstanceID / gl_InstanceCustomIndexEXT, and the committed
// hit must carry them too. One ray per lane, each through a different instance.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = threadIdx.x;
  if (tid >= NUM_RAYS) return;

  vx_ray_t ray = {
    { arg->ray_origin[tid][0], arg->ray_origin[tid][1], arg->ray_origin[tid][2] },
    { arg->ray_direction[0],   arg->ray_direction[1],   arg->ray_direction[2] },
    arg->tmin, arg->tmax
  };

  uint32_t h = vx_rt_wtrace((uint32_t)arg->scene_addr, 0u, 0u, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);
  uint32_t is_calls = 0, is_inst = ~0u, is_cust = ~0u;
  vx_objray_t o = {};
  while (vx_rt_sts_is_yield(sts)) {
    uint32_t action = VX_RT_CB_IGNORE;
    float    hit_t  = 0.0f;
    if (sts == VX_RT_STS_YIELD_PROC) {
      ++is_calls;
      is_inst = hit.instance_id;
      is_cust = hit.instance_custom;
      vx_rt_get_objray(&o);
      // |o + t d - C|^2 = r^2 with C = (0,0,CZ)
      float ocz  = o.origin[2] - RTU_SPHERE_CZ;
      float a    = o.dir[0]*o.dir[0] + o.dir[1]*o.dir[1] + o.dir[2]*o.dir[2];
      float b    = 2.0f * (o.origin[0]*o.dir[0] + o.origin[1]*o.dir[1] + ocz*o.dir[2]);
      float c    = o.origin[0]*o.origin[0] + o.origin[1]*o.origin[1] + ocz*ocz
                 - RTU_SPHERE_R*RTU_SPHERE_R;
      float disc = b*b - 4.0f*a*c;
      if (disc >= 0.0f) {
        hit_t  = (-b - __builtin_sqrtf(disc)) / (2.0f * a);
        action = VX_RT_CB_ACCEPT;
      }
    }
    sts = vx_rt_continue(h, action, hit_t, 0u, &hit);
  }

  rtu_result_t* r = (rtu_result_t*)((uintptr_t)arg->results_addr) + tid;
  r->status   = sts;
  r->hit_t    = hit.t;
  r->hit_inst = hit.instance_id;
  r->hit_cust = hit.instance_custom;
  r->is_calls = is_calls;
  r->is_inst  = is_inst;
  r->is_cust  = is_cust;
  for (int i = 0; i < 3; ++i) {
    r->obj_ray[i]     = o.origin[i];
    r->obj_ray[3 + i] = o.dir[i];
  }
}
