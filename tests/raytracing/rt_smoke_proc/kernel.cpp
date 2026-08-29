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
// PRISM RTU procedural intersection-shader smoke kernel (ray-sphere) —
// candidate-return loop.
//
// A procedural primitive returns a candidate to the warp (YIELD_PROC).
// The kernel reads the object-space ray the RTU staged into
// VX_RT_OBJECT_RAY_*, runs the ray-sphere test, and on a hit stages the
// computed VX_RT_HIT_T + a hitAttribute sentinel before ACCEPTing (or
// IGNOREs on a miss). vx_rt_continue resumes traversal until a terminal
// status ends it.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  vx_ray_t ray = {
    { arg->ray_origin[0],    arg->ray_origin[1],    arg->ray_origin[2] },
    { arg->ray_direction[0], arg->ray_direction[1], arg->ray_direction[2] },
    arg->tmin, arg->tmax
  };

  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  // procedural primitive → IS decides (flags = 0), no payload.
  uint32_t h   = vx_rt_wtrace(scene_lo, 0u, 0u, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait(h, &hit);
  while (vx_rt_sts_is_yield(sts)) {
    uint32_t action;
    float    hit_t = 0.0f;
    uint32_t attr  = 0;
    // one windowed read pulls the whole object-space ray into the f0..f5 FP window
    vx_objray_t objray;
    vx_rt_get_objray(&objray);
    float ox = objray.origin[0], oy = objray.origin[1], oz = objray.origin[2];
    float dx = objray.dir[0],    dy = objray.dir[1],    dz = objray.dir[2];

    // |o + t d - C|^2 = r^2  →  a t^2 + b t + c = 0
    float ocx = ox - RTU_SPHERE_CX, ocy = oy - RTU_SPHERE_CY, ocz = oz - RTU_SPHERE_CZ;
    float a = dx*dx + dy*dy + dz*dz;
    float b = 2.0f * (ocx*dx + ocy*dy + ocz*dz);
    float c = ocx*ocx + ocy*ocy + ocz*ocz - RTU_SPHERE_R*RTU_SPHERE_R;
    float disc = b*b - 4.0f*a*c;

    if (disc < 0.0f) {
      action = VX_RT_CB_IGNORE;
    } else {
      hit_t  = (-b - __builtin_sqrtf(disc)) / (2.0f * a);   // near root
      attr   = RTU_IS_ATTR_MAGIC;
      action = VX_RT_CB_ACCEPT;
    }
    // the verdict carries its own t and attribute -- no RTU state written here
    sts = vx_rt_continue(h, action, hit_t, attr, &hit);
  }

  uint32_t hit_attr   = vx_rt_get_attr(VX_RT_HIT_ATTR_0, sts);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].status              = sts;
  results[0].hit_t               = hit.t;
  results[0].hit_attr            = hit_attr;
  results[0].pad                 = 0;
}
