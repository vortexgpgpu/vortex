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
// PRISM RTU procedural intersection-shader smoke kernel (ray-sphere).

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

static inline float u2f(uint32_t u) { float f; __builtin_memcpy(&f, &u, 4); return f; }
static inline uint32_t f2u(float f) { uint32_t u; __builtin_memcpy(&u, &f, 4); return u; }

// Intersection-shader dispatcher (mtvec target). Uses the RISC-V
// machine-interrupt attribute so the compiler emits a full caller-saved
// context save/restore + mret epilogue, letting the IS run real
// floating-point work. Reads the object-space ray the RTU staged into
// VX_RT_OBJECT_RAY_* (feature 1), does the ray-sphere test, and on a hit
// writes the computed VX_RT_HIT_T + a hitAttribute sentinel (feature 2)
// before ACCEPTing.
__attribute__((interrupt("machine"), used))
void rt_is_dispatcher(void) {
  float ox = u2f(vx_rt_get(VX_RT_OBJECT_RAY_ORIGIN + 0));
  float oy = u2f(vx_rt_get(VX_RT_OBJECT_RAY_ORIGIN + 1));
  float oz = u2f(vx_rt_get(VX_RT_OBJECT_RAY_ORIGIN + 2));
  float dx = u2f(vx_rt_get(VX_RT_OBJECT_RAY_DIRECTION + 0));
  float dy = u2f(vx_rt_get(VX_RT_OBJECT_RAY_DIRECTION + 1));
  float dz = u2f(vx_rt_get(VX_RT_OBJECT_RAY_DIRECTION + 2));

  // |o + t d - C|^2 = r^2  →  a t^2 + b t + c = 0
  float ocx = ox - RTU_SPHERE_CX, ocy = oy - RTU_SPHERE_CY, ocz = oz - RTU_SPHERE_CZ;
  float a = dx*dx + dy*dy + dz*dz;
  float b = 2.0f * (ocx*dx + ocy*dy + ocz*dz);
  float c = ocx*ocx + ocy*ocy + ocz*ocz - RTU_SPHERE_R*RTU_SPHERE_R;
  float disc = b*b - 4.0f*a*c;

  if (disc < 0.0f) {
    vx_rt_cb_ret(VX_RT_CB_IGNORE);
    return;
  }
  float t = (-b - __builtin_sqrtf(disc)) / (2.0f * a);   // near root
  vx_rt_set1(VX_RT_HIT_T,      f2u(t));
  vx_rt_set1(VX_RT_HIT_ATTR_0, RTU_IS_ATTR_MAGIC);
  vx_rt_cb_ret(VX_RT_CB_ACCEPT);
}

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t tid = blockIdx.x;
  if (tid != 0) return;

  // Register the IS dispatcher as the M-mode trap handler (mtvec).
  csr_write(0x305, (uintptr_t)&rt_is_dispatcher);

  vx_rt_set3(VX_RT_RAY_ORIGIN,
             vx_rt_f2u(arg->ray_origin[0]),
             vx_rt_f2u(arg->ray_origin[1]),
             vx_rt_f2u(arg->ray_origin[2]));
  vx_rt_set3(VX_RT_RAY_DIRECTION,
             vx_rt_f2u(arg->ray_direction[0]),
             vx_rt_f2u(arg->ray_direction[1]),
             vx_rt_f2u(arg->ray_direction[2]));
  vx_rt_set3(VX_RT_T_MIN,
             vx_rt_f2u(arg->tmin),
             vx_rt_f2u(arg->tmax),
             0u);
  vx_rt_set1(VX_RT_RAY_FLAGS, 0u);   // procedural primitive → IS decides

  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t h   = vx_rt_trace(scene_lo);
  uint32_t sts = vx_rt_wait(h);

  uint32_t hit_t_bits = vx_rt_get_after(VX_RT_HIT_T,      sts);
  uint32_t hit_attr   = vx_rt_get_after(VX_RT_HIT_ATTR_0, sts);

  rtu_result_t* results = (rtu_result_t*)((uintptr_t)arg->results_addr);
  results[0].status              = sts;
  *(uint32_t*)&results[0].hit_t  = hit_t_bits;
  results[0].hit_attr            = hit_attr;
  results[0].pad                 = 0;
}
