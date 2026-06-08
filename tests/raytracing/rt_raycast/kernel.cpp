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
// rt_raycast kernel — one primary ray per pixel, intersected on the RTU.

#include <vx_spawn2.h>
#include <vx_raytrace.h>
#include "common.h"

static vec3 shade_hit(const kernel_arg_t* arg, vec3 ro, vec3 rd,
                      float t, float u, float v, uint32_t tri_id) {
  const tri_shade_t* shade = (const tri_shade_t*)((uintptr_t)arg->shade_addr);
  tri_shade_t s = shade[tri_id];

  // Barycentric normal: rtu Möller-Trumbore returns u along (v1-v0),
  // v along (v2-v0), so weights are (1-u-v, u, v) for (n0, n1, n2).
  float w0 = 1.f - u - v;
  vec3 N = v3add(v3add(v3scale(s.n0, w0), v3scale(s.n1, u)),
                 v3scale(s.n2, v));
  N = v3norm(N);

  vec3 I = v3add(ro, v3scale(rd, t));
  vec3 L = v3norm(v3sub(arg->light_pos, I));
  float ndotl = v3dot(N, L);
  if (ndotl < 0.f) ndotl = 0.f;

  // ambient + lambertian diffuse (white albedo).
  vec3 c = v3add(arg->ambient_color, v3scale(arg->light_color, ndotl));
  return c;
}

__kernel void kernel_main(kernel_arg_t* arg) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y;
  if (x >= arg->dst_width || y >= arg->dst_height) return;

  // Perspective primary ray (same framing as regression/raycast).
  float x_ndc = (x + 0.5f) / arg->dst_width  - 0.5f;
  float y_ndc = (y + 0.5f) / arg->dst_height - 0.5f;
  float x_vp = x_ndc * arg->viewplane_x;
  float y_vp = y_ndc * arg->viewplane_y;
  vec3 pt_cam = v3add(v3add(v3scale(arg->camera_right, x_vp),
                            v3scale(arg->camera_up, y_vp)),
                      arg->camera_forward);
  vec3 ro = arg->camera_pos;
  vec3 rd = v3norm(pt_cam);  // camera_forward/right/up already in world space

  // Fire the ray on the RTU (v2 single-issue trace).
  vx_ray_t ray = { {ro.x, ro.y, ro.z}, {rd.x, rd.y, rd.z}, 0.001f, 1e30f };
  uint32_t scene_lo = (uint32_t)(arg->scene_addr & 0xffffffffu);
  uint32_t h   = vx_rt_trace2(scene_lo, 0u, VX_RT_FLAG_OPAQUE, 0xffu, &ray);
  vx_hit_t hit;
  uint32_t sts = vx_rt_wait2(h, &hit);

  uint32_t* fb = (uint32_t*)((uintptr_t)arg->dst_addr);
  vec3 color;
  if (sts == VX_RT_STS_DONE_HIT) {
    color = shade_hit(arg, ro, rd, hit.t, hit.u, hit.v, hit.geometry_index);
  } else {
    color = arg->background_color;
  }
  fb[x + y * arg->dst_width] = pack_rgb(color);
}
