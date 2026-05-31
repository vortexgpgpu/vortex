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

#include "rtu_isect.h"
#include <cmath>

namespace vortex { namespace rtu {

bool ray_triangle(const float ro[3], const float rd[3],
                  const float v0[3], const float v1[3], const float v2[3],
                  float tmin, float tmax,
                  float& out_t, float& out_u, float& out_v,
                  bool& out_back_facing) {
  Vec3 O  = { ro[0], ro[1], ro[2] };
  Vec3 D  = { rd[0], rd[1], rd[2] };
  Vec3 V0 = { v0[0], v0[1], v0[2] };
  Vec3 V1 = { v1[0], v1[1], v1[2] };
  Vec3 V2 = { v2[0], v2[1], v2[2] };

  Vec3  e1  = V1 - V0;
  Vec3  e2  = V2 - V0;
  Vec3  P   = cross(D, e2);
  float det = dot(e1, P);
  constexpr float EPS = 1e-6f;
  if (det > -EPS && det < EPS) return false;
  float invDet = 1.0f / det;
  Vec3  T = O - V0;
  float u = dot(T, P) * invDet;
  if (u < 0.f || u > 1.f) return false;
  Vec3  Q = cross(T, e1);
  float v = dot(D, Q) * invDet;
  if (v < 0.f || u + v > 1.f) return false;
  float t = dot(e2, Q) * invDet;
  if (t < tmin || t > tmax) return false;
  out_t = t;
  out_u = u;
  out_v = v;
  out_back_facing = (det < 0.f);
  return true;
}

bool ray_aabb_intersect(const float ro[3], const float rd[3],
                        const float mn[3], const float mx[3],
                        float tmin, float tmax, float& t_near) {
  float tn = tmin, tf = tmax;
  for (int i = 0; i < 3; ++i) {
    float inv = 1.0f / rd[i];
    float t0 = (mn[i] - ro[i]) * inv;
    float t1 = (mx[i] - ro[i]) * inv;
    if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
    if (t0 > tn) tn = t0;
    if (t1 < tf) tf = t1;
    if (tn > tf) return false;
  }
  t_near = tn;
  return true;
}

void affine_inverse_transform_ray(const float xform[12],
                                  const float ro[3], const float rd[3],
                                  float ro_out[3], float rd_out[3]) {
  const float r00 = xform[0],  r01 = xform[1],  r02 = xform[2],  tx = xform[3];
  const float r10 = xform[4],  r11 = xform[5],  r12 = xform[6],  ty = xform[7];
  const float r20 = xform[8],  r21 = xform[9],  r22 = xform[10], tz = xform[11];

  // det(R) by cofactor expansion along row 0.
  float det = r00 * (r11 * r22 - r12 * r21)
            - r01 * (r10 * r22 - r12 * r20)
            + r02 * (r10 * r21 - r11 * r20);
  if (det > -1e-9f && det < 1e-9f) {
    // Singular — pass through (treat as identity).
    for (int i = 0; i < 3; ++i) { ro_out[i] = ro[i]; rd_out[i] = rd[i]; }
    return;
  }
  float inv_det = 1.f / det;

  // R^(-1) = (1/det) * adj(R).
  float i00 =  (r11 * r22 - r12 * r21) * inv_det;
  float i01 = -(r01 * r22 - r02 * r21) * inv_det;
  float i02 =  (r01 * r12 - r02 * r11) * inv_det;
  float i10 = -(r10 * r22 - r12 * r20) * inv_det;
  float i11 =  (r00 * r22 - r02 * r20) * inv_det;
  float i12 = -(r00 * r12 - r02 * r10) * inv_det;
  float i20 =  (r10 * r21 - r11 * r20) * inv_det;
  float i21 = -(r00 * r21 - r01 * r20) * inv_det;
  float i22 =  (r00 * r11 - r01 * r10) * inv_det;

  // ro_obj = R^(-1) * (ro - t).
  float dx = ro[0] - tx, dy = ro[1] - ty, dz = ro[2] - tz;
  ro_out[0] = i00 * dx + i01 * dy + i02 * dz;
  ro_out[1] = i10 * dx + i11 * dy + i12 * dz;
  ro_out[2] = i20 * dx + i21 * dy + i22 * dz;

  // rd_obj = R^(-1) * rd.
  rd_out[0] = i00 * rd[0] + i01 * rd[1] + i02 * rd[2];
  rd_out[1] = i10 * rd[0] + i11 * rd[1] + i12 * rd[2];
  rd_out[2] = i20 * rd[0] + i21 * rd[1] + i22 * rd[2];
}

}}  // namespace vortex::rtu
