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
#include <cfloat>
#include <cmath>
#include <utility>

namespace vortex { namespace rtu {

bool ray_triangle(const float ro[3], const float rd[3],
                  const float v0[3], const float v1[3], const float v2[3],
                  float tmin, float tmax,
                  float& out_t, float& out_u, float& out_v,
                  bool& out_back_facing) {
  const float* vin[3] = { v0, v1, v2 };

  // Watertight ray/triangle test (Woop, Benthin, Wald, JCGT 2013), F32 only:
  // shear the triangle into the ray's frame so the ray runs along +z, then
  // test the 2D edge functions. Each edge function is two rounded products and
  // a rounded difference of the sheared vertices alone, so an edge shared by
  // two triangles evaluates to exactly negated weights in both and no ray
  // slips between them. Mirrors VX_rtu_tri_pe op for op.
  const float ad[3] = { std::fabs(rd[0]), std::fabs(rd[1]), std::fabs(rd[2]) };
  int kz = (ad[0] >= ad[1]) ? ((ad[0] >= ad[2]) ? 0 : 2)
                            : ((ad[1] >= ad[2]) ? 1 : 2);
  int kx = (kz + 1) % 3;
  int ky = (kx + 1) % 3;
  if (rd[kz] < 0.f) std::swap(kx, ky);  // keep the winding

  const float sz = 1.0f / rd[kz];
  const float sx = rd[kx] * sz;
  const float sy = rd[ky] * sz;

  float px[3], py[3], pz[3];
  for (int i = 0; i < 3; ++i) {
    const float* q = vin[i];
    const float rx = q[kx] - ro[kx];
    const float ry = q[ky] - ro[ky];
    const float rz = q[kz] - ro[kz];
    px[i] = std::fma(-sx, rz, rx);
    py[i] = std::fma(-sy, rz, ry);
    pz[i] = sz * rz;
  }

  // Edge functions: w[i] is the weight of vertex i.
  const float w0 = px[2] * py[1] - py[2] * px[1];
  const float w1 = px[0] * py[2] - py[0] * px[2];
  const float w2 = px[1] * py[0] - py[1] * px[0];
  if ((w0 < 0.f || w1 < 0.f || w2 < 0.f) && (w0 > 0.f || w1 > 0.f || w2 > 0.f))
    return false;

  const float det = (w0 + w1) + w2;
  // Reject only an edge-on or zero-area triangle: |det| scales with the
  // triangle's area, so any epsilon would drop small triangles.
  if (!(det != 0.f)) return false;

  const float T   = std::fma(w2, pz[2], std::fma(w1, pz[1], w0 * pz[0]));
  const float rcp = 1.0f / det;
  const float t   = T * rcp;
  // Vulkan's ray interval for a triangle is open at both ends: an intersection
  // candidate needs t_min < t < t_max (Ray Intersection Candidate
  // Determination).
  if (!(tmin < t && t < tmax)) return false;

  out_t = t;
  out_u = w1 * rcp;
  out_v = w2 * rcp;
  // det > 0: (v0, v1, v2) winds counter-clockwise as seen by the ray.
  out_back_facing = (det < 0.f);
  return true;
}

float ray_recip(float d) {
  // A zero (or subnormal, which the PEs flush) component has no reciprocal;
  // FLT_MAX keeps every slab product finite, so no slab is 0 * inf = NaN.
  if (std::fabs(d) < FLT_MIN) return FLT_MAX;
  return 1.0f / d;
}

float quant_corner(uint8_t q, int8_t e) {
  // Exact product; the PE flushes a subnormal result, inf past the range.
  const float c = std::ldexp(float(q), e);
  return (c < FLT_MIN) ? 0.f : c;
}

void box_rel(const float base[3], const float mn[3], const float mx[3],
             const float ro[3], float rel_mn[3], float rel_mx[3]) {
  for (int i = 0; i < 3; ++i) {
    const float c = base[i] - ro[i];
    rel_mn[i] = mn[i] + c;
    rel_mx[i] = mx[i] + c;
  }
}

bool ray_box(const float rel_mn[3], const float rel_mx[3], const float inv[3],
             float tmin, float tmax, float& t_near) {
  // fmin/fmax drop a NaN operand, so a NaN slab (a NaN box or ray) drops out
  // of the fold rather than poisoning it.
  float lo = std::fmax(-INFINITY, tmin);
  float hi = std::fmin(INFINITY, tmax);
  for (int i = 0; i < 3; ++i) {
    const float t0 = rel_mn[i] * inv[i];
    const float t1 = rel_mx[i] * inv[i];
    lo = std::fmax(lo, std::fmin(t0, t1));
    hi = std::fmin(hi, std::fmax(t0, t1));
  }
  if (!(lo <= hi)) return false;
  t_near = (lo == 0.f) ? 0.f : lo;
  return true;
}

void world_to_object_ray(const float wto[12],
                         const float ro[3], const float rd[3],
                         float ro_out[3], float rd_out[3]) {
  for (int i = 0; i < 3; ++i) {
    const float* m = wto + 4 * i;
    ro_out[i] = std::fma(ro[2], m[2], std::fma(ro[1], m[1], std::fma(ro[0], m[0], m[3])));
    rd_out[i] = std::fma(rd[2], m[2], std::fma(rd[1], m[1], rd[0] * m[0]));
  }
}

// PE cost model. There is ONE box PE and ONE tri PE per RtuCore, each streaming
// one primitive per cycle across every context — not a fan-out-wide array. The
// orchestrator hands the issue slots out itself (so a wide context array costs
// issue bandwidth instead of getting it free) and only needs the drain behind
// the last test, which is the hardware pipeline depth expressed symbolically
// from the FMA / FDIV latencies so it tracks the config.
uint32_t BoxPe::pipe_depth() {
  // 3 FMA stages (slab min/max) + 1 + 2 + 1 = 31.
  return 3 * kRtuLatencyFma + 1 + 2 + 1;
}

uint32_t TriPe::pipe_depth() {
  // input select + 1/dir[kz] + shear scale + shear + 2 edge stages + det +
  // 1/det + t/u/v scale + verdict (VX_rtu_tri_pe).
  return 2 + 2 * kRtuFdivLat + 7 * kRtuLatencyFma;
}

}}  // namespace vortex::rtu
