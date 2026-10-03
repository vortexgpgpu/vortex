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

  // Watertight ray/triangle test (Woop, Benthin, Wald, JCGT 2013): shear the
  // triangle into the ray's frame so the ray runs along +z, then test the 2D
  // edge functions. A shared edge evaluates to exactly negated values in its
  // two triangles, so no ray slips between them; F64 edge functions and t keep
  // t within half an ulp of the exact intersection. The op order is the one
  // the Vulkan reference (lavapipe) evaluates, so t matches it bit for bit,
  // coincident triangles included.
  const float ad[3] = { std::fabs(rd[0]), std::fabs(rd[1]), std::fabs(rd[2]) };
  int kz = (ad[0] >= ad[1]) ? ((ad[0] >= ad[2]) ? 0 : 2)
                            : ((ad[1] >= ad[2]) ? 1 : 2);
  int kx = (kz + 1) % 3;
  int ky = (kx + 1) % 3;
  if (rd[kz] < 0.f) std::swap(kx, ky);  // keep the winding

  const float sz = 1.0f / rd[kz];
  const float sx = rd[kx] * sz;
  const float sy = rd[ky] * sz;

  // F32 shear, as each op rounds in the pipeline; F64 from here on, where the
  // products of two F32 values are exact.
  float px[3], py[3];
  double pz[3];
  for (int i = 0; i < 3; ++i) {
    const float* q = vin[i];
    const float rx = q[kx] - ro[kx];
    const float ry = q[ky] - ro[ky];
    const float rz = q[kz] - ro[kz];
    const float mx = sx * rz;
    const float my = sy * rz;
    px[i] = rx - mx;
    py[i] = ry - my;
    pz[i] = double(sz) * double(rz);
  }

  // Edge functions: w[i] is the weight of vertex i.
  double w[3];
  w[0] = double(px[2]) * py[1] - double(py[2]) * px[1];
  w[1] = double(px[0]) * py[2] - double(py[0]) * px[2];
  w[2] = double(px[1]) * py[0] - double(py[1]) * px[0];
  if ((w[0] < 0.0 || w[1] < 0.0 || w[2] < 0.0)
   && (w[0] > 0.0 || w[1] > 0.0 || w[2] > 0.0))
    return false;

  const double det = w[0] + (w[1] + w[2]);
  // Reject only an edge-on or zero-area triangle: |det| scales with the
  // triangle's area, so any epsilon would drop small triangles.
  if (!(det != 0.0)) return false;

  const double tp0 = w[0] * pz[0];
  const double tp1 = w[1] * pz[1];
  const double tp2 = w[2] * pz[2];
  const float t = float(((tp0 + tp1) + tp2) / det);
  // Open interval, as the reference commits (lvp_build_triangle_case:
  // tmin < t and t < tmax).
  if (!(tmin < t && t < tmax)) return false;

  const float det32 = float(det);
  out_t = t;
  out_u = float(w[1]) / det32;
  out_v = float(w[2]) / det32;
  // det > 0: (v0, v1, v2) winds counter-clockwise as seen by the ray.
  out_back_facing = (det < 0.0);
  return true;
}

bool ray_aabb_intersect(const float ro[3], const float rd[3],
                        const float mn[3], const float mx[3],
                        float tmin, float tmax, float& t_near) {
  // Slab test in the Vulkan reference's (lavapipe) form: a zero direction
  // component uses FLT_MAX as its reciprocal, and the box is culled against
  // [0, tmax], NOT [tmin, tmax]. The tmin floor belongs to the primitive test
  // alone: a primitive's t carries rounding the slab distances do not (the
  // watertight triangle t of a large triangle is off by far more than the
  // slabs of its flat box), so a box whose exact exit lies below tmin can
  // still hold a hit the primitive test reports past tmin. Culling at tmin
  // would drop that hit, which the reference keeps.
  float lo = -INFINITY, hi = INFINITY;
  for (int i = 0; i < 3; ++i) {
    const float inv = (rd[i] == 0.0f) ? FLT_MAX : 1.0f / rd[i];
    const float t0 = (mn[i] - ro[i]) * inv;
    const float t1 = (mx[i] - ro[i]) * inv;
    lo = std::fmax(lo, std::fmin(t0, t1));
    hi = std::fmin(hi, std::fmax(t0, t1));
  }
  // The upper bound stays inclusive (the reference's is strict): a box
  // entered exactly at the committed t can hold an equal-t twin that the
  // lowest-(instance, geometry, prim) tie-break must still see.
  if (!(hi >= std::fmax(0.0f, lo) && lo <= tmax)) return false;
  t_near = std::fmax(tmin, lo);   // descent order only
  return true;
}

void world_to_object_ray(const float wto[12],
                         const float ro[3], const float rd[3],
                         float ro_out[3], float rd_out[3]) {
  for (int i = 0; i < 3; ++i) {
    const float* m = wto + 4 * i;
    float o = m[3];
    o = o + ro[0] * m[0];
    o = o + ro[1] * m[1];
    o = o + ro[2] * m[2];
    float d = rd[0] * m[0];
    d = d + rd[1] * m[1];
    d = d + rd[2] * m[2];
    ro_out[i] = o;
    rd_out[i] = d;
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
  // input select + 1/dir + 3 F32 stages + 5 F64 stages + F64 divide + narrow
  // + verdict (VX_rtu_tri_pe).
  return 3 + kRtuFdivLat + 3 * kRtuLatencyFma + 5 * kRtuLatencyFma64
       + kRtuFdiv64Lat;
}

}}  // namespace vortex::rtu
