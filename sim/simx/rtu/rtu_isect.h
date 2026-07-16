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
// RTU — ray-vs-primitive intersection + ray transform.
//
// Scalar inline-able functions called from the walker per triangle /
// AABB / instance, plus pipelined `BoxPe` and `TriPe` wrappers that
// model the SIMD intersection coprocessor with explicit per-PE
// latency (issue through `TriPe::issue`, drain via `TriPe::drain`).
//
// In hardware the scalar functions become combinational logic inside
// the box-PE / tri-PE / XFORM units.

#ifndef _VX_RTU_ISECT_H_
#define _VX_RTU_ISECT_H_

#include <cstdint>
#include "rtu_types.h"  // Vec3, dot, cross

namespace vortex { namespace rtu {

// ────────────────────────────────────────────────────────────────────
// Möller-Trumbore ray-triangle intersection.
//
// out_back_facing reports whether the ray hit the back side of the
// triangle's geometric normal (ray-flag face culling). Convention:
// triangle front face is the side from which (v0, v1, v2) appear CCW.
// Equivalently, det > 0 ↔ ray hits the front face.
// ────────────────────────────────────────────────────────────────────
bool ray_triangle(const float ro[3], const float rd[3],
                  const float v0[3], const float v1[3], const float v2[3],
                  float tmin, float tmax,
                  float& out_t, float& out_u, float& out_v,
                  bool& out_back_facing);

// ────────────────────────────────────────────────────────────────────
// Ray-vs-AABB slab test. Returns true if the ray's [tmin, tmax]
// interval overlaps the AABB; t_near is the entry parameter (clamped
// to tmin) used by the BVH4 walker to prune descent order.
//
// Assumes well-conditioned rays (no axis-aligned ray with zero
// direction component). A robust branchless ±inf variant is a later
// refinement.
// ────────────────────────────────────────────────────────────────────
bool ray_aabb_intersect(const float ro[3], const float rd[3],
                        const float mn[3], const float mx[3],
                        float tmin, float tmax, float& t_near);

// ────────────────────────────────────────────────────────────────────
// Apply the inverse of a 3x4 row-major affine to a ray, producing the
// object-space ray. Used by the BVH4 walker on LeafInst descent to
// convert world→object space. Mirrors the hardware XFORM unit
// (latency = 3 cycles).
//
//   xform = [r00 r01 r02 tx | r10 r11 r12 ty | r20 r21 r22 tz]
//   ro_obj = R^(-1) * (ro_world - t)
//   rd_obj = R^(-1) * rd_world
//
// For pure rotation+translation (det(R) == ±1) the t parameter is
// preserved across spaces, so the BLAS-reported hit_t is also the
// world hit_t. Non-uniform scale would require renormalising hit_t;
// out of scope.
// ────────────────────────────────────────────────────────────────────
void affine_inverse_transform_ray(const float xform[12],
                                  const float ro[3], const float rd[3],
                                  float ro_out[3], float rd_out[3]);

// ════════════════════════════════════════════════════════════════════
// The intersection coprocessors — pipelined BoxPe / TriPe.
// ════════════════════════════════════════════════════════════════════
//
//   BoxPe (ray-vs-AABB):  ONE PE, 1 box/cycle, 31-cycle pipeline depth.
//   TriPe (ray-vs-tri):   ONE PE, 1 tri/cycle, 91-cycle pipeline depth.
//
// Both are shared across the whole context array, so the issue slots are handed
// out by the orchestrator one per cycle and the contention is modelled, not
// assumed away. The math itself is done synchronously by the scalar
// ray_triangle / ray_aabb_intersect helpers above; these classes contribute only
// the drain behind the last test entered.
class BoxPe {
public:
  static uint32_t pipe_depth();
};

class TriPe {
public:
  static uint32_t pipe_depth();
};

}}  // namespace vortex::rtu

#endif  // _VX_RTU_ISECT_H_
