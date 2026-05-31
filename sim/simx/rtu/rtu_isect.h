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
// PRISM RTU — ray-vs-primitive intersection + ray transform.
// Layer 2 of the rtu_implementation.md refactor (Option C, 13 files).
//
// Today: scalar inline-able functions called from the walker per
// triangle / AABB / instance. §8.7 future: pipelined `BoxPe` and
// `TriPe` classes land in rtu_isect.cpp alongside the scalar versions
// to model the SIMD intersection coprocessor with explicit per-PE
// latency. The walker doesn't change shape — it switches from calling
// `ray_triangle(...)` directly to issuing through a `TriPe::issue` and
// later draining via `TriPe::drain`.
//
// In RTL: scalar functions → combinational logic inside the box-PE /
// tri-PE / XFORM units. The pipelined wrappers map 1:1 to SystemC
// `SC_MODULE(BoxPe)` / `SC_MODULE(TriPe)`.

#ifndef _VX_RTU_ISECT_H_
#define _VX_RTU_ISECT_H_

#include <cstdint>
#include "rtu_types.h"  // Vec3, dot, cross

namespace vortex { namespace rtu {

// ────────────────────────────────────────────────────────────────────
// Möller-Trumbore ray-triangle intersection.
//
// out_back_facing reports whether the ray hit the back side of the
// triangle's geometric normal (§8.8 ray-flag face culling). Convention:
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
// convert world→object space. Mirrors the proposal §5.7 XFORM unit
// (latency = 3 cycles in RTL).
//
//   xform = [r00 r01 r02 tx | r10 r11 r12 ty | r20 r21 r22 tz]
//   ro_obj = R^(-1) * (ro_world - t)
//   rd_obj = R^(-1) * rd_world
//
// For pure rotation+translation (det(R) == ±1) the t parameter is
// preserved across spaces, so the BLAS-reported hit_t is also the
// world hit_t. Non-uniform scale would require renormalising hit_t;
// out of scope for Phase 9 minimum.
// ────────────────────────────────────────────────────────────────────
void affine_inverse_transform_ray(const float xform[12],
                                  const float ro[3], const float rd[3],
                                  float ro_out[3], float rd_out[3]);

}}  // namespace vortex::rtu

#endif  // _VX_RTU_ISECT_H_
