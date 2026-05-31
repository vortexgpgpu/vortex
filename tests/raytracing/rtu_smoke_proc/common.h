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
// PRISM RTU procedural intersection-shader smoke.
//
// Exercises the three procedural-path P1 features end-to-end:
//   (1) object-space ray readback (VX_RT_OBJECT_RAY_*, slots 8..13) — the
//       IS reads it and echoes it into hitAttribute so the host can check
//       the RTU populated the right values.
//   (2) user hitAttributeEXT (VX_RT_HIT_ATTR_*, slots 17..20) + the
//       IS-supplied VX_RT_HIT_T committed on ACCEPT. The IS reports a t
//       distinct from the AABB-entry candidate, so a passing hit_t proves
//       the cb_hit_t commit path (not the candidate fallback).
//   (3) the CW-BVH4 LeafProc walker path (the IS only fires if LeafProc
//       yields).
//
// The IS body is intentionally integer-only (slot copies + an immediate
// t): the RTU callback trap dispatcher cannot yet execute floating-point
// (a separate, pre-existing limitation tracked outside this test), so the
// shader avoids FP while still validating all three features.

#ifndef _RTU_SMOKE_PROC_COMMON_H_
#define _RTU_SMOKE_PROC_COMMON_H_

#include <stdint.h>

// Mirror of sim/simx/rtu/rtu_bvh.h (kept local so the fixture is
// self-contained — host/kernel don't include SimX internals).
#define VX_BVH_SCENE_KIND          2     // kRtuSceneKindBvh4
#define VX_BVH_SCENE_HDR_BYTES     16
#define VX_BVH_LEAF_HDR_BYTES      16
#define VX_BVH_PROC_AABB_BYTES     24
#define VX_BVH_KIND_LEAF_PROC      3
#define VX_BVH_COUNT_SHIFT         8

// Unit sphere the procedural AABB bounds (object == world space here).
#define RTU_SPHERE_CX   0.0f
#define RTU_SPHERE_CY   0.0f
#define RTU_SPHERE_CZ   5.0f
#define RTU_SPHERE_R    1.0f

// The distance the IS reports (0x40900000 = 4.5f) — deliberately != the
// AABB-entry candidate t (4.0), so committing it proves the IS hit_t path.
#define RTU_IS_HIT_T_BITS   0x40900000u
#define RTU_IS_HIT_T        4.5f

// hitAttributeEXT sentinel the IS writes into VX_RT_HIT_ATTR_0.
#define RTU_IS_ATTR_MAGIC   0x5be12a11u

typedef struct {
  uint32_t status;       // VX_RT_STS_DONE_HIT / _MISS
  float    hit_t;        // committed (IS-supplied) hit distance
  uint32_t hit_attr0;    // committed hitAttributeEXT[0] (magic sentinel)
  uint32_t hit_attr1;    // echoed object-ray origin.z bits (feature 1)
  uint32_t hit_attr2;    // echoed object-ray direction.z bits (feature 1)
  uint32_t pad;
} rtu_result_t;

typedef struct {
  uint64_t scene_addr;
  uint64_t results_addr;
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_PROC_COMMON_H_
