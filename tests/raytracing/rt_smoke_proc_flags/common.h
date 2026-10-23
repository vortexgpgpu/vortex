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
// PRISM RTU procedural-opacity and provenance smoke.
//
// Three one-leaf CW-BVH4 scenes on the same ray: an opaque procedural box, a
// non-opaque procedural box, and an opaque triangle. Each case traces one of
// them under a set of ray flags and answers the candidate with ACCEPT (at the
// intersection shader's t = 4) or IGNORE, recording the candidate's geometry
// word and the terminal record:
//   - a box's opacity is resolved like a triangle's (leaf flag, then the ray's
//     OPAQUE / NO_OPAQUE) and travels with the candidate (HIT_OPAQUE);
//   - CULL_OPAQUE / CULL_NO_OPAQUE and SKIP_AABBS drop a box before it yields;
//   - the terminal geometry word says whether a shader verdict committed the
//     hit (COMMITTED_CB) and whether it is procedural (COMMITTED_PROC); an
//     opaque triangle the RTU committed itself carries neither.

#ifndef _RTU_SMOKE_PROC_FLAGS_COMMON_H_
#define _RTU_SMOKE_PROC_FLAGS_COMMON_H_

#include <stdint.h>

#define VX_BVH_SCENE_KIND          2
#define VX_BVH_SCENE_HDR_BYTES     16
#define VX_BVH_LEAF_HDR_BYTES      16
#define VX_BVH_PROC_AABB_BYTES     24
#define VX_BVH_TRI_STRIDE          40
#define VX_BVH_TRI_FLAGS_OFFSET    36
#define VX_BVH_KIND_LEAF_TRI       1
#define VX_BVH_KIND_LEAF_PROC      3
#define VX_BVH_COUNT_SHIFT         8
#define VX_BVH_FLAG_OPAQUE         0x1u

#define RTU_NUM_SCENES   3
#define RTU_IS_T         4.0f

typedef struct {
  uint32_t scene;      // 0 opaque box, 1 non-opaque box, 2 opaque triangle
  uint32_t ray_flags;
  uint32_t action;     // verdict on every candidate
} rtu_case_t;

typedef struct {
  uint32_t first_status;
  uint32_t cand_geom;  // raw HIT_GEOMETRY_INDEX word of the first candidate
  uint32_t num_cand;
  uint32_t status;
  float    hit_t;
  uint32_t geom;       // raw terminal HIT_GEOMETRY_INDEX word
} rtu_result_t;

typedef struct {
  uint64_t scene_addr[RTU_NUM_SCENES];
  uint64_t cases_addr;
  uint64_t results_addr;
  uint32_t num_cases;
  uint32_t pad;
} kernel_arg_t;

#endif // _RTU_SMOKE_PROC_FLAGS_COMMON_H_
