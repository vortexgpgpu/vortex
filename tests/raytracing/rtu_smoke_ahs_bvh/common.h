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
// PRISM RTU AHS-on-CW-BVH smoke — Phase 2.
//
// Same lone NON-opaque triangle + ray + AHS dispatcher as rtu_smoke_ahs,
// but the acceleration structure is a CW-BVH4 leaf (VX_CFG_RTU_BVH_WIDTH=4)
// instead of a flat triangle list. This isolates the CW-BVH walker's
// per-triangle any-hit path: a NON-opaque triangle must yield an AHS
// callback so the dispatcher's IGNORE turns the candidate into a MISS.

#ifndef _RTU_SMOKE_AHS_BVH_COMMON_H_
#define _RTU_SMOKE_AHS_BVH_COMMON_H_

#include <stdint.h>

// CW-BVH4 scene layout (matches rtu_smoke_bvh_basic).
#define VX_BVH_SCENE_KIND          2
#define VX_BVH_SCENE_HDR_BYTES     16
#define VX_BVH_LEAF_HDR_BYTES      16
#define VX_BVH_TRI_STRIDE          40
#define VX_BVH_TRI_FLAGS_OFFSET    36
#define VX_BVH_KIND_LEAF_TRI       1
#define VX_BVH_COUNT_SHIFT         8
#define VX_BVH_TRI_FLAG_OPAQUE     0x1u

#define RTU_AHS_DECISION_ACCEPT  1
#define RTU_AHS_DECISION_IGNORE  0

typedef struct {
  uint32_t status;
  float    hit_t;
  float    hit_u;
  float    hit_v;
  uint32_t primitive_id;
  uint32_t pad;
} rtu_result_t;

typedef struct {
  uint64_t scene_addr;
  uint64_t results_addr;
  uint32_t num_lanes;
  uint32_t cb_decision;   // RTU_AHS_DECISION_*
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_AHS_BVH_COMMON_H_
