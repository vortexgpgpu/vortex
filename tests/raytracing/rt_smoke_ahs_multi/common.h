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
// RTU multi-candidate any-hit smoke.
//
// Several non-opaque triangles lie along one ray -- two of them coplanar at the
// same t -- with an opaque triangle behind them. Every candidate the walk meets
// must be offered, one per callback round, in ascending (t, record) order; a
// verdict that does not end the ray resumes the walk above the decided
// candidate. Each lane accepts a different primitive and records the order it
// was offered candidates in.

#ifndef _RTU_SMOKE_AHS_MULTI_COMMON_H_
#define _RTU_SMOKE_AHS_MULTI_COMMON_H_

#include <stdint.h>

// CW-BVH4 scene layout (matches rt_smoke_bvh_basic).
#define VX_BVH_SCENE_KIND          2
#define VX_BVH_SCENE_HDR_BYTES     16
#define VX_BVH_LEAF_HDR_BYTES      16
#define VX_BVH_TRI_STRIDE          40
#define VX_BVH_TRI_FLAGS_OFFSET    36
#define VX_BVH_KIND_LEAF_TRI       1
#define VX_BVH_COUNT_SHIFT         8
#define VX_BVH_TRI_FLAG_OPAQUE     0x1u

#define RTU_MULTI_NUM_TRIS  6
#define RTU_MULTI_MAX_OFFER 8
#define RTU_MULTI_NONE      0xffu   // accept nothing

typedef struct {
  uint32_t status;
  float    hit_t;
  uint32_t primitive_id;
  uint32_t num_offered;
  uint8_t  offered[RTU_MULTI_MAX_OFFER];   // primitive ids, in callback order
} rtu_result_t;

typedef struct {
  uint64_t scene_addr;
  uint64_t results_addr;
  uint64_t targets_addr;   // uint8_t per lane: the primitive it accepts
  uint32_t num_lanes;
  uint32_t pad;
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_AHS_MULTI_COMMON_H_
