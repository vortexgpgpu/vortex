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

#ifndef _RTU_SMOKE_BVH6_COMMON_H_
#define _RTU_SMOKE_BVH6_COMMON_H_

#include <stdint.h>

// Mirror of sim/simx/rtu/rtu_bvh.h (CW-BVH6, scene_kind=3). Kept here so
// the test fixture stays self-contained.
#define VX_BVH_SCENE_KIND          3      // kRtuSceneKindBvh6
#define VX_BVH_SCENE_HDR_BYTES     16
#define VX_BVH_LEAF_HDR_BYTES      16
#define VX_BVH_TRI_STRIDE          40
#define VX_BVH_TRI_FLAGS_OFFSET    36

// CW-BVH6 internal node = 96 B (fan-out 6). Field byte offsets:
//   +0   uint32 kind (bits0..7 kind, 8..15 num_children)
//   +4   float  origin[3]      (12 B)
//   +16  int8   exp[3]         (3 B)
//   +19  uint8  pad0           (1 B)
//   +20  uint32 child_offsets[6] (24 B)
//   +44  uint8  qaabb_min[6][3]  (18 B)
//   +62  uint8  qaabb_max[6][3]  (18 B)
//   +80  uint8  pad1[16]
#define VX_BVH6_NODE_BYTES         96
#define VX_BVH6_WIDTH              6
#define VX_BVH6_OFF_ORIGIN         4
#define VX_BVH6_OFF_EXP            16
#define VX_BVH6_OFF_CHILD          20
#define VX_BVH6_OFF_QMIN           44
#define VX_BVH6_OFF_QMAX           62

#define VX_BVH_KIND_INTERNAL       0
#define VX_BVH_KIND_LEAF_TRI       1
#define VX_BVH_KIND_LEAF_INST      2
#define VX_BVH_KIND_LEAF_PROC      3
#define VX_BVH_COUNT_SHIFT         8

#define VX_BVH_CHILD_LEAF_FLAG     0x80000000u

#define VX_BVH_TRI_FLAG_OPAQUE     0x1u

typedef struct {
  uint32_t status;
  float    hit_t;
  float    hit_u;
  float    hit_v;
  uint32_t primitive_id;
  uint32_t geometry_index;
} rtu_result_t;

typedef struct {
  uint64_t scene_addr;
  uint64_t results_addr;
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_BVH6_COMMON_H_
