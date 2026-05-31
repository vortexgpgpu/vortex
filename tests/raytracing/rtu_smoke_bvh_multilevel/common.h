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

#ifndef _RTU_SMOKE_BVH_MULTILEVEL_COMMON_H_
#define _RTU_SMOKE_BVH_MULTILEVEL_COMMON_H_

#include <stdint.h>

// Mirrors sim/simx/rtu/rtu_bvh.h.
#define VX_BVH_SCENE_KIND           2
#define VX_BVH_SCENE_HDR_BYTES      16
#define VX_BVH_INTERNAL_NODE_BYTES  64
#define VX_BVH_LEAF_HDR_BYTES       16
#define VX_BVH_TRI_STRIDE           40
#define VX_BVH_TRI_FLAGS_OFFSET     36
#define VX_BVH_WIDTH                4

#define VX_BVH_KIND_INTERNAL        0
#define VX_BVH_KIND_LEAF_TRI        1
#define VX_BVH_COUNT_SHIFT          8

#define VX_BVH_TRI_FLAG_OPAQUE      0x1u

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
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_BVH_MULTILEVEL_COMMON_H_
