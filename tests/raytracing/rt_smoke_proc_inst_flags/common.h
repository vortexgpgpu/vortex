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
// PRISM RTU smoke: instance opacity flags on procedural geometry, and a box
// entered exactly at the committed distance.
//
// A one-leaf instance TLAS over procedural BLASes, three lanes of one warp:
//   - lane 0: a non-opaque box under an instance with FORCE_OPAQUE: its
//     candidate carries HIT_OPAQUE;
//   - lane 1: an opaque box under an instance with FORCE_NO_OPAQUE: its
//     candidate does not;
//   - lane 2: two instances of a non-opaque box along the ray, the second
//     entered at t = 4, the distance the intersection shader reports for the
//     first: it is still offered, and its accept at the same distance commits
//     its own instance.
// Every candidate is accepted at the intersection shader's t = 4.

#ifndef _RTU_SMOKE_PROC_INST_FLAGS_COMMON_H_
#define _RTU_SMOKE_PROC_INST_FLAGS_COMMON_H_

#include <stdint.h>

#define VX_BVH_SCENE_KIND          2
#define VX_BVH_SCENE_HDR_BYTES     16
#define VX_BVH_LEAF_HDR_BYTES      16
#define VX_BVH_PROC_AABB_BYTES     24
#define VX_BVH_INSTANCE_STRIDE     64
#define VX_BVH_INSTANCE_BLAS_OFF   48
#define VX_BVH_INSTANCE_CUSTOM_ID  52
#define VX_BVH_INSTANCE_ID_OFFSET  56
#define VX_BVH_INSTANCE_CULL_MASK  60
#define VX_BVH_KIND_LEAF_INST      2
#define VX_BVH_KIND_LEAF_PROC      3
#define VX_BVH_COUNT_SHIFT         8
#define VX_BVH_FLAG_OPAQUE         0x1u

// VkGeometryInstanceFlagBitsKHR, packed into the cull-mask word's bits 15..8.
#define RTU_INST_FLAGS_SHIFT       8
#define RTU_INST_FORCE_OPAQUE      0x4u
#define RTU_INST_FORCE_NO_OPAQUE   0x8u

#define RTU_NUM_LANES   3
#define RTU_IS_T        4.0f

typedef struct {
  uint32_t first_status;
  uint32_t cand_geom;  // raw HIT_GEOMETRY_INDEX word of the first candidate
  uint32_t num_cand;
  uint32_t status;
  float    hit_t;
  uint32_t geom;       // raw terminal HIT_GEOMETRY_INDEX word
  uint32_t inst;       // terminal HIT_INSTANCE_ID
} rtu_result_t;

typedef struct {
  uint64_t scene_addr;
  uint64_t results_addr;
  float    ray_x[RTU_NUM_LANES];
} kernel_arg_t;

#endif // _RTU_SMOKE_PROC_INST_FLAGS_COMMON_H_
