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
// RTU partial-warp trace smoke.
//
// A trace's warp-uniform config (scene, payload, flags|cull) is gathered into
// lanes 1-3 of its config register. A trace issued by a warp whose low lanes
// are masked off must still deliver its own config: the kernel first traces an
// EMPTY scene from every lane (leaving that scene in the register's lanes),
// then traces the triangle scene from the warp's last lane only.

#ifndef _RTU_SMOKE_PARTIAL_MASK_COMMON_H_
#define _RTU_SMOKE_PARTIAL_MASK_COMMON_H_

#include <stdint.h>

#define VX_BVH_SCENE_KIND          2
#define VX_BVH_SCENE_HDR_BYTES     16
#define VX_BVH_LEAF_HDR_BYTES      16
#define VX_BVH_TRI_STRIDE          40
#define VX_BVH_TRI_FLAGS_OFFSET    36
#define VX_BVH_KIND_LEAF_TRI       1
#define VX_BVH_COUNT_SHIFT         8
#define VX_BVH_TRI_FLAG_OPAQUE     0x1u

typedef struct {
  uint32_t first_status;    // the all-lane trace of the empty scene
  uint32_t second_status;   // the last-lane trace of the triangle scene
  float    second_t;
  uint32_t pad;
} rtu_result_t;

typedef struct {
  uint64_t empty_scene_addr;
  uint64_t tri_scene_addr;
  uint64_t results_addr;
  uint32_t num_lanes;
  uint32_t pad;
} kernel_arg_t;

#endif // _RTU_SMOKE_PARTIAL_MASK_COMMON_H_
