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
// PRISM RTU reformation divergent-SBT smoke — Phase 3-A2 option B.
//
// 1 block × VX_CFG_NUM_THREADS lanes (single warp). Each lane uses its
// OWN scene (cache-line-aligned per-lane scene_root) whose lone
// non-opaque tri carries sbt_idx = (tid / SBT_GROUP_SIZE). With the
// default SBT_GROUP_SIZE=2 the warp splits 2+2 across sbt 0 and sbt 1.
// The dispatcher branches on VX_RT_HIT_SBT_IDX: sbt 0 → ACCEPT, sbt!=0
// → IGNORE. Expected post-condition: lanes whose sbt is 0 HIT, others
// MISS. With debug=3 the run log shows TWO "reform cb_yield" lines,
// each with a per-sbt cb_mask (e.g. 0x3 and 0xc for 4 lanes / group=2).

#ifndef _RTU_SMOKE_REFORM_SBT_COMMON_H_
#define _RTU_SMOKE_REFORM_SBT_COMMON_H_

#include <stdint.h>

#define RTU_SCENE_MAX_TRIS    1
#define RTU_SCENE_HDR_BYTES   16
#define RTU_SCENE_BYTES       64    // per-lane scene fits in one cache line
#define RTU_TRI_STRIDE_BYTES  40
#define RTU_TRI_FLAGS_OFFSET  36
#define RTU_TRI_FLAG_OPAQUE   0x1u
#define RTU_TRI_SBT_SHIFT     8
#define RTU_TRI_SBT_MASK      0xffu

typedef struct {
  uint32_t status;
  float    hit_t;
  float    hit_u;
  float    hit_v;
  uint32_t primitive_id;
  uint32_t pad;
} rtu_result_t;

typedef struct {
  uint64_t scene_base_addr;   // base of per-lane scene array
  uint64_t results_addr;
  uint32_t num_lanes;
  uint32_t sbt_group_size;    // lanes per sbt group; tid/group → sbt
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_REFORM_SBT_COMMON_H_
