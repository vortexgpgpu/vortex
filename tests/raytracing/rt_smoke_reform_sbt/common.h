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
// PRISM RTU reformation divergent-SBT smoke.
//
// 1 block x VX_CFG_NUM_THREADS lanes (single warp), ONE shared scene holding
// num_lanes non-opaque tris laid out along +x: tri i spans x in
// [i*SPACING, i*SPACING+1] at z=Z and carries sbt_idx = i / sbt_group_size.
// Lane i shoots a +z ray aimed at tri i, so the per-lane SBT divergence comes
// from the rays/hits, not the scene pointer — one warp-uniform vx_rt_wtrace
// covers it. The dispatcher branches on VX_RT_HIT_SBT_IDX: sbt 0 -> ACCEPT,
// sbt != 0 -> IGNORE. Post-condition: sbt-0 lanes HIT, the rest MISS, and
// reformation emits one CB_YIELD per sbt (grouped by sbt_idx).

#ifndef _RTU_SMOKE_REFORM_SBT_COMMON_H_
#define _RTU_SMOKE_REFORM_SBT_COMMON_H_

#include <stdint.h>

#define RTU_SCENE_HDR_BYTES   16
#define RTU_TRI_STRIDE_BYTES  40
#define RTU_TRI_FLAGS_OFFSET  36
#define RTU_TRI_FLAG_OPAQUE   0x1u
#define RTU_TRI_SBT_SHIFT     8
#define RTU_TRI_SBT_MASK      0xffu

#define RTU_TRI_SPACING       1.0f   // x-spacing between adjacent per-lane tris
#define RTU_TRI_Z             5.0f   // tri plane (== expected hit_t for a z=0 ray)
#define RTU_RAY_XOFF          0.25f  // ray aim inside its tri (local x)
#define RTU_RAY_Y             0.25f  // ray aim inside its tri (local y)

typedef struct {
  uint32_t status;
  float    hit_t;
  float    hit_u;
  float    hit_v;
  uint32_t primitive_id;
  uint32_t pad;
} rtu_result_t;

typedef struct {
  uint64_t scene_base_addr;   // base of the single shared scene
  uint64_t results_addr;
  uint32_t num_lanes;
  uint32_t sbt_group_size;    // lanes per sbt group; tid/group -> sbt
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_REFORM_SBT_COMMON_H_
