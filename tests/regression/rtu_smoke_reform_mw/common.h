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
// PRISM RTU reformation multi-warp smoke — Phase 3-A2 non-interference.
//
// num_warps blocks × VX_CFG_NUM_THREADS lanes each. Every warp fires
// the same ray at the same non-opaque triangle (sbt_idx=0). The Phase
// 3-A2 reformation must NEVER bundle lanes from different warps into
// one CB_YIELD — same-sbt or not, a cb_active_mask is per-warp scope.
// Validation:
//   * every lane HITs at t=5
//   * with debug=3 the run log shows exactly one "reform cb_yield"
//     line per dispatched warp, each with cb_mask = 0xf (all lanes of
//     that warp), and warp= takes a different value each time

#ifndef _RTU_SMOKE_REFORM_MW_COMMON_H_
#define _RTU_SMOKE_REFORM_MW_COMMON_H_

#include <stdint.h>

#define RTU_SCENE_MAX_TRIS    1
#define RTU_SCENE_HDR_BYTES   16
#define RTU_TRI_STRIDE_BYTES  40
#define RTU_TRI_FLAGS_OFFSET  36
#define RTU_TRI_FLAG_OPAQUE   0x1u

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
  uint32_t total_lanes;     // num_warps * lanes_per_warp
  uint32_t lanes_per_warp;
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_REFORM_MW_COMMON_H_
