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
// PRISM RTU reformation smoke — Phase 3-A2 (option A: same-sbt batching).
//
// All N lanes belong to ONE warp and trace the SAME ray against the SAME
// non-opaque triangle (sbt_idx = 0). The RtuCore yields an AHS callback
// for every active lane; reformation_dispatch groups them by (warp_id,
// sbt_idx) and emits a SINGLE CB_YIELD whose cb_mask covers all N lanes.
// That trap runs one ACCEPT dispatcher for the whole virtual warp — the
// SIMT-coherence win Phase 3-A2 exists to demonstrate. Per-lane HIT
// status is then validated host-side.

#ifndef _RTU_SMOKE_REFORM_COMMON_H_
#define _RTU_SMOKE_REFORM_COMMON_H_

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
  uint32_t num_lanes;
  uint32_t reserved;
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_REFORM_COMMON_H_
