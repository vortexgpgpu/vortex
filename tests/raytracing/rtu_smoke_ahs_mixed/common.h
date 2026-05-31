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
// PRISM RTU mixed-scene AHS smoke — Phase 11.
//
// Scene = one NON-OPAQUE triangle at z=5 followed by one OPAQUE
// triangle at z=10 along the ray. With the Phase 11 walker the
// closest non-opaque hit (z=5) is yielded to AHS; the kernel's
// ACCEPT / IGNORE choice then determines which surface wins:
//   ACCEPT -> non-opaque wins, hit_t = 5.
//   IGNORE -> opaque (z=10) wins as fallback, hit_t = 10.
// Before Phase 11 the walker `break`ed on first non-opaque and the
// opaque tri at z=10 was never even checked: IGNORE produced MISS
// (status=1, hit_t=0) instead of the correct HIT-at-10.

#ifndef _RTU_SMOKE_AHS_MIXED_COMMON_H_
#define _RTU_SMOKE_AHS_MIXED_COMMON_H_

#include <stdint.h>

#define RTU_SCENE_HDR_BYTES    16
#define RTU_TRI_STRIDE_BYTES   40
#define RTU_TRI_FLAGS_OFFSET   36
#define RTU_TRI_FLAG_OPAQUE    0x1u

#define RTU_AHS_DECISION_ACCEPT  1
#define RTU_AHS_DECISION_IGNORE  0

typedef struct {
  uint32_t status;
  float    hit_t;
  uint32_t pad0;
  uint32_t pad1;
} rtu_result_t;

typedef struct {
  uint64_t scene_addr;
  uint64_t results_addr;
  uint32_t cb_decision;          // RTU_AHS_DECISION_*
  uint32_t reserved;
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_AHS_MIXED_COMMON_H_
