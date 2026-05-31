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
// PRISM RTU AHS-callback smoke — Phase 2.
//
// Scene format identical to rtu_smoke; tri.flags bit 0 = OPAQUE. For
// this test the lone triangle is marked NON-opaque so the RtuCore yields
// an AHS callback. The kernel's dispatcher (registered into `mtvec`)
// either ACCEPTs or IGNOREs based on `kernel_arg_t.cb_decision`.

#ifndef _RTU_SMOKE_AHS_COMMON_H_
#define _RTU_SMOKE_AHS_COMMON_H_

#include <stdint.h>

#define RTU_SCENE_MAX_TRIS    1
#define RTU_SCENE_HDR_BYTES   16
#define RTU_TRI_STRIDE_BYTES  40
#define RTU_TRI_FLAGS_OFFSET  36
#define RTU_TRI_FLAG_OPAQUE   0x1u

#define RTU_AHS_DECISION_ACCEPT  1
#define RTU_AHS_DECISION_IGNORE  0

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
  uint32_t cb_decision;   // RTU_AHS_DECISION_*
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_AHS_COMMON_H_
