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
// PRISM RTU any-hit candidate instance-attribute smoke — I1 parity guard.
//
// A single-instance CW-BVH4 TLAS holding a non-opaque triangle. The
// non-opaque triangle yields an AHS *candidate* callback; the dispatcher
// reads the CANDIDATE instance attributes (VX_RT_HIT_INSTANCE_CUSTOM /
// VX_RT_HIT_INSTANCE_ID) straight out of the register window and stashes
// them so the host can assert the candidate stage — NOT the post-commit
// terminal path — reports the true instance custom index / id.

#ifndef _RTU_SMOKE_AHS_CUSTOM_COMMON_H_
#define _RTU_SMOKE_AHS_CUSTOM_COMMON_H_

#include <stdint.h>

// Terminal (post-vx_rt_wait) result.
typedef struct {
  uint32_t status;
  float    hit_t;
  uint32_t instance_id;
  uint32_t instance_custom;
} rtu_result_t;

// Candidate capture written by the AHS dispatcher during the callback.
typedef struct {
  uint32_t cand_instance_custom;   // gl_InstanceCustomIndexEXT (candidate slot)
  uint32_t cand_instance_id;       // gl_InstanceID (candidate slot)
} rtu_cand_t;

typedef struct {
  uint64_t scene_addr;
  uint64_t results_addr;
  uint64_t cand_addr;      // candidate-capture buffer (carried as the trace payload)
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_AHS_CUSTOM_COMMON_H_
