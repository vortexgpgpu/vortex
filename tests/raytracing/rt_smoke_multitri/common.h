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
// PRISM RTU multi-triangle smoke — Phase 4.
//
// Scene with N opaque triangles, all at distinct depths along +Z. The
// ray fires from (0.25, 0.25, 0) toward +Z, intersecting every
// triangle. The RtuCore must walk all N tris (linear-scan replacement
// for Phase 1's kPhase1MaxTris=1 cap) and return the CLOSEST hit. The
// scene spans multiple cache lines for N >= 2, so this also exercises
// the two-phase fetch (header line → body lines) plumbing.

#ifndef _RTU_SMOKE_MULTITRI_COMMON_H_
#define _RTU_SMOKE_MULTITRI_COMMON_H_

#include <stdint.h>

#define RTU_SCENE_HDR_BYTES    16
#define RTU_TRI_STRIDE_BYTES   40
#define RTU_TRI_FLAGS_OFFSET   36
#define RTU_TRI_FLAG_OPAQUE    0x1u
#define RTU_MAX_TRIS           8

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
  uint32_t num_tris;
  uint32_t reserved;
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_MULTITRI_COMMON_H_
