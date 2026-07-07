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
// PRISM RTU smoke test — Phase 1 + 2.
//
// Scene format (host-defined, also matches RtuCore's expected layout):
//   struct simple_tri_t {
//       float    v0[3], v1[3], v2[3];  // [0..35]
//       uint32_t flags;                 // [36..39]  bit 0 = OPAQUE
//   };
//   struct simple_scene_t {
//       uint32_t triangle_count;        // [0..3]   N triangles
//       uint32_t reserved[3];           // [4..15]  align to 16
//       simple_tri_t tris[N];           // [16..]   stride 40 B
//   };
// Fits in one 64 B cache line for N=1: 16 + 40 = 56 bytes.

#ifndef _RTU_SMOKE_COMMON_H_
#define _RTU_SMOKE_COMMON_H_

#include <stdint.h>

#define RTU_SCENE_MAX_TRIS    1
#define RTU_SCENE_HDR_BYTES   16
#define RTU_TRI_STRIDE_BYTES  40
#define RTU_TRI_FLAGS_OFFSET  36
#define RTU_TRI_FLAG_OPAQUE   0x1u

// Per-lane result: status + hit_t for validation against CPU oracle.
typedef struct {
  uint32_t status;
  float    hit_t;
  float    hit_u;
  float    hit_v;
  uint32_t primitive_id;
  uint32_t pad;
} rtu_result_t;

typedef struct {
  uint64_t scene_addr;     // device-side scene buffer pointer
  uint64_t results_addr;   // device-side per-lane results buffer
  uint32_t num_lanes;      // total active lanes
  uint32_t ray_pattern;    // 0 = all hit, 1 = alternate hit/miss
  // Per-lane ray descriptor (origin + direction). One ray per lane.
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_COMMON_H_
