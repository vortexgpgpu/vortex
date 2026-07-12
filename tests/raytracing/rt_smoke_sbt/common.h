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
// PRISM RTU Shader Binding Table smoke — Phase 7.
//
// Single primitive marked PROCEDURAL with per-tri sbt_idx = 1.
// The kernel populates a 2-record SBT in memory: record 0 has an IS
// shader writing MAGIC_0 and record 1 has one writing MAGIC_1. The
// trap dispatcher reads (sbt_idx, cb_type), looks up the SBT entry,
// and tail-jumps to the matched shader. The host verifies the
// payload contains MAGIC_1 (not MAGIC_0) — proving the runtime SBT
// lookup actually selected the right shader rather than dispatch
// being hardcoded.

#ifndef _RTU_SMOKE_SBT_COMMON_H_
#define _RTU_SMOKE_SBT_COMMON_H_

#include <stdint.h>

#define RTU_SCENE_HDR_BYTES    16
#define RTU_TRI_STRIDE_BYTES   40
#define RTU_TRI_FLAGS_OFFSET   36
#define RTU_TRI_FLAG_OPAQUE    0x1u
#define RTU_TRI_FLAG_PROC      0x2u
#define RTU_TRI_SBT_SHIFT      8
#define RTU_TRI_SBT_MASK       0xffu

// Per-SBT-record stride. Matches the layout the trap dispatcher
// indexes into: offset = (cb_type - 1) * 4 within a record.
#define RTU_SBT_RECORD_STRIDE  16

// Distinct magic per sbt_idx so a wrong lookup is detectable.
#define RTU_SBT_MAGIC_0        0x57b00000u
#define RTU_SBT_MAGIC_1        0x57b10001u

typedef struct {
  uint32_t status;
  float    hit_t;
  uint32_t sbt_payload;
  uint32_t pad;
} rtu_result_t;

typedef struct {
  uint64_t scene_addr;
  uint64_t results_addr;
  uint64_t payload_addr;
  uint64_t sbt_addr;         // base of the SBT records buffer
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_SBT_COMMON_H_
