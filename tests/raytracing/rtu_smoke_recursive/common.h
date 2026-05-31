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
// PRISM RTU recursive-trace smoke — Phase 12.
//
// Kernel fires a primary ray at scene A (opaque tri @ z=5). The CHS
// dispatcher reads the sub-scene address from VX_RT_HIT_ATTR_0
// (which the kernel pre-loaded), fires a sub-trace into scene B
// (opaque tri @ z=10), waits for its TERMINAL, and writes the
// sub-ray status to the payload. The shader then cb_ret(DONE) +
// mret; the parent slot transitions to RESP and the kernel sees
// status=HIT, parent hit_t=5 (NOT the sub-ray's hit_t=10, because
// the parent's apply_response refreshes the regfile after the
// shader returns), and payload=HIT (= sub-ray status from inside
// the recursive trace).

#ifndef _RTU_SMOKE_RECURSIVE_COMMON_H_
#define _RTU_SMOKE_RECURSIVE_COMMON_H_

#include <stdint.h>

#define RTU_SCENE_HDR_BYTES    16
#define RTU_TRI_STRIDE_BYTES   40
#define RTU_TRI_FLAGS_OFFSET   36
#define RTU_TRI_FLAG_OPAQUE    0x1u

typedef struct {
  uint32_t status;
  float    hit_t;
  uint32_t sub_status;       // written by CHS dispatcher's sub-trace
  uint32_t pad;
} rtu_result_t;

typedef struct {
  uint64_t scene_addr;       // parent (primary) scene
  uint64_t sub_scene_addr;   // sub-scene used by CHS recursive trace
  uint64_t results_addr;
  uint64_t payload_addr;     // CHS writes sub_status here
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_RECURSIVE_COMMON_H_
